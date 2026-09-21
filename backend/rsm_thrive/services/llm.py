"""LLM backends behind one small interface.

`TritonAiLLM` is the working default — UCSD's TritonAI proxy
(https://tritonai-api.ucsd.edu/v1), OpenAI-compatible, model
`claude-sonnet-5` (live-verified 2026-09-11; ids drift — the portal's models page is the source of truth). Retries are same-model only (429/5xx), no
model fallback: TritonAI is the single sanctioned backend, so a
failure here is meant to surface honestly and feed the existing
degraded/503 paths rather than silently swap models. `FakeLLM` is
the test double.
"""

import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod

from django.conf import settings

logger = logging.getLogger("rsm_thrive.llm")

BASE_URL = "https://tritonai-api.ucsd.edu/v1"

# Models that refuse `temperature`. claude-sonnet-5 is the first: Vertex
# answers `temperature is deprecated for this model` with a 400, but ONLY
# when `response_format` is also present -- either parameter alone is fine,
# which is why this surfaced on the eight `json_mode=True` call sites (router,
# situation, the course advisor, jobs) and nowhere else.
#
# LEARNED AT RUNTIME, not hardcoded, for the reason the module docstring
# already gives: ids drift. A hardcoded list goes stale silently at the next
# rename and takes the main path down with it; this costs ONE wasted request
# per model per process and is then free, and a model that later starts
# accepting temperature again just never lands here.
_NO_TEMPERATURE = set()
_NO_TEMPERATURE_LOCK = threading.Lock()


def _rejects_temperature(model: str) -> bool:
    with _NO_TEMPERATURE_LOCK:
        return model in _NO_TEMPERATURE


def _remember_rejects_temperature(model: str) -> None:
    with _NO_TEMPERATURE_LOCK:
        if model not in _NO_TEMPERATURE:
            _NO_TEMPERATURE.add(model)
            logger.info("%s rejects temperature; dropping it for this process.",
                        model)


def _is_temperature_rejection(exc, status) -> bool:
    """A 400 naming `temperature` as the offending parameter.

    Deliberately narrow. The gateway is litellm in front of several providers,
    so the wording is the upstream provider's and not ours to predict -- but a
    400 that does not mention temperature is a real bad request and must keep
    raising, because retrying it without temperature would only bury it.

    `status` comes from the caller's `_status_of` rather than being read off
    the exception here, so the backend keeps ONE way of deciding what a status
    is and this cannot drift from the ladder it sits in.
    """
    if status != 400:
        return False
    return "temperature" in str(exc).lower()


def _log_before_raising(exc, status, kwargs) -> None:
    """Say what failed before it disappears into a caller's `except`.

    Every caller of `chat` swallows exceptions -- deliberately, because a
    classifier or a job score is not worth taking a student's turn down for.
    The cost is that a request WE built wrong looks exactly like a provider
    outage: `_model_route` answers "the language model could not be reached"
    either way, and nothing reaches a log. That is how a 400 on every
    `json_mode` call sat behind a plausible content refusal.

    So the distinction is drawn HERE, at the one place that still holds the
    status code, rather than at the six call sites that would each have to
    repeat it:

    * **4xx that is not 429** -- we sent something the provider will not
      accept. It is a defect in this code or in a model id, it is identical
      on every retry, and it will still be there tomorrow. ERROR, with the
      parameters we sent, because that is the bug report.
    * **anything else** -- a timeout, a 5xx, an exhausted retry ladder. The
      provider is having a bad day and the degraded path is the right answer.
      WARNING, because there is nothing to fix in here.
    """
    sent = sorted(k for k in kwargs if k != "messages")
    if status is not None and 400 <= status < 500 and status != 429:
        logger.error(
            "%s rejected our request (HTTP %s): %s -- parameters sent: %s. "
            "This is a malformed request, not an outage; retrying will not "
            "help and callers are about to degrade silently.",
            kwargs.get("model", "?"), status, exc, sent)
    else:
        logger.warning("%s call failed (HTTP %s): %s", kwargs.get("model", "?"),
                       status, exc)


class LLM(ABC):
    @abstractmethod
    def chat(self, system: str, messages: list, json_mode: bool = False) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}, ...] -> reply text."""

    def reader(self) -> "LLM":
        """The model the READING skills run on. Default: this one.

        A reading is a small JSON extraction -- which route, which track,
        how a campus page would word a question -- and a fast model does it
        as well as a large one. Measured 2026-09-20 on the TritonAI gateway:
        `read_question` took 5.5-7.5s on claude-sonnet-5 and 3.6-4.2s on
        gemini-3.5-flash with the same fields filled the same way -- but the
        planner's `read_turn` on the same model mapped an uncovered job onto
        a curated profile, so each skill opts in (`skills.read(fast=True)`).
        Only `TritonAiLLM` overrides this, and only when
        `TRITONAI_READER_MODEL` is set; answers always stay on the main model.
        """
        return self

    def search_chat(self, system: str, messages: list, json_mode: bool = False,
                    sources_out: list = None, search_query: str = None) -> str:
        """Like `chat`, but grounded in pages fetched now rather than in memory.

        Separate from `chat` rather than a flag on it, because it is a different
        promise: slower, and grounded in something outside this codebase.

        A backend with no search of its own keeps that promise here, by having
        `websearch` do the lookup and appending what it found to the system
        prompt. Only `CodexOAuthLLM` overrides this, because OpenAI hosts a real
        agent loop and a loop the provider runs beats one bolted on from
        outside. Everything else -- TritonAI included, which is what deployment
        runs -- arrives at this method, and before it existed arrived at a
        fallback that answered from training data while the prompt claimed
        current sources.

        `search_query` overrides what gets looked up. The default -- the last
        user turn -- is right when that turn IS the subject, as it is for
        `role_lookup`. It is wrong when the turn is a request wrapped around
        the subject: "what should I take if I want to work in esports?"
        searched verbatim returns Udemy and Coursera listings, because the
        student's own words are about COURSES, while the prompt those results
        ground asks what the field HIRES for. A caller that knows the subject
        says so.

        `sources_out`, when given a list, receives the `websearch.Result`s the
        answer was grounded in, so a caller can cite them. It stays EMPTY for a
        backend that searched natively: those pages are the provider's and we
        never see them, and an empty list is the honest way to say we cannot
        show our work. Search failing, or being switched off, is not an error --
        the answer degrades to the model's own knowledge, exactly as before.
        """
        from rsm_thrive.services import websearch

        results = websearch.search(search_query or websearch.query_from(messages))
        if not results:
            return self.chat(system, messages, json_mode)
        if sources_out is not None:
            sources_out.extend(results)
        return self.chat(websearch.ground(system, results), messages, json_mode)


class FakeLLM(LLM):
    """Scripted replies, recorded calls. The only LLM pytest ever runs.

    Thread-safe: the results page's parallel match-report scoring (see
    `feed._score_top_candidates_with_llm`) calls one `FakeLLM` instance from
    several worker threads at once in tests, so `calls` and `_replies` are
    guarded by a lock rather than assumed single-threaded.
    """

    def __init__(self, replies: list):
        self._replies = list(replies)
        self.calls = []
        self._lock = threading.Lock()

    def chat(self, system: str, messages: list, json_mode: bool = False) -> str:
        with self._lock:
            self.calls.append((system, messages, json_mode))
            if not self._replies:
                raise RuntimeError("FakeLLM exhausted")
            return self._replies.pop(0)


def parse_llm_json(text: str) -> dict:
    """Extract the assistant's JSON envelope from raw model output.

    Tolerates code fences, surrounding prose, and junk wrappers by scanning
    for any decodable JSON object and preferring one that has "reply". Falls
    back to treating the whole text as a plain chat reply so a malformed
    response degrades instead of crashing the turn.
    """
    candidates = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL) + [text]
    decoder = json.JSONDecoder()
    found = []
    for cand in candidates:
        try:
            out = json.loads(cand)
            if isinstance(out, dict):
                found.append(out)
                continue
        except (json.JSONDecodeError, ValueError):
            pass
        for i, ch in enumerate(cand):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(cand[i:])
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                found.append(obj)
    for obj in found:
        if "reply" in obj:
            return obj
    if found:
        return found[0]
    salvaged = salvage_json_reply(text)
    if salvaged is not None:
        return salvaged
    return {"reply": text, "action": "chat"}


_ANSWERED = re.compile(r'"answered"\s*:\s*(true|false)', re.I)
_REPLY_TAIL = re.compile(r'"reply"\s*:\s*"(.*)"\s*}\s*$', re.S)


def salvage_json_reply(text):
    """A {"answered": ..., "reply": "..."} object whose reply carries unescaped
    quotes, read anyway; or None.

    Measured live: the writer answered about orientation and quoted the
    "Meet Your MSBA Team" page inside its reply, which broke the JSON; the
    parser fell back to treating the whole thing as prose, and the student
    saw raw braces. The shape is fixed and the reply is the last value, so
    the text between `"reply": "` and the closing `"}` IS the reply, quotes
    and all.
    """
    text = str(text or "").strip()
    if not text.startswith("{") or '"reply"' not in text:
        return None
    match = _REPLY_TAIL.search(text)
    if not match:
        return None
    out = {"reply": match.group(1).replace('\\"', '"').replace("\\n", "\n").strip()}
    answered = _ANSWERED.search(text)
    if answered:
        out["answered"] = answered.group(1).lower() == "true"
    return out


class TritonAiLLM(LLM):
    """TritonAI backend (TRITONAI_API_KEY), OpenAI-compatible, with 429/5xx retries.

    Same-model retry only — deliberate, no fallback model list.
    """

    def __init__(self, api_key=None, model=None, sleep=time.sleep):
        from openai import OpenAI  # lazy: tests never import the SDK

        key = api_key or getattr(settings, "TRITONAI_API_KEY", "")
        if not key:
            raise RuntimeError("TRITONAI_API_KEY is not set.")
        self._model = model or getattr(settings, "TRITONAI_MODEL", "claude-sonnet-5")
        self._reader = None
        # A per-call ceiling, so one unresponsive socket cannot own the turn.
        # The SDK's own default is ten minutes, which is not a timeout for a
        # student waiting on a chat reply.
        self._client = OpenAI(
            api_key=key, base_url=BASE_URL,
            timeout=float(getattr(settings, "TRITONAI_TIMEOUT_SECONDS", 45)))
        self._sleep = sleep

    def _create(self, **kwargs):
        return self._client.chat.completions.create(**kwargs)

    def reader(self):
        wanted = getattr(settings, "TRITONAI_READER_MODEL", "") or self._model
        if wanted == self._model:
            return self
        if self._reader is None:
            self._reader = TritonAiLLM(model=wanted, sleep=self._sleep)
        return self._reader

    def _status_of(self, exc):
        return getattr(exc, "status_code", None)

    def chat(self, system: str, messages: list, json_mode: bool = False) -> str:
        full_messages = [{"role": "system", "content": system}] + list(messages)
        kwargs = {
            "model": self._model,
            "messages": full_messages,
            "max_tokens": 4000,
        }
        if not _rejects_temperature(self._model):
            kwargs["temperature"] = 0.4
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self._chat_with_retries(kwargs)

    def _chat_with_retries(self, kwargs) -> str:
        # A DEADLINE for the whole ladder, not just for each call.
        #
        # The per-attempt timeout bounds one request; it does not bound three
        # of them plus the 15s and 30s waits between. Measured during a real
        # connection blip on the sweep, one turn took 149 SECONDS -- long
        # after the student has decided the thing is broken, and long after
        # the point where saying "I can't reach the model" would have been the
        # better answer. The retries exist for a provider having a bad
        # second, not a bad minute.
        deadline = time.monotonic() + float(
            getattr(settings, "TRITONAI_TURN_BUDGET_SECONDS", 75))
        last_err = None
        attempt = 0
        while attempt < 3:
            try:
                resp = self._create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                status = self._status_of(e)
                # Dropping temperature does NOT spend an attempt. It is a
                # correction to a malformed request, not a retry of a failed
                # one, and it can only happen once per process per model --
                # letting it burn the last attempt would fail a request whose
                # fix we already hold.
                if _is_temperature_rejection(e, status) and "temperature" in kwargs:
                    _remember_rejects_temperature(kwargs["model"])
                    kwargs.pop("temperature")
                    continue
                attempt += 1
                # Only wait if there is room to wait AND to retry afterwards.
                remaining = deadline - time.monotonic()
                if status == 429 and attempt < 3 and remaining > 15 * attempt:
                    self._sleep(15 * attempt)
                    continue
                if (status in (500, 502, 503) and attempt < 3
                        and remaining > 3 * attempt):
                    self._sleep(3 * attempt)
                    continue
                _log_before_raising(e, status, kwargs)
                raise
        _log_before_raising(last_err, self._status_of(last_err), kwargs)
        raise last_err


class SharedCodexCredentials:
    """One OAuth credential, shared by every request on the server.

    On a laptop each developer authenticates as themselves and the vendored
    script's own load-refresh-or-login is exactly right. A hosted TEST instance
    is different: one credential file serves the whole cohort, nobody signs in,
    and three things that never matter for one person start mattering.

    **Refreshes must not race.** `ensure_openai_codex_credentials()` runs on
    every turn, and OpenAI's refresh tokens are single-use -- the vendored
    script recognises `refresh_token_reused` and `already been used` for that
    reason. Two concurrent turns arriving after expiry would both refresh, and
    the loser's token would be dead for EVERYONE. One lock, one refresh.

    **A server must never try to log in.** On failure the script falls back to
    `login_openai_codex_oauth()`, which calls `webbrowser.open()` and then
    blocks on `input()`. On a headless box that hangs the worker for good. Here
    interactive login is allowed only when someone is actually at a terminal,
    and otherwise raises something a log will explain.

    **Re-reading and re-parsing a file every turn is waste.** The credential is
    held in memory and only touched again near expiry.

    Provision the server by copying a developer's `~/.codex/auth.json` to the
    box and pointing `CODEX_CREDENTIALS_PATH` at it. Treat that file as a
    secret: it carries a long-lived refresh token for a personal ChatGPT
    account, and anyone who can read it can use that account.
    """

    _lock = threading.Lock()
    _cached = None

    # Refresh this far before the token actually expires, so a request never
    # races the expiry itself.
    MARGIN_MS = 5 * 60 * 1000

    @classmethod
    def _path(cls):
        configured = (getattr(settings, "CODEX_CREDENTIALS_PATH", "") or "").strip()
        return configured or None

    @classmethod
    def _interactive_allowed(cls):
        """Only when a human is present to complete a browser sign-in."""
        explicit = getattr(settings, "CODEX_ALLOW_BROWSER_LOGIN", None)
        if explicit is not None:
            return bool(explicit)
        try:
            import sys
            return sys.stdin is not None and sys.stdin.isatty()
        except Exception:
            return False

    @classmethod
    def get(cls):
        from rsm_thrive.services import codex_oauth

        now_ms = int(time.time() * 1000)
        cached = cls._cached
        if cached and cached.expires > now_ms + cls.MARGIN_MS:
            return cached

        with cls._lock:
            # Re-check inside the lock: another thread may have refreshed while
            # this one waited, and refreshing again would burn a single-use
            # token for no reason.
            cached = cls._cached
            now_ms = int(time.time() * 1000)
            if cached and cached.expires > now_ms + cls.MARGIN_MS:
                return cached

            path = cls._path()
            current = codex_oauth.load_openai_codex_credentials(path)
            if current is None:
                current = codex_oauth._load_codex_cli_credentials()

            if current is None:
                if cls._interactive_allowed():
                    current = codex_oauth.login_openai_codex_oauth()
                    codex_oauth.save_openai_codex_credentials(current, path)
                else:
                    raise RuntimeError(
                        "No Codex credentials on this host and no terminal to "
                        "sign in from. Copy a developer's ~/.codex/auth.json to "
                        "the server and set CODEX_CREDENTIALS_PATH to it.")
            elif current.expires <= now_ms + cls.MARGIN_MS:
                try:
                    current = codex_oauth.refresh_openai_codex_token(current.refresh)
                    codex_oauth.save_openai_codex_credentials(current, path)
                    logger.info("codex credentials refreshed")
                except RuntimeError as error:
                    # A refresh token that is expired or already spent cannot be
                    # recovered without a person. Say so plainly rather than
                    # letting the vendored fallback open a browser nobody sees.
                    if not cls._interactive_allowed():
                        raise RuntimeError(
                            "Codex credentials could not be refreshed and this "
                            "host cannot sign in interactively. Re-authenticate "
                            "on a workstation and copy the credential file over. "
                            f"Underlying error: {error}") from error
                    current = codex_oauth.login_openai_codex_oauth()
                    codex_oauth.save_openai_codex_credentials(current, path)

            cls._cached = current
            return current

    @classmethod
    def reset(cls):
        """Drop the in-memory copy. For tests and for a credential swap."""
        with cls._lock:
            cls._cached = None


class CodexOAuthLLM(LLM):
    """LOCAL DEVELOPMENT ONLY: a personal ChatGPT/Codex subscription over OAuth.

    Exists because the TritonAI keys this project runs on carry a fixed one-time
    budget with no reset, so an exhausted key stops all local work on the
    chatbots. This backend needs no API key at all -- it reuses the Codex CLI's
    stored credentials (`~/.codex/auth.json`), refreshing them as needed, and
    falls back to a browser sign-in if there are none.

    NOT a production backend, and deliberately not the default. It authenticates
    as a person rather than as a metered service, it talks to an endpoint that
    carries no compatibility promise, and the deployed service must stay on
    `TritonAiLLM`. `THRIVE_LLM=codex` is a laptop switch.

    It provides COMPLETIONS ONLY -- no embeddings, and its token reaches
    api.openai.com but the account has no API billing, so `/v1/embeddings`
    answers 429. Retrieval therefore needs a provider of its own, which is why
    `get_embeddings()` reads `THRIVE_EMBEDDINGS` separately and resolves this
    backend to `LocalEmbeddings` (on-machine, keyless). With that in place the
    FAQ bot scores 6/9 on the golden eval -- the same as the documented
    TritonAI baseline -- so nothing here is degraded by the switch.
    """

    # The interview is extraction and short prose, not reasoning work. Leaving
    # effort unset keeps turns fast; the default of "high" made each extraction
    # take tens of seconds for no gain in answer quality.
    def __init__(self, model=None, reasoning_effort=None):
        self._model = model or getattr(settings, "CODEX_MODEL", "gpt-5.4")
        self._effort = reasoning_effort

    def search_chat(self, system: str, messages: list, json_mode: bool = False,
                    sources_out: list = None, search_query: str = None) -> str:
        """Native search: OpenAI runs the loop, so `websearch` is not used here.

        `sources_out` is accepted and left untouched. The pages behind a hosted
        search are never handed back, so there is nothing truthful to put in it.
        """
        return self.chat(system, messages, json_mode, search=True)

    def chat(self, system: str, messages: list, json_mode: bool = False,
             search: bool = False) -> str:
        from rsm_thrive.services import codex_oauth

        instructions = system
        if json_mode:
            # The Responses API's structured-output field is not relied on here:
            # `parse_llm_json` already tolerates fences and surrounding prose, so
            # an instruction plus that parser is the combination that cannot
            # fail closed on an endpoint whose schema support is undocumented.
            instructions = (f"{system}\n\nReturn ONLY a single JSON object. "
                            "No prose, no explanation, no code fences.")
        credentials = SharedCodexCredentials.get()
        return self._with_retries(credentials, list(messages), instructions, search)

    # One retry, on TRANSIENT failures only. `TritonAiLLM` has had this from the
    # start and this backend went without, which showed up as soon as it was
    # exercised at volume: one probe in 120 died on "read operation timed out"
    # -- a flaky socket, not a bad request. The view would turn that into the
    # degraded reply, so a student loses a turn to something a second attempt
    # answers. Not retried: anything that will fail again the same way.
    _TRANSIENT = ("timed out", "timeout", "connection reset", "connection aborted",
                  "temporarily unavailable", "bad gateway", "502", "503", "504")

    def _with_retries(self, credentials, messages, instructions, search=False):
        from rsm_thrive.services import codex_oauth

        last = None
        for attempt in range(2):
            try:
                return self._create(codex_oauth, credentials, messages, instructions,
                                    search)
            except Exception as error:
                last = error
                text = str(error).lower()
                if attempt == 0 and any(s in text for s in self._TRANSIENT):
                    logger.warning("codex turn failed transiently, retrying: %s",
                                   str(error)[:120])
                    continue
                raise
        raise last

    def _create(self, codex_oauth, credentials, messages, instructions, search=False):
        return codex_oauth.create_codex_chat(
            messages, credentials=credentials, model=self._model,
            instructions=instructions, reasoning_effort=self._effort, search=search,
            # Well above a normal turn (~2.5s) and well below anything a user
            # or a batch run should wait on. See the deadline in
            # `create_codex_chat` for why the module default is not enough.
            timeout=float(getattr(settings, "CODEX_TIMEOUT_SECONDS", 90)))


def get_llm() -> LLM:
    """The configured backend. Views hold this behind an injectable seam."""
    backend = getattr(settings, "THRIVE_LLM", "tritonai")
    if backend == "fake":
        raise RuntimeError(
            "THRIVE_LLM=fake requires the test to inject a FakeLLM explicitly.")
    if backend == "codex":
        return CodexOAuthLLM()
    return TritonAiLLM()
