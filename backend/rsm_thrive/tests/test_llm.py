import json
import logging
import types

import pytest

from rsm_thrive.services.bot_config import bot_config, load_bot_config
from rsm_thrive.services import llm as llm_module
from rsm_thrive.services.llm import FakeLLM, TritonAiLLM, parse_llm_json


class TestParseLlmJson:
    def test_plain_json(self):
        assert parse_llm_json('{"reply": "hi", "action": "chat"}') == {
            "reply": "hi", "action": "chat"}

    def test_fenced_json(self):
        text = 'Sure!\n```json\n{"reply": "ok"}\n```\nDone.'
        assert parse_llm_json(text)["reply"] == "ok"

    def test_junk_wrapped_json(self):
        text = '{\\text{ {"reply": "inner"} }}'
        assert parse_llm_json(text)["reply"] == "inner"

    def test_prefers_envelope_with_reply(self):
        text = '{"other": 1} and then {"reply": "the one"}'
        assert parse_llm_json(text)["reply"] == "the one"

    def test_garbage_falls_back_to_chat_reply(self):
        out = parse_llm_json("just prose, no json at all")
        assert out == {"reply": "just prose, no json at all", "action": "chat"}


class TestFakeLLM:
    def test_pops_replies_in_order_and_records_calls(self):
        fake = FakeLLM(replies=["one", "two"])
        assert fake.chat("sys", [{"role": "user", "content": "q"}]) == "one"
        assert fake.chat("sys", [], json_mode=True) == "two"
        assert fake.calls[0] == ("sys", [{"role": "user", "content": "q"}], False)
        assert fake.calls[1][2] is True

    def test_exhaustion_raises(self):
        fake = FakeLLM(replies=[])
        with pytest.raises(RuntimeError):
            fake.chat("sys", [])


class TestTritonRetries:
    """TritonAiLLM's retry ladder (same-model only, no fallback), SDK faked out."""

    def _client(self, script, sleeps):
        """script: list of ('ok', text) | ('err', status_code) consumed per API call."""
        client = TritonAiLLM.__new__(TritonAiLLM)
        client._model = "claude-opus-4-6-v1"
        client._sleep = sleeps.append  # records requested waits, no real sleep

        class FakeAPIError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code

        def create(**kwargs):
            kind, value = script.pop(0)
            if kind == "err":
                raise FakeAPIError(value)
            message = types.SimpleNamespace(content=value)
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

        client._create = create
        client._status_of = lambda e: getattr(e, "status_code", None)
        return client

    def test_retries_503_then_succeeds(self):
        sleeps = []
        client = self._client([("err", 503), ("ok", "answer")], sleeps)
        assert client._chat_with_retries({}) == "answer"
        assert sleeps == [3]

    def test_429_waits_15(self):
        sleeps = []
        client = self._client([("err", 429), ("ok", "x")], sleeps)
        client._chat_with_retries({})
        assert sleeps == [15]

    def test_non_retryable_raises_immediately_without_sleeping(self):
        sleeps = []
        client = self._client([("err", 400)], sleeps)
        with pytest.raises(Exception):
            client._chat_with_retries({})
        assert sleeps == []

    def test_third_consecutive_503_raises(self):
        sleeps = []
        client = self._client(
            [("err", 503), ("err", 503), ("err", 503)], sleeps)
        with pytest.raises(Exception):
            client._chat_with_retries({})
        assert sleeps == [3, 6]


class TestTemperatureRejection:
    """claude-sonnet-5 rejects `temperature` alongside `response_format`.

    Vertex answers 400 `temperature is deprecated for this model`. The model
    is not named in the code -- the rejection is learned from the first 400
    and remembered for the process -- so these tests drive it through that
    discovery rather than asserting on a hardcoded list.
    """

    @pytest.fixture(autouse=True)
    def _clear_learned_models(self):
        llm_module._NO_TEMPERATURE.clear()
        yield
        llm_module._NO_TEMPERATURE.clear()

    def _client(self, script, sleeps, model="claude-sonnet-5"):
        """script: ('ok', text) | ('err', status, message), plus seen kwargs."""
        client = TritonAiLLM.__new__(TritonAiLLM)
        client._model = model
        client._sleep = sleeps.append
        seen = []

        class FakeAPIError(Exception):
            def __init__(self, status_code, message):
                super().__init__(message)
                self.status_code = status_code

        def create(**kwargs):
            seen.append(kwargs)
            entry = script.pop(0)
            if entry[0] == "err":
                raise FakeAPIError(entry[1], entry[2])
            message = types.SimpleNamespace(content=entry[1])
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message)])

        client._create = create
        client._status_of = lambda e: getattr(e, "status_code", None)
        return client, seen

    DEPRECATED = "`temperature` is deprecated for this model."

    def test_drops_temperature_and_succeeds(self):
        sleeps = []
        client, seen = self._client(
            [("err", 400, self.DEPRECATED), ("ok", '{"reply": "hi"}')], sleeps)
        out = client._chat_with_retries(
            {"model": "claude-sonnet-5", "temperature": 0.4, "messages": []})
        assert out == '{"reply": "hi"}'
        assert "temperature" in seen[0]
        assert "temperature" not in seen[1]
        assert sleeps == [], "a malformed parameter is corrected, not waited out"

    def test_rejection_is_remembered_so_later_calls_never_send_it(self):
        sleeps = []
        client, _ = self._client(
            [("err", 400, self.DEPRECATED), ("ok", "x")], sleeps)
        client._chat_with_retries(
            {"model": "claude-sonnet-5", "temperature": 0.4, "messages": []})

        client2, seen2 = self._client([("ok", "y")], sleeps)
        client2.chat("sys", [{"role": "user", "content": "hi"}], json_mode=True)
        assert "temperature" not in seen2[0]
        assert seen2[0]["response_format"] == {"type": "json_object"}

    def test_unaffected_model_still_sends_temperature(self):
        sleeps = []
        client, seen = self._client([("ok", "x")], sleeps, model="gemini-3.5-flash")
        client.chat("sys", [{"role": "user", "content": "hi"}], json_mode=True)
        assert seen[0]["temperature"] == 0.4

    def test_other_400s_still_raise(self):
        sleeps = []
        client, _ = self._client(
            [("err", 400, "Unsupported parameter: 'messages'.")], sleeps)
        with pytest.raises(Exception):
            client._chat_with_retries(
                {"model": "gpt-5.4", "temperature": 0.4, "messages": []})
        assert llm_module._NO_TEMPERATURE == set()
        assert sleeps == []

    def test_drop_does_not_spend_the_retry_budget(self):
        """A 400 correction plus a full 503 ladder still gets all three tries."""
        sleeps = []
        client, seen = self._client(
            [("err", 400, self.DEPRECATED), ("err", 503, "down"),
             ("err", 503, "down"), ("ok", "recovered")], sleeps)
        out = client._chat_with_retries(
            {"model": "claude-sonnet-5", "temperature": 0.4, "messages": []})
        assert out == "recovered"
        assert sleeps == [3, 6]
        assert len(seen) == 4


class TestATurnCannotRunAwayWithTheClock:
    """The per-call timeout bounds one request; it does not bound three of
    them plus the 15s and 30s waits between. Measured on a real connection
    blip during the conversation sweep, one turn took 149 SECONDS -- long
    after the student had decided the thing was broken, and long after
    "I can't reach the model" became the better answer."""

    def _client(self, status, sleeps, clock):
        client = TritonAiLLM.__new__(TritonAiLLM)
        client._model = "claude-sonnet-5"

        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds        # waiting spends the budget

        client._sleep = sleep

        class FakeAPIError(Exception):
            def __init__(self):
                super().__init__("upstream unavailable")
                self.status_code = status

        def create(**kwargs):
            clock[0] += 45             # a call that runs to its own timeout
            raise FakeAPIError()

        client._create = create
        client._status_of = lambda e: getattr(e, "status_code", None)
        return client

    def test_the_ladder_stops_when_the_budget_is_gone(self, monkeypatch):
        clock = [0.0]
        monkeypatch.setattr(llm_module.time, "monotonic", lambda: clock[0])
        sleeps = []
        client = self._client(503, sleeps, clock)
        with pytest.raises(Exception):
            client._chat_with_retries({"model": "claude-sonnet-5", "messages": []})
        assert clock[0] <= 200, f"turn ran {clock[0]}s"
        assert sleeps != [3, 6], "it waited the full ladder past the budget"

    def test_a_healthy_provider_still_gets_its_retries(self, monkeypatch):
        """The budget must not cost a retry when the calls are fast."""
        clock = [0.0]
        monkeypatch.setattr(llm_module.time, "monotonic", lambda: clock[0])
        sleeps = []
        client = TritonAiLLM.__new__(TritonAiLLM)
        client._model = "m"
        client._sleep = sleeps.append
        script = [("err",), ("err",), ("ok",)]

        class Err(Exception):
            status_code = 503

        def create(**kwargs):
            if script.pop(0)[0] == "err":
                raise Err()
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="recovered"))])

        client._create = create
        client._status_of = lambda e: getattr(e, "status_code", None)
        assert client._chat_with_retries(
            {"model": "m", "messages": []}) == "recovered"
        assert sleeps == [3, 6]


class TestFailuresAreLogged:
    """A swallowed exception must still leave a trace at the boundary.

    Every `chat` caller catches broadly on purpose, so the log line is the
    only thing standing between a malformed request and a silent degrade.
    """

    def _client(self, status, message, sleeps):
        client = TritonAiLLM.__new__(TritonAiLLM)
        client._model = "claude-sonnet-5"
        client._sleep = sleeps.append

        class FakeAPIError(Exception):
            def __init__(self, status_code, msg):
                super().__init__(msg)
                self.status_code = status_code

        def create(**kwargs):
            raise FakeAPIError(status, message)

        client._create = create
        client._status_of = lambda e: getattr(e, "status_code", None)
        return client

    def _kwargs(self):
        return {"model": "claude-sonnet-5", "messages": [], "max_tokens": 4000,
                "response_format": {"type": "json_object"}}

    def test_malformed_request_logs_error_with_parameters(self, caplog):
        client = self._client(400, "Unsupported parameter: 'response_format'.", [])
        with caplog.at_level(logging.ERROR, logger="rsm_thrive.llm"):
            with pytest.raises(Exception):
                client._chat_with_retries(self._kwargs())
        records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(records) == 1
        text = records[0].getMessage()
        assert "malformed request" in text
        # The parameters we sent are the bug report; the prompt is not.
        assert "response_format" in text
        assert "messages" not in text

    def test_outage_logs_warning_not_error(self, caplog):
        sleeps = []
        client = self._client(503, "upstream down", sleeps)
        with caplog.at_level(logging.DEBUG, logger="rsm_thrive.llm"):
            with pytest.raises(Exception):
                client._chat_with_retries(self._kwargs())
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR], \
            "a provider outage is not this codebase's bug"
        assert [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_429_is_not_reported_as_malformed(self, caplog):
        sleeps = []
        client = self._client(429, "budget_exceeded", sleeps)
        with caplog.at_level(logging.DEBUG, logger="rsm_thrive.llm"):
            with pytest.raises(Exception):
                client._chat_with_retries(self._kwargs())
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_temperature_rejection_is_corrected_not_logged_as_a_defect(self, caplog):
        """The one 400 we handle ourselves must not cry wolf."""
        llm_module._NO_TEMPERATURE.clear()
        client = TritonAiLLM.__new__(TritonAiLLM)
        client._model = "claude-sonnet-5"
        client._sleep = lambda _: None
        script = [("err",), ("ok",)]

        class FakeAPIError(Exception):
            def __init__(self):
                super().__init__("`temperature` is deprecated for this model.")
                self.status_code = 400

        def create(**kwargs):
            if script.pop(0)[0] == "err":
                raise FakeAPIError()
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content="ok"))])

        client._create = create
        client._status_of = lambda e: getattr(e, "status_code", None)
        with caplog.at_level(logging.DEBUG, logger="rsm_thrive.llm"):
            assert client._chat_with_retries(
                {"model": "claude-sonnet-5", "temperature": 0.4, "messages": []}) == "ok"
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        llm_module._NO_TEMPERATURE.clear()


class TestBotConfig:
    def test_defaults_load_and_have_required_keys(self):
        config = load_bot_config()
        assert set(config) >= {"faq", "electives", "career"}
        for bot in ("faq", "electives", "career"):
            entry = bot_config(bot)
            assert isinstance(entry["top_k"], int)
            assert 0.0 <= entry["min_similarity"] <= 1.0
        # Retrieval gates only, for every bot. The prompts are skill files
        # beside the bot that owns them (`services/*/skills/*/SKILL.md`),
        # deliberately out of reach of a runtime overlay.
        for bot in ("faq", "career", "electives"):
            assert "system_prompt" not in bot_config(bot)
        assert "refusal_reply" in bot_config("faq")

    def test_override_path(self, tmp_path, settings):
        override = {"faq": {"top_k": 99}}
        path = tmp_path / "bots.json"
        path.write_text(json.dumps(override))
        settings.THRIVE_BOT_CONFIG = str(path)
        assert bot_config("faq")["top_k"] == 99
        # unspecified keys fall through to the repo defaults
        assert bot_config("faq")["refusal_reply"]
