"""The Resources (FAQ) bot and the career bot.

A search engine with a reader in front and a writer behind. Every question is
READ first (`read_question`): is it about their studies, what is it really
asking, how would a campus page word the answer, does it turn on something
unsaid, what would a campus-scoped web search look for. The corpus is then
searched with the student's words and the campus wordings, what it admits is
re-ranked so the programme's own pages about the question come first, and
ONE writer (`answer_question`) is shown the passages -- and, when the corpus
looks thin, web results retrieved just now -- each kind marked so the reply
can say where every claim came from. What it used decides how the reply is
labelled: programme material is cited as such; a web claim carries the
web-search notice and its links.

The course planner lives in `services/courses/`; it shares only `BotReply`
and the grounding helpers in `services/reply.py` with this module.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rsm_thrive.services import skills, websearch
from rsm_thrive.services.bot_config import bot_config
from rsm_thrive.services.llm import parse_llm_json
from rsm_thrive.services.reply import BotReply, append_sources, build_context
from rsm_thrive.services.resources.retrieval import retrieve

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
READ_QUESTION = "read_question"
ANSWER_QUESTION = "answer_question"
CAREER_ADVICE = "career_advice"


def _trimmed(history, config):
    return history[-config["max_history_turns"]:]


def _retrieve(query, config, destination="resources"):
    return retrieve(query, destination, config["top_k"],
                    config["min_similarity"], config.get("lexical_min"),
                    config.get("lexical_floor", 0.0))


def _merge(*hit_lists):
    """Hits from several retrievals, best score per chunk, best first."""
    best = {}
    for hits in hit_lists:
        for chunk, score in hits or []:
            if chunk.pk not in best or score > best[chunk.pk][1]:
                best[chunk.pk] = (chunk, score)
    return sorted(best.values(), key=lambda pair: -pair[1])


def _refusal(config):
    """Nothing here answers it: the corpus-miss refusal, policy-flavoured."""
    return BotReply(config["refusal_reply"], [], "refusal", refused=True)


def _out_of_scope(config):
    """Not a question for this bot at all. Said as that, not as a policy
    hedge -- "I'd rather not guess about a policy" was the reply to "who
    won the super bowl"."""
    return BotReply(config.get("out_of_scope_reply") or config["refusal_reply"],
                    [], "refusal", refused=True)


# ---------------------------------------------------------------------------
# Ranking what retrieval admitted
# ---------------------------------------------------------------------------
#
# The on-machine embedding model ranks loosely: for "how to register for
# classes" it put "Student Services" and "Network Connections" above
# "Graduate Enrollment" and the MSBA cohort's own RadyReady page, and gave
# four of ten slots to one catalog page. Three deterministic corrections,
# applied after the cosine tiers have decided what is IN, only to decide the
# ORDER and the mix. The reported score stays the cosine, for the logs.

# Programme material beats campus material when both match: the MSBA cohort's
# Canvas pages and Rady's own site say what THIS student must do.
PROGRAMME_BOOST = 0.15
# A document whose title or section heading shares a content word with the
# question is about the question; a chunk that merely contains the words may
# not be. Programme material is boosted only on top of this.
TITLE_BOOST = 0.10
# One page may not fill the prompt.
MAX_CHUNKS_PER_DOCUMENT = 3

_STOPWORDS = frozenset("""
about after again also and any are because been before being between both
but can could does doing down during each even for from further has have
having here how into its just like more most much not now off once only other
our out over own same should some such than that the their them then there
these they this those through too under until very was were what when where
which while who whom why will with would you your ucsd diego san uc
""".split())


def _stems(*texts):
    """Five-letter stems of the content words in `texts`.

    "register", "registering" and "Registration" share "regis"; "enroll",
    "enrolled" and "Enrollment" share "enrol". Crude, and exactly as much
    stemming as a title match needs.
    """
    out = set()
    for text in texts:
        for word in re.findall(r"[a-z]{4,}", str(text or "").lower()):
            if word not in _STOPWORDS:
                out.add(word[:5])
    return out


def _programme_specific(document):
    url = (document.source_url or "").lower()
    title = (document.title or "").lower()
    return ("rady" in url or "rady" in title or "msba" in title)


def _boilerplate(chunk):
    """The provenance chunk every ingested page starts with. No content.

    Crawled pages open "Source: <url> Fetched: ..."; hand-captured Canvas
    pages open "Source: <url> (official Rady Canvas ...) Captured ...". Both
    are the pipeline talking about the page, not the page.
    """
    return (chunk.text or "").lstrip().startswith("Source: http")


def _rerank(hits, question, read, top_k):
    """The passages the model is shown: re-ordered, diversified, de-noised."""
    wanted = _stems(question, read.get("topic"), *(read.get("search_terms") or []))
    scored = []
    for chunk, cosine in hits:
        if _boilerplate(chunk):
            continue
        document = chunk.document
        adjusted = cosine
        about_it = bool(wanted & _stems(document.title, chunk.heading))
        # Programme material wins only when it is ABOUT the question. An
        # unconditional boost put the cohort's laptop guidelines and IT
        # overview above "Graduate Enrollment" for a registration question,
        # because every MSBA page carries the same tag.
        if about_it and _programme_specific(document):
            adjusted += PROGRAMME_BOOST
        if about_it:
            adjusted += TITLE_BOOST
        scored.append((adjusted, cosine, chunk))
    scored.sort(key=lambda row: -row[0])
    out, per_document = [], {}
    for _adjusted, cosine, chunk in scored:
        key = chunk.document_id
        if per_document.get(key, 0) >= MAX_CHUNKS_PER_DOCUMENT:
            continue
        per_document[key] = per_document.get(key, 0) + 1
        out.append((chunk, cosine))
        if len(out) >= top_k:
            break
    return out


_CITATION = re.compile(r"\[(\d{1,2})\]")


def _cited(body, hits):
    """The hits the reply actually cited by number, in prompt order.

    The passages are numbered [1]..[n] in the prompt in the order of `hits`,
    so a reply's [3] is hits[2]. Listing every retrieved document under an
    answer that used two of them put "UC San Diego PRONTO FAQ" under an
    answer about registering for classes. Nothing cited means the model did
    not use the numbers, and then every hit is listed as before.
    """
    numbers = {int(n) for n in _CITATION.findall(body or "")}
    chosen = [hits[n - 1] for n in sorted(numbers) if 1 <= n <= len(hits)]
    return chosen or hits


def _transcript(history, limit=6):
    """The last few turns, for the reading skill's context."""
    lines = []
    for message in list(history or [])[-limit:]:
        who = "student" if message.get("role") == "user" else "THRIVE"
        text = " ".join(str(message.get("content") or "").split())[:240]
        if text:
            lines.append(f"{who}: {text}")
    return " / ".join(lines)


def _read_question(llm, question, history):
    """What the `read_question` skill made of a question the corpus missed.

    Every field is checked: flags are flags, terms are short strings and at
    most three, and a term identical to the question is dropped because it
    has already been tried. {} when the model could not be reached or said
    nothing usable, which the caller treats as "carry on as before".
    """
    reading = skills.read(SKILLS_DIR, READ_QUESTION, llm, question,
                          {"conversation so far": _transcript(history)},
                          fast=True)
    raw = reading.raw
    if reading.degraded or not raw:
        return {}
    out = {}
    if "in_scope" in raw:
        out["in_scope"] = skills.flag_of(raw.get("in_scope"))
    asked = " ".join(str(question or "").lower().split())
    terms = []
    for term in (raw.get("search_terms") or [])[:3] if isinstance(
            raw.get("search_terms"), list) else []:
        term = skills.words_of(term, 80)
        if term and term.lower() != asked and term not in terms:
            terms.append(term)
    out["search_terms"] = terms
    out["needs_context"] = skills.flag_of(raw.get("needs_context"))
    out["clarifying_question"] = skills.words_of(raw.get("clarifying_question"), 300)
    out["web_query"] = skills.words_of(raw.get("web_query"), 200)
    out["topic"] = skills.words_of(raw.get("topic"), 60)
    return out


# ---------------------------------------------------------------------------
# Writing the answer
# ---------------------------------------------------------------------------

WEB_DISCLAIMER = (
    "_Part of this comes from a web search of public pages rather than "
    "program documentation -- the parts marked W. Please verify anything you "
    "act on, and for anything about policy, deadlines, fees or approvals, "
    "confirm with MSBA advising (bookable from the **Appointments** tab)._")
WEB_ONLY_DISCLAIMER = (
    "_Rady's own material doesn't cover this, so the above comes from a web "
    "search rather than program documentation. Please verify anything you act "
    "on -- and for anything about policy, deadlines, fees or approvals, confirm "
    "with MSBA advising (bookable from the **Appointments** tab)._")

# One marker, or a run of them the writer packed into one bracket: [W1],
# [W 2], [W1, W3], [W1-W2]. Every number inside is a citation.
_WEB_CITATION = re.compile(r"\[W\s?\d{1,2}(?:\s*[-–,]\s*W?\s?\d{1,2})*\]")
_WEB_NUMBER = re.compile(r"\d{1,2}")

PARTIAL_LEAD = ("_I couldn't find a definitive answer in the material I have, so "
                "treat this as a starting point and confirm the specifics with "
                "MSBA advising._\n\n")


def _web_block(results):
    if not results:
        return "Web results: none were retrieved for this question."
    lines = [f"[W{i}] {r.title}\n{r.url}\n{r.snippet}" for i, r in enumerate(results, 1)]
    return "Web results, retrieved just now:\n\n" + "\n\n".join(lines)


def _web_sources(results, body):
    """The web results the reply cited, as links -- or all of them when the
    reply used the web without numbering. "" when there were none."""
    if not results:
        return ""
    numbers = {int(n) for marker in _WEB_CITATION.findall(body or "")
               for n in _WEB_NUMBER.findall(marker)}
    chosen = [results[n - 1] for n in sorted(numbers) if 1 <= n <= len(results)]
    listed = [r for r in (chosen or results) if getattr(r, "url", "")][:3]
    if not listed:
        return ""
    return "\n\nRead from the web just now:\n" + "\n".join(
        f"- [W{results.index(r) + 1}] [{(r.title or r.url)[:80]}]({r.url})" for r in listed)


def _write(llm, config, question, history, hits, results, final=False, topic=""):
    """One writer, shown whatever grounding there is; the reply labelled by
    what it actually used. None when the material did not answer.

    The passages are numbered [1]..[n] and the web results [W1]..[Wm] in the
    prompt, and the reply cites by marker. A passage marker means programme
    material was used, so its documents are listed as sources; a W marker
    means the web was used, so the web notice and the pages read are
    appended. Both may appear, and then both are shown -- which is the
    honest rendering of an answer that took the steps from Rady's page and
    a current detail from a public one.

    Everything that reads the database (`build_context`, `append_sources`)
    runs on the caller's thread; only the model call may be handed to a
    worker.
    """
    system = skills.load(SKILLS_DIR, ANSWER_QUESTION).prompt(
        passages=(f"Passages from Rady's and UC San Diego's own pages:\n\n"
                  f"{build_context(hits)}" if hits
                  else "Passages: none matched this question."),
        web=_web_block(results))
    asked = question
    if topic and history:
        # A one-word follow-up ("canvas", "to drop a class") answers the
        # clarifying question before it; the reading combined the two into
        # `topic`, and the writer needs the combination, not the fragment.
        # Measured live: "when is the deadline" -> "to drop a class" was
        # answered as though "to drop a class" were the whole question.
        asked = f"{question}\n\n(Understood as: {topic})"
    messages = _trimmed(history, config) + [{"role": "user", "content": asked}]
    try:
        if not results and not hits:
            # Nothing retrieved here: a backend that searches natively (Codex)
            # may still ground the answer itself; every other backend answers
            # from what it knows, and the notice below says so.
            raw = llm.search_chat(system, messages, True)
        else:
            raw = llm.chat(system, messages, json_mode=True)
    except Exception:
        # A failed call is not an answer. The caller refuses, plainly.
        return None
    parsed = parse_llm_json(raw or "")
    body = str(parsed.get("reply") or "").strip()
    if not body:
        return None
    if parsed.get("answered") is False and not final:
        # Declined with strong passages: the caller brings the web results
        # and asks again. Only that LAST attempt shows a declined reply.
        return None
    if parsed.get("answered") is False:
        # The last attempt, and the material still did not settle it. What
        # the writer found is still worth more than a canned refusal to a
        # question that IS about the programme -- measured live: "when is
        # tuition due" and "when can i get grades" both got "I'd rather not
        # guess about a policy" after two declines. The skill tells the
        # writer to put the closest useful material and what to confirm
        # in `reply` even when it declines; labelled as partial here.
        body = PARTIAL_LEAD + body
    if not results:
        # Told "Web results: none", the writer still occasionally writes a
        # [W1]. A marker that points at nothing is removed rather than shown.
        body = re.sub(r"\s?" + _WEB_CITATION.pattern, "", body)
    cites_passages = bool(_CITATION.search(body))
    cites_web = bool(_WEB_CITATION.search(body))
    if hits and results:
        # Shown both: what it cited decides, and citing neither means both.
        used_passages = cites_passages or not cites_web
        used_web = cites_web or not cites_passages
    else:
        used_passages, used_web = bool(hits), not hits
    cited_hits = _cited(body, hits) if used_passages else []
    if used_passages:
        body = append_sources(body, cited_hits)
    if used_web:
        body += _web_sources(results, body)
        body += "\n\n" + (WEB_DISCLAIMER if used_passages else WEB_ONLY_DISCLAIMER)
    note = ("llm+web" if used_passages and used_web
            else "llm" if used_passages else "web")
    return BotReply(body, [chunk.pk for chunk, _ in cited_hits], note)


def answer_faq(llm, question, history):
    """Read the question, then follow the steps the reading sets.

    1. **Out of scope** -- trivia, sport, someone's homework -- is refused
       without a search.
    2. **The corpus**, searched with the student's own words, the campus
       wordings and the topic from the reading, merged and re-ranked.
    3. **A question back**, only when nothing was found AND the reading says
       the answer would differ with something the student has not said.
    4. **The writer**, shown the passages -- and, when the best passage is
       thin (`web_race_below`) or there are none, web results for the
       campus-scoped query retrieved just now -- in ONE call, citing each by
       its marker. A writer that declines with strong passages gets one more
       call with the web results added.

    A reading that fails leaves the old flow intact, and `web_fallback:
    false` keeps the spec's original guarantee -- a corpus miss refuses with
    no model call at all.
    """
    config = bot_config("faq")
    if not config.get("web_fallback", True):
        hits = _rerank(_retrieve(question, config), question, {}, config["top_k"])
        if not hits:
            return _refusal(config)
        return _write(llm, config, question, history, hits, []) or _refusal(config)
    # The reading is a model round trip and the first retrieval is local
    # arithmetic; neither needs the other, so they run at the same time.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(_read_question, llm, question, history)
        first = _retrieve(question, config)
        read = pending.result()
    if read.get("in_scope") is False:
        return _out_of_scope(config)
    queries = list(read.get("search_terms", []))
    if read.get("topic") and read["topic"] not in queries:
        queries.append(read["topic"])
    hits = _rerank(_merge(first, *(_retrieve(term, config) for term in queries)),
                   question, read, config["top_k"])
    web_query = read.get("web_query") or question
    if not hits and read.get("needs_context") and read.get("clarifying_question"):
        return BotReply(read["clarifying_question"], [], "clarify")
    strong = bool(hits) and hits[0][1] >= config.get("web_race_below", 0.0)
    if strong:
        answered = _write(llm, config, question, history, hits, [],
                          topic=read.get("topic", ""))
        if answered is not None:
            return answered
    # Thin or no corpus support, or strong passages that declined: the web
    # results join the prompt. `websearch.search` is one HTTP call; the
    # writer is still one model call.
    results = websearch.search(web_query)
    if strong and not results:
        # The passages already declined and the search returned nothing;
        # showing the same passages again would only reproduce the refusal.
        # The writer gets a clean slate -- a backend that searches natively
        # may still ground it -- and the reply is labelled web-sourced.
        hits = []
    answered = _write(llm, config, question, history, hits, results, final=True,
                      topic=read.get("topic", ""))
    if answered is None and read.get("needs_context") and read.get("clarifying_question"):
        # Nothing settled it, and the reading said the answer depends on
        # something unsaid: ask for that rather than refuse.
        return BotReply(read["clarifying_question"], [], "clarify")
    return answered or _refusal(config)


def answer_career(llm, question, history):
    """The career bot: our own retrieval, the same ranking and citation.

    Answers from general best practice when the corpus has nothing, and says
    so -- the `career_advice` skill carries that rule -- so there is no
    refusal and no web search here.
    """
    config = bot_config("career")
    hits = _rerank(_retrieve(question, config, "career"), question, {},
                   config["top_k"])
    system = skills.load(SKILLS_DIR, CAREER_ADVICE).prompt(
        passages=(f"Context passages:\n\n{build_context(hits)}" if hits
                  else "No passages were retrieved for this question."))
    messages = _trimmed(history, config) + [{"role": "user", "content": question}]
    body = llm.chat(system, messages)
    cited = _cited(body, hits) if hits else []
    # "llm" means answered from retrieved passages, with sources; "advice"
    # means general best practice because the corpus had nothing, which the
    # skill tells the model to say.
    return BotReply(append_sources(body, cited),
                    [chunk.pk for chunk, _ in cited], "llm" if cited else "advice")
