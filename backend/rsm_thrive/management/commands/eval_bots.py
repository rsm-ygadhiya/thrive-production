"""Run the golden cases through the bots; fail on any regression.

Two sources of cases, deliberately kept in separate files:

* `data/evals/faq_golden.json` — hand-written, and honest about what it is: a
  smoke test over the corpus that ships.
* `data/evals/from_feedback.json` — every turn a tester thumbed down, exported
  by `export_golden`. This is the one that grows, and it grows with failures
  nobody on the team predicted.

Pass `--golden` more than once to run both.

## Cases that cannot pass or fail yet

A thumbs-down says an answer was wrong; it does not say what right looks like.
So an exported case arrives `needs_expectation` and is REPORTED, not run — it
appears in the output as SKIP with the tester's note, so the set of cases
waiting on a human is visible every time the eval runs instead of sitting
unread in a file. A skip never fails the command; an unwritten expectation is
work outstanding, not a regression.
"""

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from rsm_thrive.models import DocumentChunk
from rsm_thrive.services.bot_config import bot_config
from rsm_thrive.services.resources.bot import answer_career, answer_faq
from rsm_thrive.services.llm import FakeLLM, get_llm
from rsm_thrive.services.resources.retrieval import retrieve

EVALS = Path(__file__).resolve().parents[2] / "data" / "evals"
GOLDEN = EVALS / "faq_golden.json"
NEEDS_EXPECTATION = "needs_expectation"

# Which bot a case runs against. `courses` needs a Conversation and makes
# several model calls per turn, so it cannot be driven by a scripted FakeLLM;
# it is skipped unless a real backend is asked for.
RUNNABLE = {"resources", "career"}

# What `read_question` says when it has nothing to add, and when it judges a
# question outside the bot's remit.
READ_OUT_OF_SCOPE = '{"in_scope": false, "search_terms": [], "web_query": ""}'
READ_NOTHING_MORE = ('{"in_scope": true, "topic": "", "search_terms": [], "needs_context": false, "clarifying_question": "", "web_query": ""}')


class Command(BaseCommand):
    help = "Run the golden sets through the bots; fails on any regression."

    def add_arguments(self, parser):
        parser.add_argument("--llm", choices=["fake", "real"], default="fake")
        # The shipped golden set describes the corpus that ships. Unit tests
        # need their own tiny set against their own fixture corpus, or the two
        # become coupled and the golden set can never grow past the fixture.
        parser.add_argument("--golden", action="append", default=None,
                            help="A golden-case JSON file. Repeatable. "
                                 "(default: the shipped set.)")
        parser.add_argument("--include-feedback", action="store_true",
                            help="Also run data/evals/from_feedback.json, if present.")

    def _cases(self, options):
        paths = [Path(p) for p in (options["golden"] or [str(GOLDEN)])]
        if options["include_feedback"] and (EVALS / "from_feedback.json").exists():
            paths.append(EVALS / "from_feedback.json")
        cases = []
        for path in paths:
            for case in json.loads(path.read_text()):
                cases.append({**case, "_file": path.name})
        return cases

    def handle(self, *args, **options):
        if not DocumentChunk.objects.exists():
            raise CommandError("The knowledge table is empty — run ingest_corpus first.")
        cases = self._cases(options)
        failures, skipped = 0, 0
        for case in cases:
            verdict = self._run(case, options)
            if verdict is None:
                skipped += 1
                continue
            ok, why, chunks = verdict
            if ok:
                self.stdout.write(f"PASS {case['id']} (chunks: {chunks})")
            else:
                failures += 1
                self.stdout.write(f"FAIL {case['id']}: {why} (chunks: {chunks})")
        ran = len(cases) - skipped
        self.stdout.write(f"{ran - failures}/{ran} passed"
                          + (f", {skipped} skipped" if skipped else ""))
        if failures:
            raise CommandError(f"{failures} eval case(s) failed.")

    def _run(self, case, options):
        """(ok, why, chunks), or None when the case was skipped."""
        bot = case.get("bot", "resources")
        if case.get("status") == NEEDS_EXPECTATION:
            note = ((case.get("reported") or {}).get("note") or "").strip()
            self.stdout.write(
                f"SKIP {case['id']}: no expectation written yet"
                + (f" — tester said {note[:70]!r}" if note else "")
                + f" ({case['question'][:60]!r})")
            return None
        if bot not in RUNNABLE:
            self.stdout.write(f"SKIP {case['id']}: the {bot!r} bot is not "
                              f"driveable from the eval harness yet")
            return None

        reply = self._answer(case, bot, options)
        chunks = ", ".join(str(i) for i in reply.chunk_ids) or "none"
        refused = reply.refused or reply.model_note == "refusal"
        if case.get("must_refuse"):
            return refused, "answered instead of refusing", chunks
        if refused:
            return False, "refused instead of answering (no-retrieval?)", chunks
        if not reply.chunk_ids and bot == "resources":
            return False, "no-retrieval", chunks
        body = reply.body.lower()
        missing = [kw for kw in case.get("must_contain") or []
                   if kw.lower() not in body]
        if missing:
            return False, f"missing {missing}", chunks
        present = [kw for kw in case.get("must_not_contain") or []
                   if kw.lower() in body]
        if present:
            return False, f"must not contain {present}", chunks
        return True, "", chunks

    def _answer(self, case, bot, options):
        destination = "career" if bot == "career" else "resources"
        config = bot_config("career" if bot == "career" else "faq")
        if options["llm"] == "fake":
            hits = retrieve(case["question"], destination,
                            config["top_k"], config["min_similarity"])
            answer = " ".join(c.text for c, _ in hits) or "?"
            # The resources bot READS the question before it answers (see
            # `resources.bot.answer_faq`), so its scripted model gets a
            # reading that adds nothing, then the answer.
            # A scripted reader cannot judge scope, so the fake stands in
            # for it the way the old flat refusal did: no corpus hit, out of
            # scope. That keeps the golden set's refusal cases meaningful
            # under `--llm fake`; `--llm real` reads for itself.
            reading = READ_NOTHING_MORE if hits else READ_OUT_OF_SCOPE
            llm = FakeLLM(replies=([reading, answer] if bot != "career"
                                   else [answer]))
        else:
            llm = get_llm()
        if bot == "career":
            return answer_career(llm, case["question"], [])
        return answer_faq(llm, case["question"], [])
