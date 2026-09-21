import pytest

from rsm_thrive.models import Document, DocumentChunk
from rsm_thrive.services.reply import build_context
from rsm_thrive.services.resources.bot import answer_career, answer_faq
from rsm_thrive.services.embeddings import FakeEmbeddings
from rsm_thrive.services.llm import FakeLLM

pytestmark = pytest.mark.django_db


# What `read_question` says when it has nothing to add: in scope, no better
# wording to retry, nothing to ask -- straight on to the web, as before.
READ_NOTHING_MORE = ('{"in_scope": true, "topic": "", "search_terms": [], '
                     '"needs_context": false, "clarifying_question": "", "web_query": ""}')


def _seed(destinations, text="Students may drop a course before week two."):
    doc = Document.objects.create(source=f"t:{destinations[0]}", title="MSBA Handbook",
                                  kind="policy", destinations=destinations)
    [vector] = FakeEmbeddings().embed([text])
    return DocumentChunk.objects.create(document=doc, seq=0, heading="Drops",
                                        text=text, embedding=vector)


class TestFaqBot:
    def test_thin_retrieval_falls_back_to_the_web(self, settings):
        """No corpus hit no longer means no answer.

        The spec's flat refusal was there because "a confident wrong policy
        answer is worse than none", and that reasoning is kept -- the prompt
        will not state a deadline or fee as settled. What it also refused was
        anything the corpus has not got to yet: "What is VMock?" is a real Rady
        careers tool, band C in the scorecard and so not ingested, and a student
        asking got sent to advising for a one-sentence answer.
        """
        fake = FakeLLM(replies=[READ_NOTHING_MORE,
                                '{"answerable": true, "reply": "VMock is a resume tool."}'])
        reply = answer_faq(fake, "what is vmock", [])
        assert reply.model_note == "web"
        assert reply.body.startswith("VMock is a resume tool.")
        assert "verify" in reply.body.lower()

    def test_a_web_answer_is_never_cited_as_rady_material(self):
        """The defect this pairs with: "2 plus 2 is 4." arrived with
        "Source: MGTA 495 — Special Topics: AI & Prescriptive Analytics"
        attached. An answer the corpus did not give carries no citation."""
        fake = FakeLLM(replies=[READ_NOTHING_MORE,
                                '{"answerable": true, "reply": "Parking costs vary."}'])
        reply = answer_faq(fake, "what is vmock", [])
        assert reply.chunk_ids == []
        assert "source:" not in reply.body.lower()

    def test_an_off_mission_question_still_refuses(self):
        """The golden set's weather / sport / write-my-essay cases. This must
        not turn the FAQ bot into a general chatbot."""
        fake = FakeLLM(replies=['{"in_scope": false, "search_terms": [], "web_query": ""}'])
        reply = answer_faq(fake, "who won the super bowl", [])
        assert reply.model_note == "refusal"
        assert len(fake.calls) == 1, "read as out of scope; the web is never asked"
        assert "advising" in reply.body.lower()

    def test_it_fails_closed_when_the_model_misbehaves(self):
        # Unparseable output means the in-scope classification never happened,
        # so prose must not pass through as though it had been judged.
        # Scope is judged by the READING now, so prose from the writer is an
        # answer -- labelled as web-sourced -- and only an empty one refuses.
        fake = FakeLLM(replies=[READ_NOTHING_MORE, '{"answered": true, "reply": ""}'])
        assert answer_faq(fake, "what is vmock", []).model_note == "refusal"
        fake = FakeLLM(replies=[READ_NOTHING_MORE, "not json at all"])
        reply = answer_faq(fake, "what is vmock", [])
        assert reply.model_note == "web" and reply.body.startswith("not json at all")

    def test_it_fails_closed_when_the_call_raises(self):
        fake = FakeLLM(replies=[])  # any call raises "exhausted"
        reply = answer_faq(fake, "what is the meaning of life", [])
        assert reply.chunk_ids == [] and reply.model_note == "refusal"
        assert "advising" in reply.body.lower()

    def test_the_old_deterministic_refusal_is_still_available(self, monkeypatch):
        """`web_fallback: false` restores the spec's original guarantee exactly
        -- no LLM call at all, so the refusal cannot be argued with."""
        from rsm_thrive.services.resources import bot as bots

        real = bots.bot_config
        monkeypatch.setattr(bots, "bot_config",
                            lambda name: {**real(name), "web_fallback": False})
        fake = FakeLLM(replies=[])
        reply = answer_faq(fake, "what is the meaning of life", [])
        assert reply.chunk_ids == [] and reply.model_note == "refusal"
        assert fake.calls == []

    def test_grounded_answer_carries_context_and_sources(self):
        chunk = _seed(["resources"])
        fake = FakeLLM(replies=[READ_NOTHING_MORE, "You can drop before week two [1]."])
        reply = answer_faq(fake, "when can I drop a course", [])
        assert reply.chunk_ids == [chunk.pk]
        assert reply.body.startswith("You can drop before week two")
        assert "Source: MSBA Handbook" in reply.body
        system, messages, json_mode = fake.calls[1]
        assert "[1] MSBA Handbook" in system
        assert messages[-1] == {"role": "user",
                                "content": "when can I drop a course"}
        assert json_mode is True, "the passages skill answers in JSON, with an answered flag"

    def test_several_sources_become_a_bulleted_list(self):
        """Ten comma-joined titles on one line cannot be read, and the titles
        themselves contain commas, so the reader cannot see where one ends."""
        for n in range(3):
            doc = Document.objects.create(source=f"t:multi{n}", title=f"Doc {n}, Annotated",
                                          kind="policy", destinations=["resources"],
                                          source_url=f"https://example.test/{n}")
            [vector] = FakeEmbeddings().embed(["drop a course before week two"])
            DocumentChunk.objects.create(document=doc, seq=0, heading="Drops",
                                         text="drop a course before week two",
                                         embedding=vector)
        fake = FakeLLM(replies=[READ_NOTHING_MORE, "Before week two."])
        reply = answer_faq(fake, "drop a course before week two", [])

        assert "Sources:\n- " in reply.body
        assert "Sources: " not in reply.body  # never the comma run
        listed = reply.body.split("Sources:\n")[1].splitlines()
        assert len(listed) == 3
        for row in listed:
            assert row.startswith("- [") and "](https://example.test/" in row

    def test_one_source_stays_inline_and_singular(self):
        _seed(["resources"])
        fake = FakeLLM(replies=[READ_NOTHING_MORE, "Before week two."])
        reply = answer_faq(fake, "when can I drop a course", [])
        assert "Source: MSBA Handbook" in reply.body
        assert "Sources:" not in reply.body

    def test_history_is_forwarded_and_truncated(self):
        _seed(["resources"], "drop a course before week two of the quarter")
        fake = FakeLLM(replies=[READ_NOTHING_MORE, "answer"])
        history = [{"role": "user", "content": f"turn {i}"} for i in range(30)]
        answer_faq(fake, "drop a course when", history)
        _, messages, _ = fake.calls[1]
        assert len(messages) <= 13  # 12 history + the question


class TestCareerBot:
    def test_no_context_still_answers(self):
        fake = FakeLLM(replies=["Keep it to one page."])
        reply = answer_career(fake, "resume length?", [])
        assert reply.body == "Keep it to one page."
        assert reply.chunk_ids == [] and reply.model_note == "advice"
        assert "Sources:" not in reply.body

    def test_context_used_when_available(self):
        chunk = _seed(["career"], "Keep the resume to one page for analytics roles.")
        fake = FakeLLM(replies=["One page [1]."])
        reply = answer_career(fake, "how long should my resume for analytics be", [])
        assert reply.chunk_ids == [chunk.pk]
        assert "Source: MSBA Handbook" in reply.body


class TestBuildContext:
    def test_numbered_titled_passages(self):
        chunk = _seed(["resources"])
        context = build_context([(chunk, 0.9)])
        assert context.startswith("[1] MSBA Handbook — Drops\n")
