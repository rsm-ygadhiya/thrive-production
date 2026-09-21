"""The Resources bot reads a question the corpus missed before giving up on it.

Found live: "how to raise a IT ticket" got a page of generic help-desk advice
from a web search of those exact words, while "About IT Services" and a page
saying "Submit a Ticket (support.ucsd.edu)" sat in the corpus unmatched. The
`read_question` skill now runs on a miss and drives, in order: a second look
at the corpus with campus wording, a clarifying question only when the answer
genuinely turns on it, and a web search scoped to the campus that says where
it read from.
"""

import json

import pytest

from rsm_thrive.models import Document, DocumentChunk
from rsm_thrive.services import skills
from rsm_thrive.services.embeddings import FakeEmbeddings
from rsm_thrive.services.llm import FakeLLM
from rsm_thrive.services.resources import bot
from rsm_thrive.services.resources.bot import answer_faq

pytestmark = pytest.mark.django_db


def seed(title, text, url="https://blink.ucsd.edu/x"):
    doc = Document.objects.create(source=f"t:{title}", title=title, kind="page",
                                  destinations=["resources"], source_url=url)
    [vector] = FakeEmbeddings().embed([text])
    return DocumentChunk.objects.create(document=doc, seq=0, heading="",
                                        text=text, embedding=vector)


def read(**fields):
    base = {"in_scope": True, "topic": "", "search_terms": [],
            "needs_context": False, "clarifying_question": "", "web_query": ""}
    return json.dumps({**base, **fields})


class TestTheSkillFiles:
    def test_the_reader_takes_no_placeholders(self):
        skill = skills.load(bot.SKILLS_DIR, "read_question")
        assert skill.placeholders == set()
        assert "in_scope" in skill.body and "web_query" in skill.body

    def test_one_writer_is_shown_passages_and_web_results(self):
        skill = skills.load(bot.SKILLS_DIR, "answer_question")
        assert skill.placeholders == {"passages", "web"}
        assert "[W1]" in skill.body and '"answered"' in skill.body


def web_result(title="ITS Service Desk", url="https://support.ucsd.edu/students",
               snippet="Submit a ticket."):
    return type("R", (), {"title": title, "url": url, "snippet": snippet})()


class TestTheCorpusAnswersWhenItCan:
    def test_the_reading_rewords_the_question_and_the_corpus_answers(self):
        """The student's words miss; the campus wording the skill supplies
        hits the page that was there all along, and it is cited as Rady
        material because it is."""
        seed("About IT Services", "submit a ticket to the ITS service desk for help")
        fake = FakeLLM([read(search_terms=["submit a ticket ITS service desk"]),
                        '{"answered": true, "reply": "Submit a ticket through the ITS Service Desk [1]."}'])
        reply = answer_faq(fake, "how to raise a IT ticket", [])
        assert reply.model_note == "llm"
        assert reply.chunk_ids and "Source: [About IT Services](https://blink.ucsd.edu/x)" in reply.body
        assert "web search" not in reply.body
        system, _messages, _ = fake.calls[1]
        assert "[1] About IT Services" in system and "Web results: none" in system

    def test_every_question_is_read_first_then_the_corpus_answers(self):
        seed("Drops", "drop a course before week two")
        fake = FakeLLM([read(), "Before week two [1]."])
        reply = answer_faq(fake, "drop a course before week two", [])
        assert reply.model_note == "llm" and len(fake.calls) == 2
        assert "read_question" not in fake.calls[1][0]

    def test_an_out_of_scope_reading_refuses_before_any_search(self):
        seed("Drops", "who won the super bowl")
        fake = FakeLLM([read(in_scope=False)])
        reply = answer_faq(fake, "who won the super bowl", [])
        assert reply.model_note == "refusal" and len(fake.calls) == 1
        assert reply.body.startswith("That's outside what I can help with"), \
            "trivia is declined as trivia, not as a policy hedge"

    def test_strong_passages_never_trigger_a_web_search(self, monkeypatch):
        from rsm_thrive.services import websearch
        monkeypatch.setattr(websearch, "search",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("searched")))
        seed("Drops", "drop a course before week two")
        fake = FakeLLM([read(web_query="UCSD drop deadline"),
                        '{"answered": true, "reply": "Before week two [1]."}'])
        assert answer_faq(fake, "drop a course before week two", []).model_note == "llm"


class TestAskingForContextIsTheException:
    def test_it_asks_when_nothing_was_found_and_the_answer_turns_on_it(self):
        fake = FakeLLM([read(needs_context=True,
                             clarifying_question="Which account — **campus**, "
                                                 "**Canvas** or **TritonLink**?")])
        reply = answer_faq(fake, "how do i reset my password", [])
        assert reply.model_note == "clarify"
        assert reply.body.startswith("Which account")
        assert len(fake.calls) == 1, "no writer behind a question"

    def test_the_answer_to_the_question_is_read_with_the_conversation(self):
        history = [{"role": "user", "content": "how do i reset my password"},
                   {"role": "assistant", "content": "Which account?"}]
        fake = FakeLLM([read(web_query="UCSD Canvas password reset"),
                        '{"answered": true, "reply": "Use the Canvas login page."}'])
        reply = answer_faq(fake, "canvas", history)
        assert reply.model_note == "web"
        _system, messages, _ = fake.calls[0]
        assert "conversation so far" in messages[0]["content"]
        assert "how do i reset my password" in messages[0]["content"]


class TestTheWebSaysSo:
    def test_the_web_is_searched_for_the_campus_query_and_cited(self, monkeypatch):
        seen = {}
        from rsm_thrive.services import websearch

        def fake_search(query, limit=6):
            seen["query"] = query
            return [web_result()]
        monkeypatch.setattr(websearch, "search", fake_search)
        fake = FakeLLM([read(web_query="UC San Diego ITS Service Desk submit a ticket"),
                        '{"answered": true, "reply": "Use the ITS Service Desk portal [W1]."}'])
        reply = answer_faq(fake, "how to raise a IT ticket", [])
        assert seen["query"] == "UC San Diego ITS Service Desk submit a ticket"
        assert reply.model_note == "web"
        assert "Read from the web just now:" in reply.body
        assert "[ITS Service Desk](https://support.ucsd.edu/students)" in reply.body
        assert "comes from a web search" in reply.body
        assert reply.chunk_ids == [] and "Source:" not in reply.body
        system, _m, _ = fake.calls[1]
        assert "[W1] ITS Service Desk" in system

    def test_no_search_results_still_labels_the_answer(self):
        fake = FakeLLM([read(), '{"answered": true, "reply": "A resume tool."}'])
        reply = answer_faq(fake, "what is vmock", [])
        assert reply.model_note == "web"
        assert "Read from the web just now" not in reply.body
        assert "comes from a web search" in reply.body

    def test_passages_that_decline_go_to_the_web_labelled(self, monkeypatch):
        """Retrieval admitted a page that mentions "submit a ticket" in
        passing; the writer says so, and the question goes to the web."""
        from rsm_thrive.services import websearch
        monkeypatch.setattr(websearch, "search", lambda *a, **k: [web_result()])
        seed("Fellowships", "Contact the unit via the portal. Submit a ticket at support.ucsd.edu")
        fake = FakeLLM([read(web_query="UCSD ITS Service Desk ticket"),
                        '{"answered": false, "reply": ""}',
                        '{"answered": true, "reply": "Use the ITS Service Desk portal [W1]."}'])
        reply = answer_faq(fake, "submit a ticket", [])
        assert reply.model_note == "web"
        assert reply.chunk_ids == [] and "comes from a web search" in reply.body

    def test_passages_that_decline_with_no_web_results_get_a_clean_slate(self):
        seed("Fellowships", "Contact the unit via the portal. Submit a ticket at support.ucsd.edu")
        fake = FakeLLM([read(web_query="UCSD ITS Service Desk ticket"),
                        '{"answered": false, "reply": ""}',
                        '{"answered": true, "reply": "Use the ITS portal."}'])
        reply = answer_faq(fake, "submit a ticket", [])
        assert reply.model_note == "web" and reply.chunk_ids == []
        system, _m, _ = fake.calls[2]
        assert "Passages: none matched" in system, "the declined passages are not shown again"


class TestThinCorpusSupportIsAnsweredInOneCall:
    """When the best passage is thin, the writer sees passages AND web results
    at once, and what it cites decides how the reply is labelled."""

    def _thin(self, monkeypatch):
        real = bot.bot_config
        monkeypatch.setattr(bot, "bot_config",
                            lambda name: {**real(name), "web_race_below": 2.0})
        from rsm_thrive.services import websearch
        monkeypatch.setattr(websearch, "search", lambda *a, **k: [web_result()])

    def test_citing_only_passages_is_programme_material(self, monkeypatch):
        self._thin(monkeypatch)
        seed("Drops", "drop a course before week two")
        fake = FakeLLM([read(web_query="q"), '{"answered": true, "reply": "Before week two [1]."}'])
        reply = answer_faq(fake, "drop a course before week two", [])
        assert reply.model_note == "llm" and len(fake.calls) == 2
        assert "Source:" in reply.body and "web search" not in reply.body
        system, _m, _ = fake.calls[1]
        assert "[1] Drops" in system and "[W1] ITS Service Desk" in system

    def test_citing_only_the_web_is_web_sourced(self, monkeypatch):
        self._thin(monkeypatch)
        seed("Drops", "drop a course before week two")
        fake = FakeLLM([read(web_query="q"), '{"answered": true, "reply": "Portal [W1]."}'])
        reply = answer_faq(fake, "drop a course before week two", [])
        assert reply.model_note == "web" and reply.chunk_ids == []
        assert "Read from the web just now:" in reply.body and "Source:" not in reply.body

    def test_citing_both_shows_both_and_says_which_parts(self, monkeypatch):
        self._thin(monkeypatch)
        seed("Drops", "drop a course before week two")
        fake = FakeLLM([read(web_query="q"),
                        '{"answered": true, "reply": "Before week two [1]; the portal [W1]."}'])
        reply = answer_faq(fake, "drop a course before week two", [])
        assert reply.model_note == "llm+web" and reply.chunk_ids
        assert "Source:" in reply.body and "Read from the web just now:" in reply.body
        assert "the parts marked W" in reply.body

    def test_citing_neither_attributes_to_both(self, monkeypatch):
        self._thin(monkeypatch)
        seed("Drops", "drop a course before week two")
        fake = FakeLLM([read(web_query="q"), '{"answered": true, "reply": "Before week two."}'])
        reply = answer_faq(fake, "drop a course before week two", [])
        assert reply.model_note == "llm+web"


class TestNothingHereMakesAnAnswerWorse:
    def test_an_unreadable_reading_carries_on(self):
        fake = FakeLLM(["not json", '{"answered": true, "reply": "A resume tool."}'])
        assert answer_faq(fake, "what is vmock", []).model_note == "web"

    def test_a_model_that_cannot_be_reached_still_refuses_plainly(self):
        reply = answer_faq(FakeLLM([]), "what is vmock", [])
        assert reply.model_note == "refusal" and reply.refused

    def test_web_fallback_off_means_no_model_call_on_a_miss(self, monkeypatch):
        real = bot.bot_config
        monkeypatch.setattr(bot, "bot_config",
                            lambda name: {**real(name), "web_fallback": False})
        fake = FakeLLM([])
        assert answer_faq(fake, "what is vmock", []).model_note == "refusal"
        assert fake.calls == []


class TestTheReadingRunsOnTheReaderModel:
    def test_the_default_reader_is_the_model_itself(self):
        fake = FakeLLM([read()])
        assert fake.reader() is fake

    def test_the_reading_goes_through_reader(self):
        class Split(FakeLLM):
            def __init__(self):
                super().__init__([])
                self.fast = FakeLLM([read(in_scope=False)])
            def reader(self):
                return self.fast
        llm = Split()
        reply = answer_faq(llm, "who won the super bowl", [])
        assert reply.model_note == "refusal"
        assert len(llm.fast.calls) == 1 and llm.calls == []


class TestTheCareerBotSharesTheRankingAndCitation:
    def test_only_cited_documents_are_listed(self):
        from rsm_thrive.services.resources.bot import answer_career
        for title, text in [("Resume Guide", "keep your resume to one page"),
                            ("Parking", "keep your permit visible")]:
            doc = Document.objects.create(source=f"t:{title}", title=title, kind="page",
                                          destinations=["career"], source_url="https://x/" + title)
            [vector] = FakeEmbeddings().embed(["keep your resume to one page"])
            DocumentChunk.objects.create(document=doc, seq=1, heading="", text=text, embedding=vector)
        fake = FakeLLM(["One page [1]."])
        reply = answer_career(fake, "keep your resume to one page", [])
        assert "Resume Guide" in reply.body and "Parking" not in reply.body


class TestWhatTheModelIsShownIsRankedNotJustRetrieved:
    """Retrieval decides what is IN; this decides the order and the mix."""

    def _hit(self, title, text, url="https://students.ucsd.edu/x", seq=1, score=0.6):
        doc = Document.objects.create(source=f"t:{title}:{seq}", title=title, kind="page",
                                      destinations=["resources"], source_url=url)
        chunk = DocumentChunk.objects.create(document=doc, seq=seq, heading="", text=text,
                                             embedding=[0.0] * 8)
        return (chunk, score)

    def test_programme_material_about_the_question_outranks_campus_material(self):
        campus = self._hit("General Requirements", "students enroll via the registrar", score=0.66)
        rady = self._hit("If you are not enrolled in classes - FINISH RadyReady (MSBA 2027)",
                         "complete RadyReady before the program team enrolls you",
                         url="https://rady.instructure.com/x", score=0.55)
        ranked = bot._rerank([campus, rady], "how to register for classes", {}, 10)
        assert [c.document.title[:6] for c, _ in ranked] == ["If you", "Genera"]
        assert ranked[0][1] == 0.55, "the reported score is still the cosine"

    def test_but_programme_material_about_something_else_does_not(self):
        """Every cohort page carries the MSBA tag. Laptop guidelines are not
        an answer to a registration question, whatever the tag says."""
        campus = self._hit("Graduate Enrollment", "log into TSS to book courses", score=0.58)
        laptop = self._hit("Review Laptop Guidelines (MSBA 2027 cohort)", "a laptop with 16GB",
                           url="https://rady.instructure.com/y", score=0.60)
        ranked = bot._rerank([laptop, campus], "how to register for classes",
                             {"topic": "course enrollment", "search_terms": []}, 10)
        assert ranked[0][0].document.title == "Graduate Enrollment"

    def test_a_title_about_the_question_moves_up(self):
        other = self._hit("Network Connections", "register your device", score=0.62)
        about = self._hit("Graduate Enrollment", "log into TSS to book courses", score=0.58)
        ranked = bot._rerank([other, about], "how to register for classes",
                             {"topic": "course enrollment", "search_terms": []}, 10)
        assert ranked[0][0].document.title == "Graduate Enrollment"

    def test_one_page_cannot_fill_the_prompt(self):
        doc = Document.objects.create(source="t:cat", title="General Catalog", kind="page",
                                      destinations=["resources"])
        hits = [(DocumentChunk.objects.create(document=doc, seq=i, heading="", text=f"para {i}",
                                              embedding=[0.0] * 8), 0.9 - i * 0.01)
                for i in range(1, 6)]
        hits.append(self._hit("Booking Your Classes", "book in TSS", score=0.5))
        ranked = bot._rerank(hits, "register", {}, 10)
        assert sum(1 for c, _ in ranked if c.document_id == doc.pk) == bot.MAX_CHUNKS_PER_DOCUMENT
        assert any(c.document.title == "Booking Your Classes" for c, _ in ranked)

    def test_the_fetch_metadata_chunk_is_never_shown(self):
        meta = self._hit("Booking Your Classes",
                         "Source: https://students.ucsd.edu/academics/enroll/ Fetched: 2026-08-24 · fidelity 98.5",
                         seq=0, score=0.9)
        canvas = self._hit("Review Laptop Guidelines (MSBA 2027 cohort)",
                           "Source: https://rady.instructure.com/courses/2062 (official Rady Canvas — hand-captured) Captured 2026-08-17",
                           seq=0, score=0.9)
        real = self._hit("Booking Your Classes", "Log into TSS to book your courses", seq=2, score=0.6)
        ranked = bot._rerank([meta, canvas, real], "register", {}, 10)
        assert [c.text[:8] for c, _ in ranked] == ["Log into"]

    def test_only_the_documents_the_answer_cited_are_listed(self):
        hits = [self._hit("Graduate Enrollment", "book in TSS", url="https://a.example/1", score=0.7),
                self._hit("PRONTO FAQ", "bus passes", url="https://a.example/2", score=0.6),
                self._hit("Booking Your Classes", "see links", url="https://a.example/3", score=0.5)]
        cited = bot._cited("Log into TSS [1] and see the booking guide [3].", hits)
        assert [c.document.title for c, _ in cited] == ["Graduate Enrollment", "Booking Your Classes"]
        assert bot._cited("No numbers here.", hits) == hits, "uncited means list everything"


class TestADeclineOnTheLastAttemptIsShownNotRefused:
    def test_a_declined_reply_with_content_is_shown_as_a_starting_point(self, monkeypatch):
        """Live: the writer declined with the passages, then declined again
        with web results, and an in-scope question got the corpus-miss
        refusal. The last attempt's words are worth more than that."""
        from rsm_thrive.services import websearch
        monkeypatch.setattr(websearch, "search", lambda *a, **k: [web_result()])
        seed("Fees", "tuition is due quarterly, billed by student financial solutions")
        fake = FakeLLM([read(web_query="UCSD tuition due date"),
                        '{"answered": false, "reply": "Fees are billed quarterly [1]; confirm the due date with advising."}',
                        '{"answered": false, "reply": "Fees are billed quarterly [1]; the public billing page lists Sept 30 [W1]."}'])
        reply = answer_faq(fake, "when is tuition due", [])
        assert not reply.refused
        assert reply.body.startswith(bot.PARTIAL_LEAD)
        assert "Fees are billed quarterly" in reply.body
        assert reply.model_note == "llm+web"

    def test_the_first_decline_still_goes_on_to_the_web(self, monkeypatch):
        from rsm_thrive.services import websearch
        monkeypatch.setattr(websearch, "search", lambda *a, **k: [web_result()])
        seed("Fees", "tuition is due quarterly, billed by student financial solutions")
        fake = FakeLLM([read(web_query="q"),
                        '{"answered": false, "reply": "Billed quarterly [1]."}',
                        '{"answered": true, "reply": "Due September 30 [W1]."}'])
        reply = answer_faq(fake, "when is tuition due", [])
        assert reply.model_note == "web" and reply.body.startswith("Due September 30")

    def test_an_empty_decline_with_a_clarifying_question_asks_it(self):
        seed("Fees", "tuition is due quarterly, billed by student financial solutions")
        fake = FakeLLM([read(needs_context=True, clarifying_question="Which fee — **tuition** or **housing**?"),
                        '{"answered": false, "reply": ""}',
                        '{"answered": false, "reply": ""}'])
        reply = answer_faq(fake, "when is it due", [])
        assert reply.model_note == "clarify" and reply.body.startswith("Which fee")

    def test_an_empty_decline_with_nothing_to_ask_still_refuses(self):
        seed("Fees", "tuition is due quarterly, billed by student financial solutions")
        fake = FakeLLM([read(), '{"answered": false, "reply": ""}', '{"answered": false, "reply": ""}'])
        assert answer_faq(fake, "when is it due", []).refused


class TestAFollowUpIsWrittenAsTheCombinedQuestion:
    def test_the_topic_reaches_the_writer_with_the_fragment(self):
        seed("Drops", "the deadline to drop a class is the end of week two")
        history = [{"role": "user", "content": "when is the deadline"},
                   {"role": "assistant", "content": "Which deadline?"}]
        fake = FakeLLM([read(topic="deadline to drop a class"),
                        '{"answered": true, "reply": "End of week two [1]."}'])
        reply = answer_faq(fake, "to drop a class", history)
        assert reply.model_note == "llm"
        _system, messages, _ = fake.calls[1]
        assert messages[-1]["content"] == "to drop a class\n\n(Understood as: deadline to drop a class)"

    def test_a_first_turn_is_sent_as_typed(self):
        seed("Drops", "the deadline to drop a class is the end of week two")
        fake = FakeLLM([read(topic="deadline to drop a class"),
                        '{"answered": true, "reply": "End of week two [1]."}'])
        answer_faq(fake, "deadline to drop a class", [])
        _system, messages, _ = fake.calls[1]
        assert messages[-1]["content"] == "deadline to drop a class"


class TestTheCareerBotSaysWhenItIsGeneralAdvice:
    def test_no_passages_means_advice_not_a_corpus_answer(self):
        from rsm_thrive.services.resources.bot import answer_career
        reply = answer_career(FakeLLM(["Keep it to one page."]), "resume length?", [])
        assert reply.model_note == "advice" and reply.chunk_ids == []


class TestAWebMarkerWithNoWebResultsIsRemoved:
    def test_a_stray_w_marker_is_stripped(self):
        seed("Drops", "drop a course before week two")
        fake = FakeLLM([read(), '{"answered": true, "reply": "Before week two [1] [W1]."}'])
        reply = answer_faq(fake, "drop a course before week two", [])
        assert reply.model_note == "llm"
        assert "[W1]" not in reply.body and "Before week two [1]." in reply.body


class TestPackedWebMarkers:
    def test_a_bracket_holding_several_web_numbers_is_read_and_cited(self, monkeypatch):
        from rsm_thrive.services import websearch
        monkeypatch.setattr(websearch, "search", lambda *a, **k: [
            web_result("A", "https://a.example/1"), web_result("B", "https://b.example/2"),
            web_result("C", "https://c.example/3")])
        fake = FakeLLM([read(web_query="q"), '{"answered": true, "reply": "Do this [W1, W3]."}'])
        reply = answer_faq(fake, "what is vmock", [])
        assert reply.model_note == "web"
        assert "[A](https://a.example/1)" in reply.body and "[C](https://c.example/3)" in reply.body
        assert "b.example" not in reply.body

    def test_a_packed_stray_marker_is_stripped_when_there_were_no_results(self):
        seed("Drops", "drop a course before week two")
        fake = FakeLLM([read(), '{"answered": true, "reply": "Before week two [1] [W1, W2]."}'])
        reply = answer_faq(fake, "drop a course before week two", [])
        assert "[W" not in reply.body and reply.model_note == "llm"
