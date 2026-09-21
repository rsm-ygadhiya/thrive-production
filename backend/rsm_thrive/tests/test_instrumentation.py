"""The turn log, read back.

`ChatTurnLog` was written from the day the chatbots landed and never read, so
none of it was ever exercised end to end. These cover the three things the
trace surface promises: that a turn records what was asked and how it was
answered, that a tester's verdict lands against that record, and that a wrong
answer can be traced to the exact chunks behind it.
"""

import datetime as dt
import json

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.utils import timezone

from rsm_thrive.models import ChatTurnLog, TurnFeedback
from rsm_thrive.services.reply import BotReply
from rsm_thrive.services.embeddings import FakeEmbeddings
from rsm_thrive.services.resources.ingest import ingest_document
from rsm_thrive.testing import (make_conversation, make_feedback, make_message,
                                make_student, make_turn_log)
from rsm_thrive.views import chat as chat_views

pytestmark = pytest.mark.django_db


def _staff(username="staffer"):
    profile = make_student(username=username)
    profile.user.is_staff = True
    profile.user.save(update_fields=["is_staff"])
    return profile


def _rated_turn(profile, **turn_fields):
    conversation = make_conversation(profile)
    make_message(conversation, role="student", body=turn_fields.get("question", "q"))
    reply = make_message(conversation, role="thrive", body="an answer")
    return make_turn_log(reply, **turn_fields), reply


# ---------------------------------------------------------------------------
# Writing the log
# ---------------------------------------------------------------------------

def test_turn_log_records_the_question_and_the_route(client, monkeypatch):
    """The trace is useless without the question that produced it."""
    profile = make_student()
    monkeypatch.setattr(
        chat_views, "_run_bot",
        lambda *a, **k: (BotReply("here you go", [], "llm", route="factual",
                                  route_confidence=0.82), 91))
    client.force_login(profile.user)
    client.post("/api/thrive/conversations",
                data=json.dumps({"destination": "resources",
                                 "body": "does 464 have prerequisites?"}),
                content_type="application/json")

    log = ChatTurnLog.objects.get()
    assert log.question == "does 464 have prerequisites?"
    assert (log.route, log.route_confidence) == ("factual", 0.82)
    assert log.refused is False
    assert log.duration_ms == 91


def test_a_refusal_is_flagged_as_one(client, monkeypatch):
    """`refused` is set at the refusal site, not parsed back out of the prose."""
    profile = make_student()
    monkeypatch.setattr(
        chat_views, "_run_bot",
        lambda *a, **k: (BotReply("I don't have that.", [], "refusal",
                                  route="factual", refused=True), 40))
    client.force_login(profile.user)
    client.post("/api/thrive/conversations",
                data=json.dumps({"destination": "resources",
                                 "body": "what is the parking fine appeal window"}),
                content_type="application/json")
    assert ChatTurnLog.objects.get().refused is True


# ---------------------------------------------------------------------------
# The thumb
# ---------------------------------------------------------------------------

def test_thumb_then_note_is_two_writes_on_one_row(client):
    """The click is recorded before the sentence is asked for."""
    profile = make_student()
    turn, reply = _rated_turn(profile)
    client.force_login(profile.user)
    url = (f"/api/thrive/conversations/conv-{reply.conversation_id}"
           f"/messages/msg-{reply.pk}/feedback")

    first = client.post(url, data=json.dumps({"rating": "down"}),
                        content_type="application/json")
    assert first.status_code == 200
    assert first.json() == {"rating": "down", "note": ""}

    second = client.post(url, data=json.dumps({"note": "  it invented a deadline "}),
                         content_type="application/json")
    assert second.json() == {"rating": "down", "note": "it invented a deadline"}
    assert TurnFeedback.objects.count() == 1
    assert TurnFeedback.objects.get().turn_id == turn.pk


def test_changing_your_mind_updates_rather_than_appends(client):
    profile = make_student()
    _turn, reply = _rated_turn(profile)
    client.force_login(profile.user)
    url = (f"/api/thrive/conversations/conv-{reply.conversation_id}"
           f"/messages/msg-{reply.pk}/feedback")
    client.post(url, data=json.dumps({"rating": "down"}),
                content_type="application/json")
    client.post(url, data=json.dumps({"rating": "up"}),
                content_type="application/json")
    assert [f.rating for f in TurnFeedback.objects.all()] == ["up"]

    assert client.delete(url).json() == {"rating": None, "note": ""}
    assert TurnFeedback.objects.count() == 0


def test_a_note_with_no_thumb_is_refused(client):
    profile = make_student()
    _turn, reply = _rated_turn(profile)
    client.force_login(profile.user)
    resp = client.post(
        f"/api/thrive/conversations/conv-{reply.conversation_id}"
        f"/messages/msg-{reply.pk}/feedback",
        data=json.dumps({"note": "bad"}), content_type="application/json")
    assert resp.status_code == 400


def test_you_can_only_rate_your_own_replies(client):
    mine = make_student()
    theirs = make_student(username="other")
    _turn, their_reply = _rated_turn(theirs)
    conversation = make_conversation(mine)
    my_question = make_message(conversation, role="student", body="hi")
    client.force_login(mine.user)

    for path in (f"/api/thrive/conversations/conv-{their_reply.conversation_id}"
                 f"/messages/msg-{their_reply.pk}/feedback",
                 f"/api/thrive/conversations/conv-{conversation.pk}"
                 f"/messages/msg-{my_question.pk}/feedback"):
        resp = client.post(path, data=json.dumps({"rating": "down"}),
                           content_type="application/json")
        assert resp.status_code == 404, path


def test_a_reply_with_no_turn_log_cannot_be_rated(client):
    """Fixtures and pre-log replies have nothing to hang a verdict on."""
    profile = make_student()
    conversation = make_conversation(profile)
    reply = make_message(conversation, role="thrive", body="old answer")
    client.force_login(profile.user)
    resp = client.post(
        f"/api/thrive/conversations/conv-{conversation.pk}/messages/msg-{reply.pk}/feedback",
        data=json.dumps({"rating": "down"}), content_type="application/json")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "no_turn_log"


def test_the_verdict_comes_back_with_the_conversation(client):
    """A thumb that vanishes on reload reads as one that was not recorded."""
    profile = make_student()
    turn, reply = _rated_turn(profile)
    make_feedback(turn, rating="down", note="wrong quarter")
    client.force_login(profile.user)
    body = client.get(f"/api/thrive/conversations/conv-{reply.conversation_id}").json()
    answer = next(m for m in body["messages"] if m["id"] == f"msg-{reply.pk}")
    assert answer["feedback"] == {"rating": "down", "note": "wrong quarter"}
    assert answer["rateable"] is True
    student_turn = next(m for m in body["messages"] if m["role"] == "student")
    assert student_turn["feedback"] is None and student_turn["rateable"] is False


# ---------------------------------------------------------------------------
# Reading it back
# ---------------------------------------------------------------------------

def test_traces_are_staff_only(client):
    profile = make_student()
    client.force_login(profile.user)
    assert client.get("/api/thrive/traces").status_code == 403


def test_traces_filter_and_inline_the_chunks(client):
    """The promise: a bad answer traced to the exact passages, in one look."""
    staff = _staff()
    document = ingest_document(
        source="fixture://handbook", title="Handbook", kind="policy",
        destinations=["resources"],
        text="# Drops\n\nDrop before week two without a W.\n",
        embeddings=FakeEmbeddings())
    chunk = document.chunks.first()

    good, _reply = _rated_turn(staff, question="when can I drop",
                               chunk_ids=[chunk.pk], route="factual")
    _rated_turn(staff, question="what's the weather", route="out-of-scope",
                refused=True, model_note="refusal")
    make_feedback(good, rating="down", note="answered for the wrong term")

    client.force_login(staff.user)
    body = client.get("/api/thrive/traces?thumbs=down&chunks=1").json()
    assert body["count"] == 1
    trace = body["traces"][0]
    assert trace["question"] == "when can I drop"
    assert trace["feedback"]["note"] == "answered for the wrong term"
    assert trace["chunks"][0]["text"].strip().startswith("Drop before week two")
    assert trace["chunks"][0]["documentTitle"] == "Handbook"

    refused = client.get("/api/thrive/traces?refused=1").json()
    assert [t["route"] for t in refused["traces"]] == ["out-of-scope"]
    # Ids without `?chunks=1`; text only when a trace is opened.
    assert "chunks" not in refused["traces"][0]


def test_a_missing_chunk_is_reported_not_dropped(client):
    """A re-ingested corpus is the case where "it came from nowhere" is the finding."""
    staff = _staff()
    turn, _reply = _rated_turn(staff, chunk_ids=[424242])
    client.force_login(staff.user)
    body = client.get(f"/api/thrive/traces/turn-{turn.pk}").json()
    assert body["chunks"] == [{"id": 424242, "missing": True}]


def test_refusal_report_groups_and_skips_out_of_scope(client):
    staff = _staff()
    for _ in range(3):
        _rated_turn(staff, question="Where do I get a locker?", refused=True,
                    route="factual")
    _rated_turn(staff, question="who won the world cup", refused=True,
                route="out-of-scope")
    _rated_turn(staff, question="answered fine", refused=False)

    client.force_login(staff.user)
    body = client.get("/api/thrive/traces/refusals").json()
    assert body["count"] == 1
    assert body["refusals"][0]["question"] == "Where do I get a locker?"
    assert body["refusals"][0]["asked"] == 3

    everything = client.get("/api/thrive/traces/refusals?scope=all").json()
    assert everything["count"] == 2


def test_refusals_window_and_scoping(client):
    staff = _staff()
    old, _reply = _rated_turn(staff, question="ancient question", refused=True)
    ChatTurnLog.objects.filter(pk=old.pk).update(
        created_at=timezone.now() - dt.timedelta(days=40))
    _rated_turn(staff, question="recent question", refused=True)
    client.force_login(staff.user)
    body = client.get("/api/thrive/traces/refusals?days=7").json()
    assert [r["question"] for r in body["refusals"]] == ["recent question"]


# ---------------------------------------------------------------------------
# Feedback -> golden cases
# ---------------------------------------------------------------------------

def test_export_golden_writes_cases_and_keeps_curation(tmp_path):
    profile = make_student()
    turn, _reply = _rated_turn(profile, question="does MGTA 464 have prerequisites?",
                               bot="courses", route="factual")
    make_feedback(turn, rating="down", note="said none; the catalog says MGTA 452")
    out = tmp_path / "from_feedback.json"

    call_command("export_golden", out=str(out))
    cases = json.loads(out.read_text())
    assert len(cases) == 1
    case = cases[0]
    assert case["id"] == f"fb-{turn.pk}"
    assert case["question"] == "does MGTA 464 have prerequisites?"
    assert case["status"] == "needs_expectation"
    assert case["reported"]["note"].startswith("said none")

    # A human writes the expectation; a re-run must not undo that.
    case["must_contain"] = ["MGTA 452"]
    case["status"] = "ready"
    case["reported"] = {"note": "hand-edited"}
    out.write_text(json.dumps(cases))
    call_command("export_golden", out=str(out))
    again = json.loads(out.read_text())[0]
    assert again["must_contain"] == ["MGTA 452"]
    assert again["reported"] == {"note": "hand-edited"}


def test_thumbs_up_is_not_exported_by_default(tmp_path):
    profile = make_student()
    turn, _reply = _rated_turn(profile, question="fine answer")
    make_feedback(turn, rating="up")
    out = tmp_path / "cases.json"
    call_command("export_golden", out=str(out))
    assert json.loads(out.read_text()) == []
    call_command("export_golden", out=str(out), include_up=True)
    assert len(json.loads(out.read_text())) == 1


def test_eval_skips_cases_with_no_expectation_yet(tmp_path, capsys):
    """An unwritten expectation is work outstanding, not a regression."""
    ingest_document(source="fixture://h", title="Handbook", kind="policy",
                    destinations=["resources"],
                    text="# Drops\n\nDrop before week two without a W.\n",
                    embeddings=FakeEmbeddings())
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"id": "fb-1", "question": "when can I drop?", "bot": "resources",
         "must_contain": [], "must_refuse": False, "status": "needs_expectation",
         "reported": {"note": "it said week nine"}},
    ]))
    call_command("eval_bots", golden=[str(cases)])
    out = capsys.readouterr().out
    assert "SKIP fb-1" in out and "it said week nine" in out
    assert "0/0 passed, 1 skipped" in out


def test_eval_still_fails_a_curated_case_that_regresses(tmp_path):
    ingest_document(source="fixture://h", title="Handbook", kind="policy",
                    destinations=["resources"],
                    text="# Drops\n\nDrop before week two without a W.\n",
                    embeddings=FakeEmbeddings())
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"id": "fb-1", "question": "when can I drop a course?", "bot": "resources",
         "must_contain": ["week nine"], "must_refuse": False, "status": "ready"},
    ]))
    with pytest.raises(CommandError):
        call_command("eval_bots", golden=[str(cases)])


# ---------------------------------------------------------------------------
# The admin page
# ---------------------------------------------------------------------------

def test_admin_trace_page_shows_the_passages_inline(admin_client):
    """The promise is "in one look" — a list of chunk ids is not one look."""
    profile = make_student(username="rated")
    document = ingest_document(
        source="fixture://handbook", title="Handbook", kind="policy",
        destinations=["resources"],
        text="# Drops\n\nDrop before week two without a W.\n",
        embeddings=FakeEmbeddings())
    chunk = document.chunks.first()
    turn, _reply = _rated_turn(profile, question="when can I drop",
                               chunk_ids=[chunk.pk, 999999])

    listing = admin_client.get("/admin/rsm_thrive/chatturnlog/")
    assert listing.status_code == 200
    assert b"when can I drop" in listing.content

    detail = admin_client.get(f"/admin/rsm_thrive/chatturnlog/{turn.pk}/change/")
    assert detail.status_code == 200
    assert b"Drop before week two without a W." in detail.content
    assert b"no longer exists" in detail.content


def test_admin_feedback_page_lists_what_was_wrong(admin_client):
    profile = make_student(username="rater")
    turn, _reply = _rated_turn(profile, question="prereqs for MGTA 464?")
    make_feedback(turn, rating="down", note="it invented a prerequisite")
    page = admin_client.get("/admin/rsm_thrive/turnfeedback/")
    assert page.status_code == 200
    assert b"it invented a prerequisite" in page.content
