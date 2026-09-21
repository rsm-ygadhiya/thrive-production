import json

import pytest

from rsm_thrive.models import CoursePlan
from rsm_thrive.testing import make_student

pytestmark = pytest.mark.django_db

INTAKE = "/api/thrive/plan/intake"
PLAN = "/api/thrive/plan"
ALTS = "/api/thrive/plan/alternatives"
SWAP = "/api/thrive/plan/swap"


@pytest.fixture
def student_client(client):
    """A logged-in student with a profile — what every planner route requires."""
    profile = make_student()
    client.force_login(profile.user)
    return client

ANSWERS = {
    "track": "11 month", "goals": ["data-scientist"],
    "skill_python": "comfortable", "skill_sql": "basic",
    "skill_stats": "comfortable", "skill_ml": "basic",
    "skill_communication": "basic", "workload": "moderate",
    "interests": ["machine-learning"],
}


def _post(client, url, payload):
    return client.post(url, data=json.dumps(payload),
                       content_type="application/json")


class TestIntakeEndpoint:
    def test_requires_login(self, client):
        assert client.get(INTAKE).status_code == 401

    def test_the_opening_invites_free_text_rather_than_asking_a_question(
            self, student_client):
        """The interview opened here with "Which track are you on?" and two
        buttons. It works, and it makes the bot feel like a form — a student
        who types a real sentence was pushed back into a script. What opens the
        surface now says what it can do and gets out of the way."""
        starter = student_client.get(INTAKE).json()["starter"]
        assert starter["quickReplies"] == []
        assert starter["form"] is None
        assert "Step 1 of" not in starter["body"]
        # It names the shapes of question the router can actually route.
        for example in ("data scientist", "prerequisites", "industries",
                        "11 month"):
            assert example in starter["body"], example

    def test_returns_the_script_and_the_profile_track_as_a_default(self, student_client):
        body = student_client.get(INTAKE).json()
        assert [q["key"] for q in body["questions"]][0] == "track"
        assert body["suggested"]["track"]
        assert body["hasPlan"] is False


class TestPlanEndpoint:
    def test_no_plan_yet_is_a_404_not_an_empty_plan(self, student_client):
        response = student_client.get(PLAN)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "no_plan"

    def test_submitting_the_intake_returns_a_complete_plan(self, student_client):
        response = _post(student_client, PLAN, {"answers": ANSWERS})
        assert response.status_code == 201
        plan = response.json()
        assert plan["totals"]["total"] == 50
        assert plan["unfilled"] == []
        assert len(plan["quarters"]) == 4
        assert {"tss", "plans-drive"} <= {l["key"] for l in plan["links"]}

    def test_every_quarter_reports_core_and_elective_rows(self, student_client):
        plan = _post(student_client, PLAN, {"answers": ANSWERS}).json()
        for quarter in plan["quarters"]:
            assert {r["requirement"] for r in quarter["courses"]} <= {
                "Core", "Elective", "Required", "Done"}
        assert any(r["requirement"] == "Core"
                   for q in plan["quarters"] for r in q["courses"])

    def test_a_bad_intake_is_rejected_with_the_reason(self, student_client):
        response = _post(student_client, PLAN, {"answers": {**ANSWERS, "track": "x"}})
        assert response.status_code == 400
        assert "track" in response.json()["error"]["message"]

    def test_answers_must_be_an_object(self, student_client):
        assert _post(student_client, PLAN, {"answers": "11 month"}).status_code == 400

    def test_the_plan_persists_and_reads_back(self, student_client):
        _post(student_client, PLAN, {"answers": ANSWERS})
        again = student_client.get(PLAN).json()
        assert again["totals"]["total"] == 50
        assert again["intake"]["goals"] == ["data-scientist"]

    def test_resubmitting_the_intake_clears_old_swaps(self, student_client):
        _post(student_client, PLAN, {"answers": ANSWERS})
        alt = student_client.get(
            ALTS,
            {"quarter": "fall", "slot": 2}).json()
        _post(student_client, SWAP,
              {"quarter": "fall", "slot": 2, "courseId": alt["options"][0]["courseId"]})
        assert CoursePlan.objects.get().selections != {}
        _post(student_client, PLAN, {"answers": {**ANSWERS, "goals": ["consultant"]}})
        assert CoursePlan.objects.get().selections == {}

    def test_delete_removes_the_plan(self, student_client):
        _post(student_client, PLAN, {"answers": ANSWERS})
        assert student_client.delete(PLAN).status_code == 200
        assert not CoursePlan.objects.exists()


class TestAlternativesEndpoint:
    def test_needs_a_plan_first(self, student_client):
        response = student_client.get(ALTS,
                                      {"quarter": "fall", "slot": 2})
        assert response.status_code == 404

    def test_offers_same_size_options_with_a_rationale(self, student_client):
        _post(student_client, PLAN, {"answers": ANSWERS})
        body = student_client.get(ALTS,
                                  {"quarter": "fall", "slot": 2}).json()
        assert body["options"]
        for option in body["options"]:
            assert option["units"] == body["current"]["units"]
            assert option["sharedSkills"] or option["sharedFocus"] or option["reasons"]

    def test_a_non_numeric_slot_is_a_bad_request(self, student_client):
        _post(student_client, PLAN, {"answers": ANSWERS})
        response = student_client.get(ALTS,
                                      {"quarter": "fall", "slot": "middle"})
        assert response.status_code == 400

    def test_an_unknown_quarter_is_a_bad_request(self, student_client):
        _post(student_client, PLAN, {"answers": ANSWERS})
        response = student_client.get(ALTS,
                                      {"quarter": "nope", "slot": 0})
        assert response.status_code == 400


class TestSwapEndpoint:
    def _alt(self, client):
        return client.get(ALTS,
                          {"quarter": "fall", "slot": 2}).json()["options"][0]

    def test_a_swap_updates_the_plan_and_marks_the_choice(self, student_client):
        _post(student_client, PLAN, {"answers": ANSWERS})
        option = self._alt(student_client)
        plan = _post(student_client, SWAP,
                     {"quarter": "fall", "slot": 2,
                      "courseId": option["courseId"]}).json()
        row = plan["quarters"][1]["courses"][2]
        assert row["courseId"] == option["courseId"]
        assert row["note"] == "your choice"
        assert plan["totals"]["total"] == 50

    def test_a_swap_leaves_the_rest_of_the_plan_alone(self, student_client):
        before = _post(student_client, PLAN, {"answers": ANSWERS}).json()
        option = self._alt(student_client)
        after = _post(student_client, SWAP,
                      {"quarter": "fall", "slot": 2,
                       "courseId": option["courseId"]}).json()
        moved = [(q["key"], i)
                 for q, qa in zip(before["quarters"], after["quarters"])
                 for i, (a, b) in enumerate(zip(q["courses"], qa["courses"]))
                 if a["courseId"] != b["courseId"]]
        assert moved == [("fall", 2)]

    def test_only_the_swapped_slot_is_called_the_students_choice(self, student_client):
        """Pinning writes auto-picks into selections; they must not be relabelled
        as the student's choice just because they are now pinned."""
        _post(student_client, PLAN, {"answers": ANSWERS})
        option = self._alt(student_client)
        plan = _post(student_client, SWAP,
                     {"quarter": "fall", "slot": 2,
                      "courseId": option["courseId"]}).json()
        chosen = [(q["key"], i) for q in plan["quarters"]
                  for i, r in enumerate(q["courses"])
                  if r["note"] == "your choice"]
        assert chosen == [("fall", 2)]

    def test_an_illegal_swap_is_refused_with_the_rule_that_stopped_it(self, student_client):
        _post(student_client, PLAN, {"answers": ANSWERS})
        response = _post(student_client, SWAP,
                         {"quarter": "fall", "slot": 2, "courseId": "MGTA 402"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "swap_refused"
        assert "units" in response.json()["error"]["message"]

    def test_swapping_a_core_slot_is_refused(self, student_client):
        _post(student_client, PLAN, {"answers": ANSWERS})
        response = _post(student_client, SWAP,
                         {"quarter": "fall", "slot": 0, "courseId": "MGTA 457"})
        assert response.status_code == 400

    def test_swap_needs_a_plan(self, student_client):
        assert _post(student_client, SWAP,
                     {"quarter": "fall", "slot": 2,
                      "courseId": "CSE 251B"}).status_code == 404

    def test_a_missing_field_is_a_bad_request(self, student_client):
        _post(student_client, PLAN, {"answers": ANSWERS})
        assert _post(student_client, SWAP,
                     {"quarter": "fall", "slot": 2}).status_code == 400


class TestAMalformedBodyIsA400NotA500:
    """The defect: `validate_intake` tested a client-supplied value against a
    set, so an UNHASHABLE value raised TypeError from inside `in` instead of
    failing validation.

    `POST /api/thrive/plan` takes `answers` straight from the caller, so
    `{"answers": {"track": {"a": 1}}}` crashed the request -- any logged-in
    student could turn a malformed body into a 500.
    """

    @pytest.mark.parametrize("answers", [
        {"track": {"a": {"b": 1}}},
        {"track": ["11 month"]},
        {"goals": [{"x": 1}]},
        {"goals": {"a": 1}},
        {"skill_python": {"x": 1}},
        {"workload": ["moderate"]},
        {"track": {"a": 1}, "goals": [{"b": 2}], "workload": {"c": 3}},
    ])
    def test_unhashable_values_are_rejected_not_raised(self, student_client, answers):
        response = student_client.post(
            PLAN, data=json.dumps({"answers": answers}),
            content_type="application/json")
        assert response.status_code == 400, response.content[:200]

    def test_validate_intake_reports_them_as_problems(self):
        from rsm_thrive.services.courses import planner

        problems = planner.validate_intake({"track": {"a": 1}, "goals": [{"b": 2}]})
        assert problems, "an unhashable value must be a problem, not an exception"
