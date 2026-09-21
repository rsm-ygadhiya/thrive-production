"""What survived the removal of the web-backed industry recommender.

This module used to test a two-stage recommender: a web lookup for what a
field hires for, then a catalog match. Both stages are gone -- the industry
answer now comes from the curated taxonomy in `data/catalog/industries.json`
(see `test_industry_flow.py`), and the catalog matching is `skill_match`'s,
tested in `test_skill_match.py`.

What is left in `advisor.py` is the half that never touched the web: reading a
field out of a student's sentence, which `router` still depends on to decide
whether a question is industry-shaped. Those are the tests kept here.
"""

from rsm_thrive.services.courses.advisor import (
    _target_field, is_industry_course_question,
)


class TestTheFieldTheStudentNamed:
    """Extraction has to survive the words around it. The pattern can match on
    several prepositions in one question, and since the field is matched whole
    -- every token present -- a single stray word credits nothing at all."""

    def test_a_pronoun_and_verb_are_not_part_of_the_field(self):
        # Matches at "help" (capturing "me work in healthcare analytics") and at
        # "in". The shorter, meaningful one wins.
        assert _target_field(
            "Which electives would help me work in healthcare analytics?"
        ) == "healthcare analytics"

    def test_a_trailing_program_mention_is_not_the_field(self):
        assert _target_field(
            "What electives can I take to assist in the aerospace field "
            "in the MSBA program?") == "aerospace"

    def test_a_question_naming_no_field_yields_none(self):
        assert _target_field("What electives are offered in the MSBA?") is None


class TestWhetherAQuestionIsIndustryShaped:
    """`router` asks this to choose a route, so it has to separate a question
    ABOUT a field from a bare request for the catalog."""

    def test_a_named_field_with_a_course_word_is_industry_shaped(self):
        assert is_industry_course_question("what electives for healthcare")

    def test_a_bare_catalog_request_is_not(self):
        assert not is_industry_course_question(
            "what courses does the MSBA offer?")

    def test_a_question_with_no_course_word_is_not(self):
        assert not is_industry_course_question("how is the healthcare industry")
