import pytest
from django.contrib.auth.models import User

from rsm_thrive.models import Enrollment
from rsm_thrive.services.courses.electives import (
    load_careers, load_catalog, rank_electives, recommend_for)
from rsm_thrive.testing import make_course

pytestmark = pytest.mark.django_db


class TestCatalogData:
    def test_careers_and_catalog_load(self):
        careers = load_careers()
        assert "data-scientist" in careers
        catalog = load_catalog()
        assert any(c["code"] == "MGTA 453" for c in catalog)


class TestRankElectives:
    def test_deterministic(self):
        profile = {"career_roles": ["data-scientist"], "interests": ["ml"]}
        first = rank_electives(load_catalog(), profile, load_careers())
        second = rank_electives(load_catalog(), profile, load_careers())
        assert [r["course"]["code"] for r in first] == \
               [r["course"]["code"] for r in second]

    def test_core_courses_never_recommended(self):
        profile = {"career_roles": ["data-scientist"]}
        results = rank_electives(load_catalog(), profile, load_careers())
        assert all(not r["course"]["is_core"] for r in results)

    def test_role_changes_ranking(self):
        ds = rank_electives(load_catalog(),
                            {"career_roles": ["data-scientist"]}, load_careers())
        pm = rank_electives(load_catalog(),
                            {"career_roles": ["product-manager"]}, load_careers())
        assert [r["course"]["code"] for r in ds[:3]] != \
               [r["course"]["code"] for r in pm[:3]]

    def test_every_result_has_reasons(self):
        results = rank_electives(load_catalog(),
                                 {"career_roles": ["data-scientist"]},
                                 load_careers())
        assert all(r["reasons"] for r in results if r["score"] > 0)


class TestRecommendFor:
    def test_taken_courses_are_excluded_and_limit_applies(self):
        user = User.objects.create_user("stu")
        baseline = recommend_for(user, ["data-scientist"], limit=5)
        assert len(baseline) == 5
        top_code = baseline[0]["course"]["code"]

        course = make_course(code=top_code)
        Enrollment.objects.create(user=user, course=course)

        after = recommend_for(user, ["data-scientist"], limit=5)
        assert top_code not in [r["course"]["code"] for r in after]
