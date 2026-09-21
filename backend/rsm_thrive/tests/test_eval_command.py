from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from rsm_thrive.services.embeddings import FakeEmbeddings
from rsm_thrive.services.resources.ingest import ingest_document

pytestmark = pytest.mark.django_db

HANDBOOK = """# Dropping a course

Students may drop a course before the end of week two without a W. After
week two, a drop requires approval from the program office.

# Laptop loans

Laptop loans are handled by the Rady tech desk in room 2W108.
"""

# Its own golden file against its own fixture corpus. Pointing this test at the
# SHIPPED golden set would mean the shipped set could never describe more than
# this two-section handbook — which is exactly what broke when the real corpus
# grew and cases for fees and drop-with-W were added.
FIXTURE_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "faq_golden_fixture.json"


def test_eval_passes_on_seeded_corpus(capsys):
    ingest_document("test:handbook", "MSBA Handbook", "policy", ["resources"],
                    HANDBOOK, FakeEmbeddings())
    call_command("eval_bots", "--llm", "fake", "--golden", str(FIXTURE_GOLDEN))
    out = capsys.readouterr().out
    assert "PASS fixture-drop-deadline" in out
    assert "PASS fixture-off-topic-refusal" in out
    assert "FAIL" not in out


def test_eval_fails_loudly_on_empty_corpus():
    with pytest.raises(CommandError):
        call_command("eval_bots", "--llm", "fake", "--golden", str(FIXTURE_GOLDEN))
