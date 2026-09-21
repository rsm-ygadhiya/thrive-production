# pytest-django picks up DJANGO_SETTINGS_MODULE from pyproject.toml.

from pathlib import Path

import pytest

_BOTS_TEST_CONFIG = str(Path(__file__).parent / "rsm_thrive" / "tests"
                        / "fixtures" / "bots-test.json")


@pytest.fixture(autouse=True)
def _no_real_llm(settings):
    """Neutralise every backend a developer's `backend/.env` could select.

    `.env` seeds `os.environ` at settings import, so whatever a laptop is
    configured for reaches the suite. Pinning only `THRIVE_LLM` was not enough:
    a `.env` carrying `THRIVE_EMBEDDINGS=local` -- which the tritonai backend
    REQUIRES, since the gateway embeds at 1024 dimensions and the stored chunks
    are 512 -- silently swapped `FakeEmbeddings` for the real local encoder and
    failed seven tests that had nothing to do with it. A test run must not
    depend on which backend the developer happens to be pointed at.
    """
    settings.THRIVE_LLM = "fake"
    # Empty, not "fake": `get_embeddings` treats unset as "follow THRIVE_LLM",
    # which is the arrangement every threshold in the fixtures is calibrated to.
    settings.THRIVE_EMBEDDINGS = ""
    # No test reaches the network. `LLM.search_chat` calls `websearch`
    # on every backend that cannot search natively, so leaving this at
    # its default would have the suite hitting DuckDuckGo.
    settings.THRIVE_SEARCH = "none"
    # The query cache is module state and outlives a test. One test
    # priming it would silently answer the next one from the wrong
    # results -- which is how a cache turns a passing suite into a
    # meaningless one.
    from rsm_thrive.services import websearch

    websearch.forget()
    # Retrieval thresholds in config/bots.json are calibrated to the REAL
    # embedding space (TritonAI, 1024-dim). FakeEmbeddings scores sit lower,
    # so tests overlay the fake-calibrated thresholds — the same deploy-free
    # override mechanism production tuning uses.
    settings.THRIVE_BOT_CONFIG = _BOTS_TEST_CONFIG


@pytest.fixture(autouse=True)
def _cold_retrieval_cache():
    """Every test starts with no cached corpus.

    The cache keys on a fingerprint of the corpus, and the fingerprint is
    strong enough that this is belt-and-braces -- but a test that edits a
    chunk in place, keeping its length and its document's timestamp, would be
    the one case the fingerprint cannot see, and tests are exactly where
    someone does that.
    """
    from rsm_thrive.services.resources import retrieval
    retrieval.forget_corpus()
    yield
    retrieval.forget_corpus()
