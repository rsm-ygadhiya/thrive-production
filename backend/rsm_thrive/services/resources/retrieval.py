"""Top-k chunk retrieval scoped to a destination.

Hybrid, not purely vector. The vector half finds paraphrases; the keyword half
finds the chunk that literally contains "parking permit" or "$21,279.39".

Why both, measured 2026-08-24 on the real corpus with TritonAI
`api-tgpt-embeddings`: cosine alone put the chunks holding the Rady fee amounts
outside the top FOURTEEN for "what does the MSBA cost per quarter and when is
payment due?", because a chunk of pipe-table rows has almost no lexical overlap
with a natural-language question, while prose chunks that merely mention fees
scored 0.68. The embedding model itself is fine (identical text 0.9999,
unrelated 0.11-0.32); the failure is specific to dense, number-heavy chunks —
exactly the ones carrying deadlines and dollar figures, i.e. the facts a
student acts on.

The blend is deliberately conservative: cosine still dominates ranking, and the
keyword term can only lift a chunk that shares distinctive words with the
question.

Admission is two-tier, and the second tier exists because cosine alone is not
measurable on SHORT questions. Measured 2026-08-24 on the shipped corpus, the
same document scores 0.840 for "How do I set up my UCSD Zoom Pro account?" and
0.531 for "How do i get zoom?" — below the 0.55 gate, so the bot refused a
question its own corpus answers on page one. That 0.55 was calibrated against
the persona probe sets, whose questions are full prose; a student types three
words. Worse, at that length the ordering inverts: for "library hours" the
right page (UC San Diego Library Tours, which contains both words) sits at
0.350 while "About Student Financial Solutions" sits at 0.447. Absolute cosine
is tracking how well-formed the QUESTION is, not whether the chunk answers it.

So a chunk is also admitted when it contains EVERY distinctive term of the
question (`keyword_score >= lexical_min`, i.e. 1.0), whatever its cosine. That
is a much stronger claim than the ranking blend makes: not "shares some words"
but "the question's whole vocabulary is in this passage". It is what rescues
"how do i get zoom", "library hours", "transcript request" and "counseling
services", and it is why the refusal rule survives — "what is the recipe for
lasagna" needs one chunk holding both `recipe` AND `lasagna`, and there is
none. Swept against 13 must-refuse controls (the 3 in faq_golden.json plus 10
off-corpus questions), this tier leaked zero. Replacing it with a cosine floor
plus a PARTIAL keyword match was tried and rejected: it bought no additional
coverage and at 0.40 it leaked 3 of the 13. (That is a different knob from
`lexical_floor`, which does not relax the "every term present" rule — it only
declines to apply it at absurdly low similarity.)

The lexical tier then needed two guards of its own, both added after measuring
100 deliberately unclear questions:

* `lexical_floor`, because "every term present" means little when there is only
  ONE term. "who is the president" was admitted at cosine 0.222 off the word
  "president" appearing in a Fellowships page, and the bot answered from that
  chunk rather than refusing.
* typo tolerance, because the tier demands EXACT term presence, so one
  misspelling closed it completely — "zooom" retrieved nothing about Zoom, not
  for want of the page but because the corpus spells the product correctly.
  Coverage on those 100 questions went from 46/70 to 52/70 with no loss on
  well-formed phrasing.

`min_similarity` still gates the cosine tier exactly as it did, and
KEYWORD_WEIGHT still only affects ranking — sweeping it above 0.25 made
coverage worse, not better.
"""

import math
import re
import threading

import numpy


from rsm_thrive.models import DocumentChunk
from rsm_thrive.services.embeddings import cosine, get_embeddings

# Question-shaped words carry no signal about which chunk answers them.
STOPWORDS = frozenset("""
a an the and or but if then than of to in on at by for with from as is are was
were be been being it its this that these those i me my we our you your they
them their he she his her do does did done have has had having can could should
would will shall may might must not no nor so about into over under out up down
off across per via more most other some such only own same too very just also
all any each few both what which who whom when where how why get got need want
""".split())

WORD_RE = re.compile(r"[a-z0-9][a-z0-9$.,/-]*")

# Weight on the keyword half. 0.25 was chosen so a perfect keyword match can
# outrank a cosine gap of ~0.25 (wide enough to rescue the fee tables, narrow
# enough that ranking stays embedding-led).
KEYWORD_WEIGHT = 0.25

# ---------------------------------------------------------------------------
# Lexical-only mode: BM25. See `NullEmbeddings` for why this mode exists.
# ---------------------------------------------------------------------------
#
# The first attempt reused the lexical TIER's rule -- every distinctive term of
# the question present in one chunk -- and it scored 3/9 on the golden eval
# against a documented 6/9, refusing five questions the corpus answers. The
# measurement that explains it: "What is the laptop requirement for MSBA?"
# reaches 0.82 term coverage and "How much is MSBA tuition per unit?" 0.71,
# because no single chunk holds EVERY word of a natural-language question. That
# rule was never meant to stand alone; cosine is what caught the rest, and the
# hand-picked probes it passed were short and keyword-shaped.
#
# BM25 is the standard answer: it ranks PARTIAL matches instead of demanding
# all of them, and its IDF term down-weights words that appear everywhere --
# which is also the guard `lexical_floor` provided and that a vector-free mode
# otherwise loses ("president" is common; "immunization" is not).
BM25_K1 = 1.5     # term-frequency saturation, standard default
BM25_B = 0.75     # length normalisation, standard default

# Admission when there are no vectors at all (see `NullEmbeddings`).
#
# Swept 2026-08-29 over the shipped corpus, 14 answerable questions against 10
# off-corpus controls. Relaxing the bar below 1.0 was tried FIRST, on the theory
# that "every distinctive term present" is strict because cosine is there to
# catch what it misses. The measurement says otherwise: 0.70 answered 14/14 but
# leaked 5 of 10, and 0.80 leaked 2 while answering no more than 1.0 does. So
# the strict bar is right with or without vectors, and the only cost is
# "what is the laptop requirement", which no threshold rescued.
# Absolute floor: below this, nothing in the corpus matched meaningfully at all.
#
# Calibrated 2026-08-29 against `faq_golden.json`, and the calibration is the
# finding: BM25 CANNOT separate on-topic from off-topic here. The answerable
# cases score 10.47-18.60 and the must-refuse cases 6.53-11.41, so
# `refuse-do-my-work` (11.41) outranks both `msba-per-unit-tuition` (10.47) and
# `laptop` (10.91). No absolute threshold and no best/second ratio splits them.
# Deciding that "how do I drop a class" is in scope and "do my homework" is not
# is exactly the semantic judgement embeddings make, and this mode has none.
#
# So the floor is set for COVERAGE, and safety is left to the layer that can
# still provide it: the FAQ prompt answers only from retrieved passages, and on
# the three must-refuse controls it correctly replies that the material does not
# cover them rather than inventing an answer. The DETERMINISTIC refusal (no
# chunks, no LLM call) is what is lost -- `eval_bots` counts those three as
# failures for that reason, and it is right to.
BM25_MIN_SCORE = 6.0
# Keep chunks within this fraction of the best score. Calibrated below.
BM25_RELATIVE_CUT = 0.45

# ...and one guard that CANNOT come with it, left off deliberately.
#
# `lexical_floor` exists because "every term present" means little when there is
# only ONE term -- it is what stops "who is the president" being admitted off
# the word "president" in a Fellowships page. It is a COSINE floor, so it cannot
# run here, and that exact leak returns: 1 of 10 controls.
#
# Requiring two distinctive terms was tried as a vector-free substitute for it
# and REJECTED on measurement. It does close the leak, but stopwords mean a
# great many real questions carry one distinctive term: "when is orientation" is
# {orientation}, "how do i get my student id" is {student}, and "how do i get
# zooom" is {zooom} -- so it also took typo tolerance out entirely. Coverage
# 13/14 -> 11/14 to remove one leak on a question the corpus cannot answer
# anyway, and whose retrieved chunk the FAQ prompt then declines to answer from.
# Losing "when is orientation" is the worse failure of the two.


def _term_counts(text):
    """Distinctive terms with their frequencies. BM25 needs counts, not presence."""
    counts = {}
    for word in WORD_RE.findall((text or "").lower()):
        word = word.strip(".,/-")
        if len(word) > 2 and word not in STOPWORDS:
            counts[word] = counts.get(word, 0) + 1
    return counts


def bm25_scores(query_groups, doc_counts, doc_lengths):
    """BM25 for one query over every chunk, as a list parallel to `doc_counts`.

    `query_groups` are the typo-expanded spellings of each query term, so a
    term matches if ANY of its spellings occurs -- the same tolerance the
    keyword tier has, kept rather than rebuilt.
    """
    total = len(doc_counts)
    if not total or not query_groups:
        return [0.0] * total
    average = sum(doc_lengths) / total or 1.0

    scores = [0.0] * total
    for group in query_groups:
        # How many chunks contain any spelling of this term -> its IDF.
        containing = sum(1 for counts in doc_counts
                         if any(spelling in counts for spelling in group))
        if not containing:
            continue
        idf = math.log(1 + (total - containing + 0.5) / (containing + 0.5))
        for index, counts in enumerate(doc_counts):
            frequency = sum(counts.get(spelling, 0) for spelling in group)
            if not frequency:
                continue
            norm = 1 - BM25_B + BM25_B * (doc_lengths[index] / average)
            scores[index] += idf * (frequency * (BM25_K1 + 1)) / (
                frequency + BM25_K1 * norm)
    return scores


def _terms(text):
    """Distinctive lowercase terms in a string, stopwords removed."""
    return {
        word.strip(".,/-")
        for word in WORD_RE.findall((text or "").lower())
        if len(word) > 2 and word not in STOPWORDS
    }


# Only words this long are typo-corrected. Measured on the misspellings students
# actually type: every one of "zooom", "trascript", "electves", "laptp",
# "orentation", "counsling", "internshp", "imunizations", "finacial", "sylabus"
# is edit-distance 1 from the right corpus term. Below 5 characters the rule
# stops being safe rather than stops being useful -- "fees" is one edit from
# "feed", "drop" from "drip" -- so short words are matched exactly.
MIN_FUZZY_LENGTH = 5

_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

# Rebuilding the corpus vocabulary on every question would mean a second full
# scan per query, so it is cached against (chunk count, highest pk): both move
# when anything is ingested, and ingestion is the only thing that changes it.
def vocabulary_of(haystacks):
    """Every distinctive term in the chunks under consideration.

    Deliberately NOT cached. The first version keyed a module-level cache on
    (chunk count, highest pk); both of those can repeat — pytest rolls each
    test's transaction back and SQLite reuses pks — so one test was scored
    against the vocabulary of the previous one. A cache whose key can collide
    silently serves the wrong answer, and retrieval already tokenises every
    chunk in order to score it, so the vocabulary costs nothing extra if it
    reuses that work rather than repeating it.

    Scoped to the destination being searched, so a question asked of the FAQ
    bot is never "corrected" into a word that exists only in the career corpus.
    """
    terms = set()
    for haystack in haystacks:
        terms |= haystack
    return frozenset(terms)


def _edits1(word):
    """Every string one edit (delete/transpose/replace/insert) from `word`."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    return set(
        [a + b[1:] for a, b in splits if b]
        + [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        + [a + c + b[1:] for a, b in splits if b for c in _ALPHABET]
        + [a + c + b for a, b in splits for c in _ALPHABET]
    )


def expand_terms(query_terms, vocabulary):
    """Group each query term with the corpus spellings it might be a typo of.

    Returns one frozenset per term: the term itself, plus any corpus word one
    edit away IF the term does not appear in the corpus at all. A word that is
    already in the corpus is never "corrected" -- "calender" really does occur
    in a UCSD page, so it stays itself rather than being rewritten to
    "calendar".

    Generating the ~1,500 strings one edit from a word and intersecting them
    with the vocabulary is far cheaper than comparing the word against 11,774
    vocabulary entries, and it is exact rather than a similarity heuristic.

    Why this matters: the lexical admission tier requires every query term to
    appear in one chunk, so a single misspelled word closed the gate entirely.
    "zooom" retrieved nothing about Zoom -- not because the corpus lacks the
    page, but because it spells the product correctly.
    """
    groups = []
    for term in query_terms:
        candidates = {term}
        if term not in vocabulary and len(term) >= MIN_FUZZY_LENGTH:
            candidates |= _edits1(term) & vocabulary
        groups.append(frozenset(candidates))
    return groups


def _as_groups(query_terms):
    """Accept either bare terms or already-grouped spellings."""
    return [frozenset([t]) if isinstance(t, str) else frozenset(t) for t in query_terms]


def _score(groups, haystack):
    """Fraction of the question's distinctive terms present in one haystack."""
    if not groups or not haystack:
        return 0.0
    hits = sum(1 for group in groups if group & haystack)
    # sqrt keeps a couple of shared words from scoring as high as a full match
    # while still rewarding the first hits generously.
    return math.sqrt(hits / len(groups))


def keyword_score(query_terms, chunk):
    """Fraction of the question's distinctive terms present in this chunk.

    Scored against heading + text, because the heading is often where the
    describing words live ("Registration fees by quarter for ...").
    """
    return _score(_as_groups(query_terms),
                  _terms(f"{chunk.heading or ''} {chunk.text or ''}"))


# ---------------------------------------------------------------------------
# The corpus, held in memory between turns
# ---------------------------------------------------------------------------
#
# Every turn on the resources bot used to load ALL 5,889 chunks from the
# database with their documents, JSON-parse 67 MB of stored vectors, tokenise
# every chunk, and then run a pure-Python cosine over 512 floats per chunk.
# Measured: 582 ms median per query, of which embedding the question was 0.1
# ms. The scan was the whole cost, and it grew linearly with the corpus.
#
# None of that changes between turns. It changes when the corpus does, which
# is `ingest_corpus` -- so it is computed once per process and again whenever
# the corpus fingerprint moves. Numerically identical to the loop it replaces:
# the eval, the golden set and every retrieval test run unchanged on it.

_CORPUS_LOCK = threading.Lock()
_CORPUS = None            # (fingerprint, {destination: _Corpus})


def _fingerprint():
    """Changes whenever the corpus does; one cheap aggregate query.

    Count and highest id catch an ingest. Total text length and the newest
    document's fetch time are there for the test suite, where a transaction
    rollback lets SQLite hand the same ids to a different fixture: measured,
    one retrieval test passed alone and failed in the suite, served a corpus
    from the test before it. Four fields agreeing across two different
    corpora is not a case worth designing for.
    """
    from django.db.models import Count, Max, Sum
    from django.db.models.functions import Length

    from rsm_thrive.models import Document

    row = DocumentChunk.objects.aggregate(
        n=Count("id"), top=Max("id"), size=Sum(Length("text")))
    # The newest document's fetch time, so two corpora that agree on count,
    # ids and total length -- which the test suite produces, because SQLite
    # hands back the same ids after a rollback -- still read as different.
    # Every ingest sets it; nothing else does.
    newest = Document.objects.aggregate(t=Max("fetched_at"))["t"]
    return (row["n"] or 0, row["top"] or 0, row["size"] or 0,
            newest.isoformat() if newest else "")


def forget_corpus():
    """Drop the cache by hand.

    Not needed after an ingest: the fingerprint is re-read on every call, so
    a corpus that changed in this process is rebuilt on the very next query.
    This exists for the cases the fingerprint cannot see -- a test that edits
    a chunk in place without changing its length, or anyone who wants the
    rebuild now for their own reasons."""
    global _CORPUS
    with _CORPUS_LOCK:
        _CORPUS = None


class _Corpus:
    """One destination's chunks, tokenised and stacked, ready to score."""

    def __init__(self, rows):
        # rows: (id, heading, text, title, embedding)
        self.ids = [r[0] for r in rows]
        texts = [f"{r[1] or ''} {r[2] or ''}" for r in rows]
        self.haystacks = [_terms(t) for t in texts]
        self.abouts = [_terms(f"{r[1] or ''} {r[3] or ''}") for r in rows]
        self.counts = [_term_counts(t) for t in texts]
        self.vocabulary = vocabulary_of(self.haystacks)
        vectors = [r[4] for r in rows]
        width = max((len(v) for v in vectors if v), default=0)
        if width:
            matrix = numpy.zeros((len(vectors), width), dtype=numpy.float32)
            for i, v in enumerate(vectors):
                if v and len(v) == width:
                    matrix[i] = v
            norms = numpy.linalg.norm(matrix, axis=1)
            norms[norms == 0.0] = 1.0        # a zero row stays zero, scores 0.0
            self.unit = matrix / norms[:, None]
        else:
            self.unit = numpy.zeros((len(vectors), 0), dtype=numpy.float32)

    def similarities(self, query_vector):
        if not query_vector or self.unit.shape[1] != len(query_vector):
            # `cosine` returns 0.0 for a width mismatch; so does this.
            return [0.0] * len(self.ids)
        q = numpy.asarray(query_vector, dtype=numpy.float32)
        norm = numpy.linalg.norm(q)
        if norm == 0.0:
            return [0.0] * len(self.ids)
        return (self.unit @ (q / norm)).tolist()


def _corpus_for(destination):
    global _CORPUS
    key = _fingerprint()
    with _CORPUS_LOCK:
        if _CORPUS is None or _CORPUS[0] != key:
            by_destination = {}
            rows = DocumentChunk.objects.select_related("document").values_list(
                "id", "heading", "text", "document__title", "embedding",
                "document__destinations")
            grouped = {}
            for cid, heading, text, title, embedding, destinations in rows:
                for dest in (destinations or []):
                    grouped.setdefault(dest, []).append(
                        (cid, heading, text, title, embedding))
            for dest, chunk_rows in grouped.items():
                by_destination[dest] = _Corpus(chunk_rows)
            _CORPUS = (key, by_destination)
        return _CORPUS[1].get(destination) or _Corpus([])


def _rows(ranked):
    """(chunk, similarity) pairs, in rank order, as real model instances.

    Only the top-k are fetched. Callers read `chunk.document.title` and
    `source_url` off these to cite sources, so what comes back has to be the
    same objects it always was -- the cache changes how the ranking is
    computed, not what a caller receives.
    """
    if not ranked:
        return []
    wanted = [chunk_id for chunk_id, _s, _r in ranked]
    fetched = {c.id: c for c in
               DocumentChunk.objects.filter(id__in=wanted).select_related("document")}
    return [(fetched[chunk_id], similarity)
            for chunk_id, similarity, _rank in ranked if chunk_id in fetched]


def retrieve(query, destination, top_k, min_similarity, lexical_min=None,
             lexical_floor=0.0, embeddings=None):
    """Top-k (chunk, cosine) pairs for `query`, scoped to one bot's corpus.

    A chunk is admitted if EITHER its cosine clears `min_similarity` (the
    embedding-led tier, unchanged) OR its keyword score clears `lexical_min`
    — every distinctive term of the question present in that one chunk. See
    the module docstring for the measurement behind the second tier.

    `lexical_floor` is the minimum cosine the lexical tier will accept. It
    gates only that tier; the cosine tier still uses `min_similarity` alone.
    Measured 2026-08-24, 0.40 removes the "who is the president" class of leak
    at no cost to coverage, while 0.45 and 0.50 start dropping real answers
    (terse coverage 98 -> 96 -> 93), so the floor is deliberately low: it is a
    sanity floor, not a relevance gate.

    `lexical_min=None` disables the lexical tier and restores the pure-cosine
    behaviour exactly, which is what the tests that predate it assert.

    The returned score is the COSINE similarity, not the blended rank score:
    callers (bot prompts, eval output, ChatTurnLog) already interpret it that
    way, and reporting a blended number would silently change what
    `min_similarity` means to everyone reading the logs. A chunk admitted by
    the lexical tier therefore reports a cosine BELOW `min_similarity`, which
    is honest: that is the number, and the reason it was kept is not that
    number.
    """
    embeddings = embeddings or get_embeddings()
    [query_vector] = embeddings.embed([query])
    # No vectors at all (see `NullEmbeddings`): rank on the keyword half alone
    # rather than returning nothing. Everything below still runs -- the same
    # tokenising, the same typo repair, the same `_score` -- only the cosine
    # contribution is absent, which is what `lexical_only` accounts for.
    lexical_only = not query_vector
    # Everything the scan needs about the corpus, from a process-level cache
    # rather than from the database on every turn. See `_Corpus`.
    corpus = _corpus_for(destination)
    scoped, haystacks = corpus.ids, corpus.haystacks
    counts = corpus.counts if lexical_only else [None] * len(scoped)

    # Group each term with the corpus spellings it may be a misspelling of, so
    # one typo cannot close the lexical tier on an answerable question.
    query_terms = expand_terms(_terms(query), corpus.vocabulary)

    if lexical_only:
        # BM25 over the whole in-scope corpus, then a relative cut. The score is
        # unbounded and scales with how many rare terms matched, so an ABSOLUTE
        # threshold cannot separate "answered" from "off-topic" across questions
        # of different lengths -- but the SHAPE does: a question the corpus
        # answers has a few chunks far above the rest, and one it does not has a
        # flat tail of incidental word matches.
        lengths = [sum(c.values()) for c in counts]
        bm25 = bm25_scores(query_terms, counts, lengths)
        best = max(bm25) if bm25 else 0.0
        if best < BM25_MIN_SCORE:
            return []
        cut = best * BM25_RELATIVE_CUT
        ranked = sorted(
            ((chunk_id, 0.0, score) for chunk_id, score in zip(scoped, bm25)
             if score >= cut),
            key=lambda triple: -triple[2])
        return _rows(ranked[:top_k])

    # A question that survives stopword stripping as ONE term is where the
    # lexical tier is weakest: "every distinctive term present" is satisfied by
    # any chunk that happens to use that one word. "What is 2 plus 2?" reduces
    # to {"plus"}, matched 11 chunks on that basis, and the bot then answered
    # from its own knowledge with a Source line pointing at a course on
    # prescriptive analytics -- a fabricated-looking citation, which costs more
    # trust than the wrong answer does.
    #
    # Cosine cannot separate that from a real one-word lookup: the leak scored
    # 0.256 while "imunizations" scored 0.104, because a misspelling embeds
    # badly no matter how clearly it is meant. Raising the floor kills typo
    # repair, lowering it re-opens the leak. So single-term questions are
    # admitted on a different kind of evidence instead -- the term naming what
    # the document is ABOUT rather than merely occurring in it. Measured over
    # the resources corpus: "plus" appears in the heading or title of 0 chunks,
    # while "immunizations", "orientation", "transcript" and "zoom" appear in
    # 2, 7, 1 and 6.
    narrow = len(query_terms) < 2

    # One matrix product for every cosine at once, in place of a Python loop
    # over 5,889 chunks x 512 floats. Same numbers: unit-normalised rows dotted
    # with the unit-normalised query, and a zero-norm row scores 0.0 exactly as
    # `cosine` does.
    similarities = corpus.similarities(query_vector)
    scored = []
    for chunk_id, haystack, about_terms, similarity in zip(
            scoped, haystacks, corpus.abouts, similarities):
        keyword = _score(query_terms, haystack)
        # Two tiers, either of which admits. A question with no distinctive
        # terms at all scores 0.0 here, so it can only ever enter on cosine.
        if lexical_min is not None and keyword >= lexical_min:
            about = _score(query_terms, about_terms)
            # A title match also STANDS IN for the cosine floor, which is what
            # lets a misspelling through: the floor is there to catch chunks
            # matched by accident, and a term in the title is not an accident.
            lexical_hit = about >= 1.0 if narrow else (
                similarity >= lexical_floor or about >= 1.0)
        else:
            lexical_hit = False
        if similarity < min_similarity and not lexical_hit:
            continue
        rank = similarity + KEYWORD_WEIGHT * keyword
        scored.append((chunk_id, similarity, rank))

    scored.sort(key=lambda triple: -triple[2])
    return _rows(scored[:top_k])
