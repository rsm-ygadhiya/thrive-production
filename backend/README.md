# rsm-django-thrive

The Django backend component for THRIVE. Not started yet — the design it
implements is `../docs/specs/2026-08-21-thrive-backend-design.md`.

Will contain:

- `rsm_thrive/` — Django app: models, the `/msba-brain/api/thrive/` API,
  Celery tasks (Canvas sync, corpus ingestion, job pipeline), chatbot
  services, appointment side effects (Zoom, ICS email)
- `deploy/` — LaunchDaemon plists, Caddy snippet, sync-to-platform scripts

Developed here; deployed by installing into the `rsm-guild-ai-brain`
platform checkout on the server (branch `feat/msba-brain`).

## Chatbots: corpus, LLM backend, and eval

The three destination bots (`resources`/FAQ, `career`, `courses`/electives)
answer from an ingested corpus of retrieved passages plus an LLM. The FAQ bot
and the course planner (below) were ported from a collaborator's prototype —
github.com/rsm-ygadhiya/Thrive-Chatbots — and re-verified end-to-end against
this system's real corpus, retrieval, and chat write path; `career` is ours.

Two knobs control the backend:

- `THRIVE_LLM` — `tritonai` (default) or `fake`. `fake` makes `get_llm()`
  raise, which is the intended way to exercise the degraded/turn-rescue path
  without an API key; it also switches embeddings to `FakeEmbeddings` for
  retrieval.
- `TRITONAI_API_KEY` — required when `THRIVE_LLM=tritonai` (or unset). Used
  for chat completions (`TRITONAI_MODEL`, default `claude-sonnet-5`); the
  Resources bot's question reading can run on a faster model via
  `TRITONAI_READER_MODEL` (measured 2026-09-20: `gemini-3.5-flash` reads
  `read_question` as well as claude-sonnet-5 in about half the time -- but
  mapped an uncovered job onto a curated profile in the planner's
  `read_turn`, so that reading, and every answer, stays on
  `TRITONAI_MODEL`). It is
  **not** usable for embeddings: the key issued 2026-09-11 serves four chat
  models and no embedding model, so `THRIVE_EMBEDDINGS=local` is mandatory
  and `TRITONAI_EMBED_MODEL` is dead weight. Get a key
  from https://tritonai-api.ucsd.edu/ with your UCSD login — this may
  require campus network access (or VPN) off-campus. `THRIVE_LLM=fake` is
  unchanged for tests and needs no key. Copy `backend/.env.example` to
  `backend/.env` and fill in the key for local dev (auto-loaded by
  `config/settings.py`).

### Retrieval: hybrid rank, two admission tiers, typo repair

`rsm_thrive/services/resources/retrieval.py` is not pure cosine search. Every question
is scored against every in-scope `DocumentChunk` two ways:

- **Cosine similarity** against the question embedding — finds paraphrases.
- **Keyword score** — the fraction of the question's distinctive terms (stop
  words stripped) present in the chunk's heading + text.

A chunk is **admitted** if either tier clears its gate: cosine ≥
`min_similarity`, OR keyword score ≥ `lexical_min` (typically 1.0 — *every*
distinctive term of the question present in that one chunk) with cosine above
a low `lexical_floor` sanity check. The lexical tier exists because short,
typo-prone student questions ("how do i get zoom?", "library hours") often
score below the cosine gate even though the corpus answers them on page
one — absolute cosine tracks how well-formed the *question* is, not whether
the chunk answers it. Admitted chunks are then **ranked** by
`cosine + 0.25 × keyword_score` — cosine still dominates ordering; the
keyword term only lifts a chunk that shares distinctive words with the
question (calibrated so it rescues dense, number-heavy chunks like fee
tables without out-ranking genuine paraphrase matches).

Typo tolerance sits in front of the lexical tier: query terms 5+ characters
that don't appear in the corpus vocabulary are expanded to every corpus word
one edit away (`expand_terms`), so "zooom" still matches "zoom" without
relaxing the "every term present" rule for words that are already spelled
correctly. See the module docstring for the specific measurements (cosine
gaps, coverage deltas) behind every constant.

### Corpus: layout and ingesting the real material

`rsm_thrive/data/corpus/` holds the real, checked-in corpus (not a test
fixture) in three source directories plus the course catalog:

- `crawled/` — public UCSD/Rady policy pages (Blink, Student Financial
  Solutions, career.ucsd.edu, etc.), one `.md` file per page.
- `canvas/` — MSBA cohort Canvas pages (orientation, laptop guidelines,
  Zoom setup, program prep).
- `program/` — the published Plans of Study (11-month and 17-month) and
  where-to-find-syllabi guidance.
- `rsm_thrive/data/catalog/courses.json` — the course catalog, ingested with
  `--catalog` rather than as files.

Ingest a corpus directory (`.pdf`/`.md`/`.txt`) and/or the course catalog.
`ingest_corpus` is **not recursive**, so each source directory is ingested
separately:

```bash
for d in crawled canvas program thrive; do
  uv run python manage.py ingest_corpus rsm_thrive/data/corpus/$d
done
uv run python manage.py ingest_corpus --catalog
```

This produces roughly 260 `Document`s / 2,450 `DocumentChunk`s at 1024
embedding dimensions (TritonAI `api-tgpt-embeddings`). It is idempotent per
source (`file:<name>` / `catalog:<code>`), so re-runs just refresh chunks —
safe to schedule as a cron once F5 adds Celery. Ingestion only touches
`Document`/`DocumentChunk` — it never reads or writes `JobPosting`, so
re-ingesting the corpus never disturbs the Career feed. A crawled document's
`destinations` are derived from its kind and source host (`destinations_for`
in `ingest_corpus.py`): syllabi also reach `courses`, and
career.ucsd.edu/career.rady.ucsd.edu pages also reach `career`; use
`--rescope` to recompute `destinations` for already-ingested documents
without re-embedding anything.

The repo also carries a small fixture corpus at
`rsm_thrive/tests/fixtures/corpus/`, used only by tests.

Run the golden FAQ eval against whatever corpus is ingested:

```bash
THRIVE_LLM=fake uv run python manage.py eval_bots --llm fake   # deterministic, no API key
uv run python manage.py eval_bots --llm real                   # needs TRITONAI_API_KEY
```

`THRIVE_LLM=fake` is required alongside `--llm fake`: without it, retrieval
still instantiates `TritonAiEmbeddings` and needs an API key, even though the
bot's own LLM calls are faked.

It prints `PASS`/`FAIL` per case in `rsm_thrive/data/evals/faq_golden.json`
with the retrieved chunk ids, and exits non-zero on any regression — add
cases to that file as the corpus grows. A `must_refuse: false` case that
retrieves nothing fails as `no-retrieval`, which usually means the corpus is
missing material rather than the bot misbehaving. The shipped golden set is
calibrated to a policy-only corpus (the seeded handbook fixture); run against
a larger mixed corpus (e.g. the fixture plus `--catalog`) and fake-embedding
retrieval dilutes, so several cases fail by design — run the eval against the
corpus the goldens target, or grow/tune the goldens as the corpus grows.

### Course surface: one orchestrator, six routes

The `courses` destination is dispatched by `rsm_thrive/services/courses/orchestrator.py`.
There is no interview — a student types a real sentence and gets an answer to
that sentence. `rsm_thrive/services/courses/router.py` classifies the question (rules
first; one LLM call, running the `read_turn` skill, only for what the rules
cannot decide — see [Skills](#skills-what-the-model-is-trusted-to-read)) and
the orchestrator dispatches:

| Route | What answers it |
| --- | --- |
| `role` | The curated bundle from `bundles.json` for one of the 14 profiles in `careers.json`. **No model call at all** when the turn names a title, alias or shorthand the table lists — the mapping is authoritative and must not become probabilistic. When nothing on disk matches, the classifier is shown the 14 profiles and may read the student's words as one of them, by id from that closed list (`services/courses/understanding.py`); the code validates the id and the reply confirms the reading back. A job no profile covers is refused and names no course. |
| `industry` | `grounded_course_advisor` — current skills for the field from the web, matched only against our catalog by `skill_match` (below). No catalog match means a refusal that names no course. |
| `factual` | `courses.json` rows plus syllabus retrieval over the `courses` destination, answered under `FACTUAL_SYSTEM`. Nothing to ground it means a refusal, and the model is not even called. |
| `combination` | Role determinism for the spine, industry context on the electives — in that order, so the deterministic half leads. |
| `situational` | `rsm_thrive/services/courses/situation.py`. See below. |
| `quarter` | One named quarter, planned on its own — "what should I take in Winter?", "just plan my spring". The whole plan is still built underneath (which electives are left in Winter depends on what Fall and Spring took); only that quarter is shown, with its own load question, and `next quarter` carries on from there. |
| `out-of-scope` / `unclear` | Says so plainly, or asks once. A greeting is matched deterministically (no model call — measured at 1.8s to classify "hi"), and an `unclear` turn in a conversation that has established nothing re-asks the opening's two questions instead of showing a menu. |

Every reply carries its `route` and (when a model decided) its confidence onto
`ChatTurnLog`, so a bad answer is a finding about a route rather than a
complaint about a black box. A rule's confidence is `None`, not `1.0`.

**The rule the whole surface holds:** no fact about a course, a unit count, a
prerequisite or a deadline is stated unless it came from the catalog, the
published skeletons, or a retrieved passage. The model interprets and explains;
it never asserts a number.

#### Skills: what the model is trusted to read

Every prompt the two bots give the model is a skill: one folder with a
`SKILL.md` — the instruction the model receives, with `<<placeholder>>`s for
closed lists rendered live from the catalog — under the bot it belongs to,
`rsm_thrive/services/courses/skills/` or `rsm_thrive/services/resources/skills/`.
`rsm_thrive/services/skills.py` is the shared runner (`read` for JSON readings,
`say` / `say_json` for prose); `rsm_thrive/services/courses/reading.py` binds it
to the planner's folder and holds the planner's validators.

| Skill | Called from | Does |
| --- | --- | --- |
| `read_turn` | `router.model_route`, after `rule_route` returns None; `router.read_position` for the situational route's follow-up | route + confidence; `act`; `track`, `load`, `completed_codes`, `skills`, `wants_menu`; the curated profile id and the curated industry id the words denote; the position (`previous_track`, `current_quarter`, `finish_on_time`); `menu_pick` positions and `menu_compare`; with a plan on screen, `plan_action` with `courses`, `load`, `quarter`, `route_choice` |
| `answer_course_facts` | `orchestrator._answer_factual` | writes the factual answer from the rendered catalog rows and passages |
| `explain_position` | `situation.explain` | writes the trade-offs in a computed ledger |
| `small_talk` | `replies._small_talk_reply` | two sentences once a plan is on screen |
| `read_question` (Resources) | `resources.bot.answer_faq`, first, on every question | `in_scope`, `search_terms` + `topic` for retrieval, `needs_context` + `clarifying_question`, a campus-scoped `web_query` |
| `answer_question` (Resources) | `resources.bot._write` | JSON `answered` + `reply`, citing passages `[n]` and web results `[Wn]`; the reply is labelled by what it cited |
| `career_advice` (Resources) | `resources.bot.answer_career` | the career bot's advice, passages rendered when retrieved |

Why a skill and not another regex: every word list on this surface was grown
from a live failure (`git log` reads as a list of them), and a list can only
hold phrasings already seen. The rules stay first — free, instant,
deterministic — and the skill reads what they could not.

What keeps it safe:

- **Context.** `orchestrator.turn_context` renders the planner's own state —
  the question asked last, what is on file, the plan on screen with its codes,
  the walk-through quarter — above the student's message, labelled as ours.
  "17" and "heavy" are read against the question they answer.
- **Validation.** Every field passes a validator in `courses/reading.py`
  (`track_of`, `load_of`, `courses_of`, `ratings_of`; `one_of` and the other generic ones in `skills.py`) that
  drops anything outside its closed set. A course named by title resolves
  only when exactly one catalog row matches. `Route.learned`, `Route.action`
  and `Route.act` carry only what survived.
- **Same machinery.** `_stated_by` refuses a question (`act == "ask"` or
  `planner.is_question`) and the situational route's own turn; what is left
  goes through `_intake_progress`, exactly as a typed "17 month" does.
  `_plan_action` calls the same `_review_reply`, `_finalise_reply`,
  `_handle_change_request` and `_apply_quarter_load` that `_maintenance`
  does. The goal is never taken from a reading: a role the model read goes
  through `_answer_role`, which confirms it back first.
- **One writer, labelled by what it used** (`resources.bot._write`). Passages
  are numbered `[n]`, web results `[Wn]`; a reply citing only passages lists
  their documents as sources, one citing only W carries the web notice and the
  pages read, one citing both shows both. When the best passage is under
  `web_race_below` the web results are in the prompt from the start, so thin
  support is one model call. Strong passages that decline get one more call
  with the web results (a clean slate if the search found nothing).
- **Ranking after retrieval** (`resources.bot._rerank`). Retrieval decides
  what is in; this decides the order and the mix: a title or heading sharing a
  content word with the question is boosted, the programme's own pages
  (Rady, MSBA, the cohort's Canvas) are boosted on top of that and only
  then, one document holds at most three of `top_k` slots, and the
  provenance chunk is dropped. Four queries feed it: the question, the
  reading's `search_terms`, and its `topic`. Measured 2026-09-20 on "how to
  register for classes": the cohort's RadyReady page went from unranked to
  first. `_cited` then lists only the documents the reply cited by number.
- **One reading for the planner.** The situational route's position (track
  now, track before, current quarter, finishing on time) is read by
  `read_turn`; `Route.position` carries it so a classified turn is read once,
  and `router.read_position` reads the follow-up turn with the same skill.
- **The plan-is-imminent read.** Once the load question has been asked, any
  turn delivers the plan with what was assumed said aloud. `_plan_is_imminent`
  spots that state and runs the skill *before* the build, so "not too heavy
  please" becomes the load the plan is built on rather than a remark it
  ignored.
- **Outage is outage.** `Reading.degraded` becomes the `degraded` route, never
  a guess and never a content refusal.

`tests/test_reading_skills.py` holds the contract: files load and render with
no placeholder left; every validator drops what is outside its set; the old
four-key classifier answers still parse (every existing test scripts them);
and each action lands through the same code the rules reach.

REST surface (`rsm_thrive/views/planner.py`, all under
`/api/thrive/plan*`, all `@profile_required`):

- `GET /plan/intake` — the interview questions, this student's answers so
  far, `hasPlan`, and `starter` (the opening question — what `/ask/courses`
  shows before the student has said anything).
- `GET /plan` — the built plan (404 `no_plan` before the intake is answered,
  409 `intake_incomplete` mid-interview) as structured JSON plus a
  ready-to-render `markdown` field.
- `POST /plan` — submit a full `answers` object directly (bypassing the chat
  interview); starts a fresh plan and clears prior swaps, since they were
  chosen against a different intake.
- `DELETE /plan` — clear this student's plan.
- `GET /plan/alternatives?quarter=&slot=` — up to 4 alternative courses for
  one elective slot.
- `POST /plan/swap` — `{quarter, slot, courseId}`; swaps one elective,
  pinning the rest.

## Jobs: ingest, sources, feed, and match reports

The Career tab is a ranked, no-query feed backed by:

- `GET /api/thrive/jobs?q=` — free-text job search.
- `GET /api/thrive/jobs/feed` — the Career tab's feed: no query required,
  ranks all active postings against the caller's current resume, overlays
  cached `MatchReport`/`PostingInteraction` state, and splits into three tabs
  (`?tab=recommended|liked|all`, default `recommended`) plus `?min_score=`
  and per-tab `counts`. `recommended` excludes dismissed postings; `all`
  and the counts still include them. With no current resume, ranking falls
  back to recency order and the response's `profileAvailable` is `false`.
- `GET /api/thrive/jobs/<id>` — job detail plus the role benchmark.
- `POST /api/thrive/jobs/<id>/like` / `POST /api/thrive/jobs/<id>/dismiss` —
  per-user, per-posting toggles (`PostingInteraction`); each call flips the
  flag and returns the new `liked`/`dismissed` state. Both require a student
  profile, same as the feed.
- `POST /api/thrive/jobs/<id>/report` — an LLM match report, cached per
  `(user, posting, resume_version)`.
- `POST /api/thrive/resume/upload` — resume ingestion feeding all of the
  above.

Same two knobs as the chatbots — `THRIVE_LLM=fake`/`tritonai` and
`TRITONAI_API_KEY` — govern all of it: `fake` makes `get_llm()` raise, so
`job_report` and `resume_upload` fail honest with a `503 llm_unavailable`
rather than a silent or fabricated result, and switches embeddings to
`FakeEmbeddings` for ranking/ingestion. Switching `THRIVE_LLM` (`fake` ↔
real) — or switching embedding backends generally, e.g. after correcting
`TRITONAI_EMBED_MODEL` — changes embedding dimensions, so re-run
`ingest_corpus`/`ingest_jobs` after switching — otherwise ranking/retrieval
silently degrades (a dim-mismatch warning is logged, but postings/chunks
keep their stale, wrong-dimension embeddings until re-ingested).

**Sources** (`rsm_thrive/services/jobs/sources.py`) sit behind one `JobSource`
ABC (`fetch() -> list[dict]`), four of them backing the shipped
`companies.json` (~20 boards):

- `GreenhouseSource` / `LeverSource` — public, keyless JSON boards
  (`boards-api.greenhouse.io`, `api.lever.co`).
- `AshbySource` / `WorkableSource` — public, keyless JSON boards
  (`api.ashbyhq.com/posting-api/job-board/<board>`,
  `apply.workable.com/api/v1/widget/accounts/<account>`), same shape as the
  above two.
- `FakeJobSource` — an in-memory source for tests and offline seeding; takes
  the same row shape the real sources normalize to.
- An aggregator source (Adzuna/JSearch) slots in later behind the same ABC
  once a free API key exists (backlog).

Sources are configured per company in `rsm_thrive/data/jobs/companies.json`
under the `greenhouse`/`lever`/`ashby`/`workable` keys. Greenhouse, Lever,
and Ashby boards are configured today; `WorkableSource` is code-ready but has
no boards listed yet.

**Ingest** (`ingest_from` in `rsm_thrive/services/jobs/ingest.py`) is
idempotent per `(source, external_id)`: it fetches each configured source,
upserts postings, recomputes `skills` (`services/jobs/skills.py`,
vocabulary in `rsm_thrive/data/jobs/skills_vocab.json`), and deactivates
postings from sources that succeeded this run but weren't seen (stale
expiry). A source that fails to fetch is skipped — its existing postings are
left untouched rather than deactivated. Embedding is hash-skipped: each row's
`sha256(title + "\n" + description)` is compared against the stored
`content_hash`, and only postings whose content actually changed are
re-embedded (title+description, truncated to 2000 chars) — unchanged rows
just get `last_seen_at`/`active` refreshed. On a repo with thousands of live
postings this makes re-runs fast and cheap: a first run against the full
`companies.json` embeds everything, a same-day second run embeds
essentially nothing.

```bash
uv run python manage.py ingest_jobs                    # real boards, needs TRITONAI_API_KEY for embeddings
THRIVE_LLM=fake uv run python manage.py ingest_jobs     # real boards, fake (deterministic) embeddings
```

Seed postings without any network access — useful for local smoke and demos —
by calling `ingest_from` directly with a `FakeJobSource`:

```bash
THRIVE_LLM=fake uv run python manage.py shell -c "
from rsm_thrive.services.jobs.ingest import ingest_from
from rsm_thrive.services.jobs.sources import FakeJobSource
from rsm_thrive.services.embeddings import get_embeddings
rows = [{'external_id': '1', 'title': 'Data Analyst', 'company': 'Acme',
         'location': 'Remote', 'url': 'https://example.com/1',
         'description': 'SQL, Python, and Tableau reporting.',
         'posted_at': None}]
print(ingest_from([FakeJobSource(rows)], embeddings=get_embeddings()))
"
```

`companies.json` is a starter list of MSBA-target employers' Greenhouse,
Lever, Ashby, and Workable board slugs — expand it over time, and correct or
drop any slug that starts 404ing (boards get renamed or migrate off the
platform; e.g. Netflix's public Lever board no longer resolves and was
dropped rather than guessed at). F5 schedules `ingest_jobs` as a nightly
Celery task; the command's idempotency is what makes that safe to run
unattended.

Search ranks by cosine similarity between the resume embedding and each
posting's embedding, blended with skill overlap
(`rsm_thrive/services/jobs/search.py`); with no current resume it falls back
to recency ordering and the response's `profileAvailable` is `false`. The
role benchmark (top skills by share, for a title) is portable icontains
filtering today — an F5 swap inside `_matching_postings` moves it to
Postgres full-text search once the corpus outgrows that.

Resume upload is PDF-only (`pypdf` text extraction; a scanned/image-only PDF
comes back `400 unreadable_resume` before any LLM call) and DOCX support is
deferred (`python-docx` is a small add when wanted). A match report is cached
per `(user, posting, resume_version)` — re-requesting it after a resume
change bypasses the cache since the version changed, but repeat requests
against the same resume version hit the cache instead of re-billing the LLM.
