import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Tiny stdlib .env loader (no python-dotenv dependency): backend/.env, if
# present, seeds os.environ for local dev. Real env vars always win —
# setdefault never overrides something already exported. Not used/needed in
# production, where the platform injects env vars directly.
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _value = _line.partition("=")
        os.environ.setdefault(_key.strip(), _value.strip())

SECRET_KEY = os.environ.get("THRIVE_SECRET_KEY", "dev-only-insecure")
DEBUG = os.environ.get("THRIVE_DEBUG", "1") == "1"
# Extra hosts (comma-separated) let a temporary tunnel or preview host reach
# the API; the localhost defaults always stay so normal dev never breaks.
ALLOWED_HOSTS = ["127.0.0.1", "localhost"] + [
    host.strip()
    for host in os.environ.get("THRIVE_EXTRA_HOSTS", "").split(",")
    if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rsm_thrive",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",
        "django.contrib.auth.context_processors.auth",
        "django.contrib.messages.context_processors.messages",
    ]},
}]

# SQLite by default (tests, quick local). THRIVE_PG=1 switches to Postgres —
# the shape production uses (spec §7: nothing debugged first on the server).
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}}
if os.environ.get("THRIVE_PG") == "1":
    DATABASES["default"] = {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("PGDATABASE", "thrive_dev"),
        "USER": os.environ.get("PGUSER", "thrive"),
        "PASSWORD": os.environ.get("PGPASSWORD", "thrive"),
        "HOST": os.environ.get("PGHOST", "127.0.0.1"),
        "PORT": os.environ.get("PGPORT", "5432"),
    }

LANGUAGE_CODE = "en-us"
TIME_ZONE = "America/Los_Angeles"
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
DEFAULT_FROM_EMAIL = os.environ.get("THRIVE_FROM_EMAIL", "thrive-noreply@rady.ucsd.edu")

# No EMAIL_BACKEND was configured before Django's MAILERS setting existed, so
# email implicitly used the SMTP backend. MAILERS makes that explicit (Django
# test runner overrides "default" to locmem for the test suite regardless).
MAILERS = {
    "default": {
        "BACKEND": "django.core.mail.backends.smtp.EmailBackend",
    },
}

THRIVE_DEV_LOGIN_ENABLED = os.environ.get("THRIVE_DEV_LOGIN", "1") == "1"
THRIVE_FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "THRIVE_FRONTEND_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://localhost:3123",
    ).split(",")
    if origin.strip()
]

# Browsers send an Origin header on POSTs and Django rejects any origin it
# does not trust — so every frontend origin (including a temporary tunnel
# host passed via THRIVE_FRONTEND_ORIGINS) must also be CSRF-trusted.
CSRF_TRUSTED_ORIGINS = [
    origin for origin in THRIVE_FRONTEND_ORIGINS if origin.startswith("http")
]

# Which chat backend answers a turn:
#
#   "codex"    LOCAL DEV DEFAULT. A personal ChatGPT/Codex subscription over
#              OAuth, needing no API key at all. See `CodexOAuthLLM`.
#   "tritonai" What the DEPLOYED service must run. Needs TRITONAI_API_KEY.
#   "fake"     Tests, which inject their own FakeLLM.
#
# The default is `codex` because the TritonAI keys this project is issued carry
# a fixed one-time budget with no reset, so an exhausted key stops all local
# work on the chatbots and there is nothing to wait for.
#
# DEPLOYMENT MUST SET THRIVE_LLM=tritonai EXPLICITLY. The codex backend
# authenticates as a person rather than as a metered service and talks to an
# endpoint with no compatibility promise; it is a laptop convenience, not a
# production dependency. `docs/VINCENT-ASKS.md` already lists the API key as
# something we supply, so the server environment is the place that pins this.
THRIVE_LLM = os.environ.get("THRIVE_LLM", "codex")
CODEX_MODEL = os.environ.get("CODEX_MODEL", "gpt-5.4")
CODEX_TIMEOUT_SECONDS = int(os.environ.get("CODEX_TIMEOUT_SECONDS", "90"))
# Shared-credential serving, for a hosted TEST instance. Point this at a
# credential file copied from a developer's ~/.codex/auth.json; every request
# then uses that one account and no student ever signs in. Unset, the app falls
# back to the Codex CLI's own file, which is what a laptop wants.
#
# TREAT THE FILE AS A SECRET -- it holds a long-lived refresh token for a
# personal ChatGPT account.
CODEX_CREDENTIALS_PATH = os.environ.get("CODEX_CREDENTIALS_PATH", "")
# Whether this host may open a browser to sign in. Unset means "only if a
# terminal is attached", which keeps a laptop convenient and stops a headless
# server hanging a worker on a sign-in nobody can see.
CODEX_ALLOW_BROWSER_LOGIN = (
    None if os.environ.get("CODEX_ALLOW_BROWSER_LOGIN") is None
    else os.environ.get("CODEX_ALLOW_BROWSER_LOGIN") == "1")
# Embeddings are a SEPARATE provider from chat -- the codex backend has none.
# Unset, this follows THRIVE_LLM. See services/embeddings.py.
THRIVE_EMBEDDINGS = os.environ.get("THRIVE_EMBEDDINGS", "")
# The on-machine embedding model used when THRIVE_EMBEDDINGS resolves to
# "local" (which THRIVE_LLM=codex does). ~123MB, downloaded once from HF.
THRIVE_LOCAL_EMBED_MODEL = os.environ.get(
    "THRIVE_LOCAL_EMBED_MODEL", "minishlab/potion-retrieval-32M")
# Retrieval thresholds are calibrated PER ENCODER, so the overlay follows the
# embedding backend rather than being set by hand. An explicit THRIVE_BOT_CONFIG
# always wins -- that is the deploy-free tuning hook and this must not shadow it.
_embed_backend = THRIVE_EMBEDDINGS or THRIVE_LLM
THRIVE_BOT_CONFIG = os.environ.get("THRIVE_BOT_CONFIG", "")
if not THRIVE_BOT_CONFIG and _embed_backend in ("local", "codex"):
    THRIVE_BOT_CONFIG = str(BASE_DIR / "config" / "bots.local-embed.json")
# Who performs a web search when a backend cannot do it itself. Only the codex
# backend searches natively (OpenAI hosts the loop); TritonAI is vLLM behind
# LiteLLM and has no such loop, so without this `search_chat` silently answered
# from training data. See services/websearch.py.
#
#   "duckduckgo"  no key, and therefore the default. Parses a public HTML
#                 endpoint, so it is the one that breaks on a markup change.
#   "brave" | "serper" | "tavily"   keyed, via THRIVE_SEARCH_API_KEY.
#   "none"        no search. `search_chat` degrades to the model's own memory,
#                 which is what the whole codebase did before this existed.
THRIVE_SEARCH = os.environ.get("THRIVE_SEARCH", "duckduckgo")
THRIVE_SEARCH_API_KEY = os.environ.get("THRIVE_SEARCH_API_KEY", "")
TRITONAI_API_KEY = os.environ.get("TRITONAI_API_KEY", "")
TRITONAI_MODEL = os.environ.get("TRITONAI_MODEL", "claude-sonnet-5")
# The model the READING skills run on (`services/skills.py` `read`): the
# small JSON extractions that decide a route or re-word a question. Unset,
# they run on TRITONAI_MODEL. A faster model here roughly halves the time a
# reading adds to a turn; the answers themselves always use TRITONAI_MODEL.
TRITONAI_READER_MODEL = os.environ.get("TRITONAI_READER_MODEL", "")
# Per call, and for the whole retry ladder. The SDK's own default is ten
# minutes, which is not a timeout for somebody waiting on a chat reply, and
# three attempts plus the 15s and 30s waits between them took a measured 149
# seconds during one connection blip. The budget ends the ladder early rather
# than letting it finish long after the honest answer became "I can't reach
# the model right now".
TRITONAI_TIMEOUT_SECONDS = float(os.environ.get("TRITONAI_TIMEOUT_SECONDS", "45"))
TRITONAI_TURN_BUDGET_SECONDS = float(
    os.environ.get("TRITONAI_TURN_BUDGET_SECONDS", "75"))
# NOT A DEFAULT THAT WORKS. The key issued 2026-09-11 serves four CHAT models
# and no embedding model at all, so any value here 404s; THRIVE_EMBEDDINGS=local
# is what actually runs. Kept only so the `tritonai` embeddings backend has a
# name to fail on loudly rather than a blank one to fail on obscurely.
TRITONAI_EMBED_MODEL = os.environ.get("TRITONAI_EMBED_MODEL", "api-tgpt-embeddings")
