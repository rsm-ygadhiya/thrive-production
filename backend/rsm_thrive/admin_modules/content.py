"""Content management — the console's flagship: edit the catalog the bots plan
from, and re-ingest the corpus they answer from, without the shell.

Catalog (`data/catalog/*.json`) is edited in place with JSON validation, then
the service caches are cleared so the change is live. Corpus edits happen on
disk; the re-ingest action wraps the `ingest_corpus` command so the change
reaches `DocumentChunk` rows (and the embeddings the retriever scans).

Mounted as a custom admin page via an unmanaged model, so it appears in the
console index alongside the record admins.
"""

import io
import json
from pathlib import Path

from django.apps import apps as django_apps
from django.contrib import admin, messages
from django.core.exceptions import PermissionDenied
from django.core.management import call_command
from django.db import models
from django.http import Http404
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path

from rsm_thrive.services.courses import electives

from .access import is_thrive_admin

APP_DIR = Path(django_apps.get_app_config("rsm_thrive").path)
CATALOG_DIR = APP_DIR / "data" / "catalog"
CORPUS_DIR = APP_DIR / "data" / "corpus"
CATALOG_FILES = ("courses.json", "careers.json", "bundles.json")

HUB_URL = "admin:rsm_thrive_contenttool_changelist"


def _clear_caches():
    """Make a saved catalog edit visible to the bots immediately."""
    electives.forget_catalog()
    try:  # bundles are cached independently in some builds
        from rsm_thrive.services.courses import bundles
        for attr in ("forget_bundles", "load_bundles"):
            fn = getattr(bundles, attr, None)
            if fn is not None and hasattr(fn, "cache_clear"):
                fn.cache_clear()
    except Exception:
        pass


def _corpus_dirs():
    if not CORPUS_DIR.exists():
        return []
    return sorted(p.name for p in CORPUS_DIR.iterdir() if p.is_dir())


class ContentTool(models.Model):
    """Not a table — a hook to host the content-management admin pages."""

    class Meta:
        managed = False
        app_label = "rsm_thrive"
        verbose_name = "content management"
        verbose_name_plural = "Content management"

    def __str__(self):
        return "Content management"


@admin.register(ContentTool)
class ContentToolAdmin(admin.ModelAdmin):

    def has_module_permission(self, request):
        return is_thrive_admin(request.user)

    def has_view_permission(self, request, obj=None):
        return is_thrive_admin(request.user)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def get_urls(self):
        base = super().get_urls()
        custom = [
            path("catalog/<str:name>/", self.admin_site.admin_view(self.catalog_edit_view),
                 name="content_catalog_edit"),
            path("corpus/reingest/", self.admin_site.admin_view(self.corpus_reingest_view),
                 name="content_corpus_reingest"),
        ]
        return custom + base

    # The index entry lands here; render the hub instead of a table changelist.
    def changelist_view(self, request, extra_context=None):
        if not is_thrive_admin(request.user):
            raise PermissionDenied
        ctx = {
            **self.admin_site.each_context(request),
            "title": "Content management",
            "catalog_files": CATALOG_FILES,
            "corpus_dirs": _corpus_dirs(),
        }
        return TemplateResponse(request, "admin/content/hub.html", ctx)

    def catalog_edit_view(self, request, name):
        if not is_thrive_admin(request.user):
            raise PermissionDenied
        if name not in CATALOG_FILES:
            raise Http404("Unknown catalog file")
        target = CATALOG_DIR / name

        if request.method == "POST":
            raw = request.POST.get("content", "")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as err:
                messages.error(request, f"Not saved — invalid JSON: {err}")
                content = raw
            else:
                target.write_text(
                    json.dumps(parsed, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
                _clear_caches()
                messages.success(request, f"Saved {name} and refreshed the bots' caches.")
                return redirect(HUB_URL)
        else:
            content = target.read_text(encoding="utf-8") if target.exists() else "{}"

        ctx = {
            **self.admin_site.each_context(request),
            "title": f"Edit {name}",
            "name": name,
            "content": content,
        }
        return TemplateResponse(request, "admin/content/catalog_edit.html", ctx)

    def corpus_reingest_view(self, request):
        if not is_thrive_admin(request.user):
            raise PermissionDenied
        if request.method != "POST":
            return redirect(HUB_URL)

        directory = request.POST.get("directory", "").strip()
        also_catalog = bool(request.POST.get("catalog"))
        rescope = bool(request.POST.get("rescope"))

        valid = set(_corpus_dirs())
        if directory and directory not in valid:
            messages.error(request, "Unknown corpus directory.")
            return redirect(HUB_URL)
        if not directory and not also_catalog:
            messages.error(request, "Choose a corpus directory and/or the catalog.")
            return redirect(HUB_URL)

        out = io.StringIO()
        args = [str(CORPUS_DIR / directory)] if directory else []
        opts = {"stdout": out}
        if also_catalog:
            opts["catalog"] = True
        if rescope:
            opts["rescope"] = True
        try:
            call_command("ingest_corpus", *args, **opts)
        except Exception as err:
            messages.error(request, f"Re-ingest failed: {err}")
        else:
            tail = out.getvalue().strip().splitlines()[-3:]
            messages.success(request, "Re-ingest complete. " + " · ".join(tail))
        return redirect(HUB_URL)
