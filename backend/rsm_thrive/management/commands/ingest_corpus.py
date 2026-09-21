import json
import re
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from rsm_thrive.services.embeddings import get_embeddings
from rsm_thrive.services.resources.ingest import extract_pdf_text, ingest_document

CATALOG = Path(__file__).resolve().parents[2] / "data" / "catalog" / "courses.json"

# The pipeline's exporters write a `Source: <url>` line at the top of every
# file (see Thrive/pipeline/5-report/export-*.ts). Capturing it here is what
# lets an answer link a student to the authoritative page instead of only
# naming it. Files without the line ingest exactly as before, with no URL.
SOURCE_LINE = re.compile(r"^Source:\s*(https?://\S+)", re.MULTILINE)

# "Some Page 2" / "Some Page 10" — Finder/iCloud duplicate naming.
ICLOUD_CONFLICT = re.compile(r".+ \d{1,2}$")

# Hosts whose pages ARE the career material. A document from one of these is
# reachable by the career bot as well as the FAQ bot.
#
# Without this every crawled page landed in "resources" alone, which left the
# career destination with an empty corpus: 62 career.ucsd.edu and
# career.rady.ucsd.edu documents were sitting in the database and the career bot
# could not retrieve one of them. It answered from the model's own knowledge
# instead, and said so out loud — "Based on general knowledge of Rady's career
# services (I don't have specific numbered context passages to cite)" — which is
# the bot guessing at a real programme's offering.
CAREER_HOSTS = frozenset({"career.ucsd.edu", "career.rady.ucsd.edu"})

# What marks a markdown file in `corpus/syllabi/` as a course rather than a
# policy page. Read from the frontmatter rather than from the directory name,
# so a syllabus keeps its kind wherever it is ingested from.
SYLLABUS_FRONTMATTER = re.compile(r"\A---\n(?:.*\n)*?code: \"", re.M)
COURSE_CODE_FM = re.compile(r'^code: "([^"]+)"', re.M)
COURSE_TITLE = re.compile(r'^title: "([^"]+)"', re.M)


def _host_of(url):
    match = re.match(r"https?://([^/]+)", url or "")
    return match.group(1).lower() if match else ""


def destinations_for(kind, source_url):
    """Which bots can retrieve this document.

    Most things stay in "resources" — the FAQ bot is the general surface and a
    student asking it about resumes should still be answered. The extra
    destination is additive.

    A SYLLABUS is the exception, and goes to "courses" alone. Full syllabus text
    is 3,385 chunks against the FAQ corpus's 2,421, and it is dense with exactly
    the words a student uses to ask about enrolment: every syllabus says "class",
    most say "book". Measured on 2026-09-05, "how to book a class" returned MGT
    408 Finance at 0.618 and pushed "Booking Your Classes — students.ucsd.edu"
    off the results entirely; the bot then said it had no steps for booking a
    class while holding the page that has them.

    The FAQ bot does not lose course coverage by this. The `catalog` documents
    are one short entry per course and stay in "resources", which is the right
    granularity for a general surface: enough to say what MGTA 458 is, and not
    enough to bury the fee tables.
    """
    if kind == "syllabus":
        return ["courses"]
    destinations = ["resources"]
    if _host_of(source_url) in CAREER_HOSTS:
        destinations.append("career")
    return destinations


def source_url_of(text):
    match = SOURCE_LINE.search(text[:2000])
    if not match:
        return ""
    # Trailing punctuation and the parenthetical notes the Data Dump exporter
    # adds ("(official Rady Canvas — SSO-gated...)") are not part of the URL.
    return match.group(1).rstrip(").,;")


class Command(BaseCommand):
    help = "Ingest a corpus directory (PDF/md/txt) and optionally the course catalog."

    def _rescope(self):
        """Re-derive `destinations` for every document already in the database."""
        from rsm_thrive.models import Document

        changed = 0
        for document in Document.objects.all():
            wanted = destinations_for(document.kind, document.source_url)
            # Only widen what the crawler decides. A document ingested from the
            # fixture corpus or the course catalog has destinations set by that
            # path, not by a host, and must not be quietly narrowed to
            # "resources" here.
            #
            # A SYLLABUS is exempt: its destination comes from its kind rather
            # than from a host, so `destinations_for` is authoritative for it
            # whether or not it carries a source URL. Without this exemption
            # the markdown syllabi -- which have no URL -- were silently
            # skipped, and the rescope reported "0 documents" while the corpus
            # it was meant to fix stayed exactly as it was.
            if not document.source_url and document.kind != "syllabus":
                continue
            if sorted(document.destinations or []) == sorted(wanted):
                continue
            self.stdout.write(f"  {document.title[:56]}: "
                              f"{document.destinations} -> {wanted}")
            document.destinations = wanted
            document.save(update_fields=["destinations"])
            changed += 1
        self.stdout.write(self.style.SUCCESS(
            f"rescoped {changed} document(s); no embeddings recomputed"))

    def add_arguments(self, parser):
        parser.add_argument("directory", nargs="?", default="")
        parser.add_argument("--catalog", action="store_true")
        parser.add_argument(
            "--rescope", action="store_true",
            help="Recompute which bots can see each document, without "
                 "re-embedding anything. `destinations` is a document field, so "
                 "fixing it does not need the 2,400 embedding calls a full "
                 "re-ingest would spend.")

    def handle(self, *args, **options):
        if options["rescope"]:
            self._rescope()
            return
        embeddings = get_embeddings()
        directory = options["directory"]
        if not directory and not options["catalog"]:
            raise CommandError("Give a corpus directory, --catalog, or both.")

        if directory:
            root = Path(directory)
            if not root.is_dir():
                raise CommandError(f"{directory} is not a directory.")
            for path in sorted(root.iterdir()):
                # iCloud syncs ~/Desktop and writes conflict copies named
                # "Some Page 2.md" whenever it races a directory rewrite. Every
                # one of those becomes a SECOND Document for the same page,
                # doubling the corpus and letting retrieval return the same
                # passage twice. Skipping them here is the durable fix; relying
                # on someone remembering to delete them is not.
                if ICLOUD_CONFLICT.match(path.stem):
                    self.stdout.write(self.style.WARNING(
                        f"skipped {path.name} (looks like an iCloud conflict copy)"))
                    continue
                if path.suffix.lower() == ".pdf":
                    text = extract_pdf_text(path)
                    kind = "syllabus"
                elif path.suffix.lower() in (".md", ".txt"):
                    text = path.read_text()
                    # A markdown file IS a syllabus when it says so in its
                    # frontmatter. `corpus/syllabi/` holds one file per course
                    # -- the source `build_catalog` compiles courses.json from
                    # -- and those must reach the COURSES bot, which "policy"
                    # would not: a course the planner can schedule and the bot
                    # cannot describe is the split this directory removes.
                    kind = "syllabus" if SYLLABUS_FRONTMATTER.search(text[:400]) \
                        else "policy"
                else:
                    continue
                url = source_url_of(text)
                # A syllabus is titled by the course, not by the filename. The
                # title is what `append_sources` prints under an answer, and
                # "MGTA-402" tells a student nothing that "MGTA 402 —
                # Data-Driven Communications" does not tell them better.
                title = path.stem
                if kind == "syllabus":
                    named = COURSE_TITLE.search(text[:1200])
                    coded = COURSE_CODE_FM.search(text[:1200])
                    if named and coded:
                        title = f"{coded.group(1)} — {named.group(1)}"
                doc = ingest_document(f"file:{path.name}", title, kind,
                                      destinations_for(kind, url), text,
                                      embeddings, source_url=url)
                self.stdout.write(f"ingested file:{path.name} "
                                  f"({doc.chunks.count()} chunks)")

        if options["catalog"]:
            for course in json.loads(CATALOG.read_text()):
                parts = [course.get("description", "")]
                for offering in course.get("offerings", []):
                    parts.append(
                        f"Offered {offering.get('term', '')} "
                        f"with {offering.get('instructor', '')}. "
                        f"{offering.get('format_notes', '')}")
                if course.get("units_note"):
                    parts.append(course["units_note"])
                doc = ingest_document(
                    f"catalog:{course['code']}",
                    f"{course['code']} — {course['title']}",
                    "catalog", ["resources", "courses"],
                    "\n\n".join(p for p in parts if p), embeddings)
                self.stdout.write(f"ingested catalog:{course['code']} "
                                  f"({doc.chunks.count()} chunks)")
