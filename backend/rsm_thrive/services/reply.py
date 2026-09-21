"""What a bot hands back, and the two helpers both bots use to ground it.

Shared by the Resources bot (`services/resources/bot.py`) and the course
planner (`services/courses/`): one reply shape, so `views/chat.py` and
`ChatTurnLog` see one thing whichever bot answered, and the passage/source
helpers that both prompts are grounded with.
"""

from dataclasses import dataclass, field


@dataclass
class BotReply:
    body: str
    chunk_ids: list = field(default_factory=list)
    model_note: str = "llm"
    # Choices to offer as buttons alongside this reply. Empty for a free-text
    # answer; a question with a fixed set of answers should not require typing.
    quick_replies: list = field(default_factory=list)
    # A small form to offer instead, when one control per item beats a row of
    # buttons. None for every other reply. See `planner.rating_form_for`.
    form: dict | None = None
    # Which route the orchestrator took to produce this, and how sure it was.
    # `route_confidence` stays None when a rule decided, because a rule has no
    # confidence to report and recording 1.0 would make the two look alike in
    # the report that exists to tell them apart. See `services/router.py`.
    route: str = ""
    route_confidence: float | None = None
    # This turn declined to answer. Set at every site that declines, rather
    # than inferred downstream from `model_note`, because the refusal report is
    # a content backlog and a backlog assembled by parsing prose loses rows
    # silently the first time the prose changes.
    refused: bool = False
    # What this reply SAYS, when the rest of its body is a re-render a caller
    # may replace with its own. Set by a swap: the body is "Swapped X for Y"
    # plus a note on where X can still go plus the whole plan, and the
    # walk-through wants the first two over its own quarter, not the third.
    lead: str = ""
    # The plan this reply renders, for a caller that needs to inspect what was
    # actually scheduled rather than re-derive it. Set only by `_plan_reply`;
    # None everywhere else. See `orchestrator._uncurated_coverage_note`, which
    # measures coverage against the courses that survived the unit limits, and
    # would otherwise have to rebuild the plan to find out what they were.
    plan: dict | None = None


def build_context(hits):
    lines = []
    for n, (chunk, _score) in enumerate(hits, start=1):
        title = chunk.document.title
        head = f"[{n}] {title} — {chunk.heading}" if chunk.heading else f"[{n}] {title}"
        lines.append(f"{head}\n{chunk.text}")
    return "\n\n".join(lines)


def append_sources(body, hits):
    """Append the sources behind an answer, as links where a URL is known.

    A student who is about to act on a deadline should be able to open the page
    that states it — "Sources: Registration Fees" is a claim, a link is
    checkable. Documents ingested without a `source_url` (fixtures, pasted
    material) still list by title alone.

    One source stays inline; several become a bulleted list. A comma-joined run
    was fine at three sources and unreadable at ten — a single line holding
    "Where to Find MSBA Plans of Study, Course Schedules and Syllabi, Analytical
    Writing Program — awp.ucsd.edu, How to Enroll in Individual Classes at
    UCSD (Community College and CSU Students) — students.ucsd.edu, ..." cannot
    be scanned, and worse, the titles themselves contain commas, so the reader
    cannot tell where one source ends and the next begins. `top_k` for the FAQ
    bot is 10, so ten is the normal case, not the pathological one.

    The list is Markdown the frontend already renders: `RichMessage` turns `- `
    lines into a real `<ul>` and parses links inside list items, so this arrives
    as clickable bullets rather than as literal hyphens.
    """
    if not hits:
        return body
    seen, entries = set(), []
    for chunk, _score in hits:
        document = chunk.document
        if document.title in seen:
            continue
        seen.add(document.title)
        url = (document.source_url or "").strip()
        entries.append(f"[{document.title}]({url})" if url else document.title)
    if len(entries) == 1:
        return f"{body}\n\nSource: {entries[0]}"
    listed = "\n".join(f"- {entry}" for entry in entries)
    return f"{body}\n\nSources:\n{listed}"
