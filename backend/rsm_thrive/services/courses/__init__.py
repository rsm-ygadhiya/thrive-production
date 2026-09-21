"""The course planner: everything behind `/ask/courses`.

`orchestrator` dispatches a turn; `router` names its route; `planner`,
`bundles`, `electives` and `situation` do the arithmetic; `replies` maintains
a plan once it exists; `reading` is what the model is trusted to read, with
its skill files under `skills/`. See `docs/HOW-THE-BOTS-WORK.md`, Part 2.
"""
