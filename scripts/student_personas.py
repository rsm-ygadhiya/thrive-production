"""Realistic MSBA student conversations, printed in full for a human to read.

The other two harnesses assert. This one does not: it drives the surface the
way a student actually types -- hedged, run-on, mid-thought -- and prints the
transcripts so somebody can judge whether the answers are any good.

    cd backend && PYTHONPATH=$PWD uv run python ../scripts/student_personas.py [chars]

Everything it has caught was invisible to the scripted harnesses, because the
scripts used the words the code already knew:

* "whats MGTA 458 about" printed the entire plan of study. "whats" is not
  "what" and "about" was not a factual word, so a question about one course
  was not ruled factual and fell through to the intake.
* "ok moderate load please" was answered with "I want to make sure I answer
  the right question" -- to the question it had just asked. The load reader
  was a whole-message test.
* "i worked in marketing for 3 years and want to move more into data" was
  refused as the career **worked marketing 3 years and move more data**.

Add a persona whenever a real student says something the tests did not.

Second batch (eight more) found seven more: an abbreviated track ("11 mo"),
"hmm actually i think marketing is more my thing" refused as a career, a
role read as its own industry ("aiming for business analyst") dropping the
track and load, stated completions ignored, "why did you pick X" answered
with alternatives, "thanks!" met with the opening, and "how many units to
graduate" asking for the student's track in return.

Third batch (walk-through, swaps, quarter loads): a mid-walk-through swap that
could not be made was indistinguishable from one that had; "can i make fall
lighter" with a plan on file went to the classifier and asked which track;
the goal prompt still promised the removed web lookup; and the load lead
reported the target units while the quarter under it rendered the actual.
"""
import os, django, time, sys, textwrap
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()
from django.contrib.auth.models import User
from rsm_thrive.models import Conversation
from rsm_thrive.services.courses import orchestrator
from rsm_thrive.services.llm import get_llm

u, _ = User.objects.get_or_create(username="persona")
LLM = get_llm()
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 700

PEOPLE = {
 "Priya - knows exactly what she wants": [
   "hi! im on the 11 month track and i want to be a product analyst at a tech company",
   "whats MGTA 458 about",
   "ok moderate load please",
   "walk me through it",
 ],
 "Marcus - no idea, needs the menu": [
   "hey so honestly i have no clue what i want to do after this",
   "consulting sounds interesting",
   "whats the difference between the first two",
   "2",
   "17",
 ],
 "Yuki - career switcher from marketing": [
   "i worked in marketing for 3 years and want to move more into data",
   "marketing analyst",
   "11 month",
   "light, i have a part time job",
 ],
 "Dev - casual, typos, wants swaps": [
   "yo i wanna do data sceince stuff, 11 mo",
   "moderate",
   "can i swap 466 for something easier",
 ],
 "Aisha - already took courses, asks why": [
   "healthcare analytics, 17 month",
   "ive already done MGTA 464 and 402",
   "moderate",
   "why did you pick MGTA 463 for me",
 ],
 "Tom - pushes back and changes mind": [
   "pricing analyst 11 month",
   "moderate",
   "hmm actually i think marketing is more my thing",
 ],
 "Lin - asks about a quarter and prereqs": [
   "what should i take in winter",
   "product analyst",
   "does MGTA 466 need anything before it",
 ],
 "Sam - small talk and thanks": [
   "hey", "thanks!", "ok data scientist, 11 month, moderate", "thank you this is great",
 ],
 "Nadia - goes back and forth in the menu": [
   "not sure yet", "3", "actually show me the industries again", "1",
   "hmm what about healthcare instead", "1",
 ],
 "Raj - constraint-first, one sentence": [
   "i have a full time job so i need the lightest possible schedule, aiming for business analyst, 17 month",
 ],
 "Elena - asks about the degree itself": [
   "how many units do i need to graduate", "how many electives is that",
   "ok bi analyst 11 month moderate then",
 ],
 "Kenji - walks the plan and swaps": [
   "data scientist, 11 month, moderate", "walk me through it", "next",
   "swap MGTA 466 for MGTA 457", "next quarter", "what does spring look like", "finalise",
 ],
 "Fatima - mid-programme, switched track": [
   "i switched from 17 month to 11 month and im already in winter, what do i still need",
   "11 month", "data scientist",
 ],
 "Omar - pushes on a specific quarter": [
   "bi analyst 11 month moderate", "can i make fall lighter",
 ],
 "Grace - questions the recommendation": [
   "product analyst 11 month moderate", "is MGTA 458 really necessary", "ok keep it",
 ],
}

for name, turns in PEOPLE.items():
    c = Conversation.objects.create(user=u, destination="courses", title=name[:50])
    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)
    for t in turns:
        start = time.time()
        r = orchestrator.answer(LLM, c, t, [])
        print(f"\n  STUDENT: {t}")
        print(f"  [{r.model_note} · {time.time()-start:.1f}s]")
        body = r.body[:LIMIT] + (" …[trimmed]" if len(r.body) > LIMIT else "")
        for line in body.split("\n"):
            print("  | " + line)
        if r.quick_replies:
            print(f"  | BUTTONS: {[b['label'] for b in r.quick_replies][:6]}")
    c.delete()
