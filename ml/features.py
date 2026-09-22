"""
Feature engineering: turns one user profile (a dict from the form) into a numeric vector.

The same function is used for
  1. building the training dataset, and
  2. predicting for a real user in the web app,
so training and prediction can never drift apart.
"""
from data.labels import STREAMS, SKILLS, CATS, GOALS, AREAS

STREAM_IDS = [s["id"] for s in STREAMS]
SKILL_IDS = [s["id"] for s in SKILLS]
CAT_IDS = [c["id"] for c in CATS]
GOAL_IDS = [g["v"] for g in GOALS]
AREA_IDS = [a["v"] for a in AREAS]

FEATURE_NAMES = (
    ["edu"]
    + [f"stream_{s}" for s in STREAM_IDS]
    + [f"skill_{s}" for s in SKILL_IDS]
    + [f"int_{c}" for c in CAT_IDS]
    + [f"goal_{g}" for g in GOAL_IDS]
    + ["internet", "relocate", "budget"]
    + [f"area_{a}" for a in AREA_IDS]
    + ["n_skills", "n_interests"]
)


def vectorize(p):
    """profile dict -> list of floats (same order as FEATURE_NAMES)."""
    skills = set(p.get("skills", []))
    interests = set(p.get("interests", []))
    edu = p.get("edu")
    row = [float(edu) if edu is not None else 0.0]
    row += [1.0 if p.get("stream") == s else 0.0 for s in STREAM_IDS]
    row += [1.0 if s in skills else 0.0 for s in SKILL_IDS]
    row += [1.0 if c in interests else 0.0 for c in CAT_IDS]
    row += [1.0 if p.get("goal") == g else 0.0 for g in GOAL_IDS]
    row += [1.0 if p.get("internet", "yes") == "yes" else 0.0,
            1.0 if p.get("relocate", "yes") == "yes" else 0.0,
            float(p.get("budget", 1))]
    row += [1.0 if p.get("area", "village") == a else 0.0 for a in AREA_IDS]
    row += [float(len(skills)), float(len(interests))]
    return row
