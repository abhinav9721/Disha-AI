"""
Domain-knowledge scoring rule ("expert system").

It is used in two places:
  * generate_dataset.py  -> to create the training labels (which careers suit which profile)
  * recommender.py       -> ONLY to explain a prediction (why it fits, which skills are missing)

The match percentage shown to the user does NOT come from this function.
It comes from the trained ML model (see train_model.py).

Points: skills 45 + interest 22 + education 15 + stream 5 + goal 8 = 95,
then practical limits (internet, relocation, budget) subtract points.
"""


def expert_score(p, c):
    skills = set(p.get("skills", []))
    tot = got = 0
    matched, missing = [], []
    for k, w in c["skills"].items():
        tot += w
        if k in skills:
            got += w
            matched.append(k)
        elif w >= 3:
            missing.append(k)
    s = (got / tot if tot else 0) * 45

    reasons, notes = [], []
    if matched:
        reasons.append(("skills", matched[:3]))
    if c["cat"] in p.get("interests", []):
        s += 22
        reasons.append(("interest", None))

    gap = c["edu"] - (p.get("edu") or 0)
    if gap <= 0:                       # eligible; far over-qualified loses a few points
        s += 15 - min(6, -gap * 2)
        reasons.append(("edu", None))
    else:                              # needs more study
        s += max(0, 15 - gap * 6)
        notes.append("edu")

    if c["streams"] and p.get("stream") and p["stream"] in c["streams"]:
        s += 5
        reasons.append(("stream", None))
    if p.get("goal") and p["goal"] in c["goals"]:
        s += 8
        reasons.append(("goal", None))

    if c["reloc"] and p.get("relocate") == "no":
        s -= 8
        notes.append("reloc")
    if c["net"] and p.get("internet") == "no":
        s -= 8
        notes.append("net")
    if c["cost"] > p.get("budget", 1):
        s -= (c["cost"] - p.get("budget", 1)) * 4
        notes.append("cost")

    return dict(score=s, matched=matched, missing=missing, reasons=reasons, notes=notes, gap=gap)
