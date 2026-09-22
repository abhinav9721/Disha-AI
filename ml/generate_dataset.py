"""
Creates a SYNTHETIC training dataset: many realistic rural-youth profiles, each with a suitability
score (0-100) for every career (score_<career id>).

Why synthetic?  There is no public dataset that links a person's education + skills + goals to a
suitable career in India.  So the targets come from the expert scoring rule in expert.py plus random
noise (people are not perfectly predictable).  The ML model then LEARNS this mapping from data and
can be re-trained on REAL data (survey / placement data) with the same columns - see README.md.

Run:  python -m ml.generate_dataset            (writes ml/dataset.csv)
"""
import os
import numpy as np
import pandas as pd

from data.careers import CAREERS
from data.labels import EDU
from ml.expert import expert_score
from ml.features import FEATURE_NAMES, SKILL_IDS, CAT_IDS, GOAL_IDS, STREAM_IDS, vectorize

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "dataset.csv")

EDU_VALUES = [e["v"] for e in EDU]
# education mix of rural youth: 8 levels (below 8th ... post-graduate)
EDU_P = [0.04, 0.08, 0.22, 0.08, 0.22, 0.06, 0.22, 0.08]
LABEL_NOISE = 4.0        # std-dev of the noise added to every score (points on the 0-100 scale)


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def sample_profile(rng):
    edu = float(rng.choice(EDU_VALUES, p=EDU_P))

    # a person "leans" towards 1-2 careers; their skills / interests follow that lean
    reachable = [c for c in CAREERS if c["edu"] <= edu + 0.5]
    arch = _pick(rng, reachable if rng.random() < 0.85 else CAREERS)
    arch2 = _pick(rng, CAREERS) if rng.random() < 0.3 else None

    skills = set()
    for a in [arch] + ([arch2] if arch2 else []):
        for k, w in a["skills"].items():
            if rng.random() < 0.30 + 0.11 * w:
                skills.add(k)
    for k in SKILL_IDS:
        if rng.random() < 0.07:
            skills.add(k)
    if not skills:
        skills.add(_pick(rng, SKILL_IDS))

    interests = set()
    if rng.random() < 0.8:
        interests.add(arch["cat"])
    if arch2 and rng.random() < 0.7:
        interests.add(arch2["cat"])
    for _ in range(int(rng.integers(0, 3))):
        interests.add(_pick(rng, CAT_IDS))
    if not interests:
        interests.add(_pick(rng, CAT_IDS))

    if edu >= 3:
        stream = _pick(rng, arch["streams"]) if arch["streams"] and rng.random() < 0.65 else _pick(rng, STREAM_IDS)
    else:
        stream = ""

    goal = _pick(rng, arch["goals"]) if rng.random() < 0.6 else _pick(rng, GOAL_IDS)

    return dict(
        edu=edu, stream=stream, skills=sorted(skills), interests=sorted(interests), goal=goal,
        internet="yes" if rng.random() < 0.65 else "no",
        relocate="yes" if rng.random() < 0.5 else "no",
        budget=int(rng.choice([0, 1, 2, 3], p=[0.30, 0.40, 0.20, 0.10])),
        area=str(rng.choice(["village", "smalltown", "city"], p=[0.6, 0.3, 0.1])),
    )


def make_dataset(n=24000, seed=42):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n):
        p = sample_profile(rng)
        scores = np.array([expert_score(p, c)["score"] for c in CAREERS]) + rng.normal(0, LABEL_NOISE, len(CAREERS))
        X.append(vectorize(p))
        Y.append(np.clip(scores, 0, 100).round(1))
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    labels = pd.DataFrame(Y, columns=[f"score_{c['id']}" for c in CAREERS])
    return pd.concat([df, labels], axis=1)


if __name__ == "__main__":
    data = make_dataset()
    data.to_csv(DATASET_PATH, index=False)
    print(f"saved {DATASET_PATH}  rows={len(data)}  columns={data.shape[1]}")
