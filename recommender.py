"""
Loads the trained ML model and turns its output into the recommendation lists shown in the app.

  * match %        -> predicted by the ML model (suitability score 0-100 for each career)
  * eligibility    -> a hard rule: does the user's education reach the career's minimum?
  * "why it fits"  -> explanation text built from the expert rule (ml/expert.py)
"""
import os

import joblib
import numpy as np
import sklearn

from data.careers import CAREERS
from ml import train_model
from ml.expert import expert_score
from ml.features import vectorize
from ml.model_utils import predict_scores

REASON_KEYS = {"interest": "rInterest", "edu": "rEdu", "stream": "rStream", "goal": "rGoal"}


class Recommender:
    def __init__(self):
        self.bundle = self._load_or_train()
        self.model = self.bundle["model"]
        self.model_name = self.bundle["name"]

    @staticmethod
    def _load_or_train():
        path = train_model.MODEL_PATH
        career_ids = [c["id"] for c in CAREERS]
        if os.path.exists(path):
            try:
                bundle = joblib.load(path)
                if bundle.get("sklearn_version") == sklearn.__version__ and bundle.get("career_ids") == career_ids:
                    return bundle
                print("Saved model is from another scikit-learn version or career list. Training again ...")
            except Exception as e:                      # corrupted / incompatible file
                print(f"Could not load the saved model ({e}). Training again ...")
        else:
            print("No trained model found. Training one now (about 30 seconds, only the first time) ...")
        return train_model.train()

    def recommend(self, p, tr, skill_name, edu_label):
        """p = profile dict. tr / skill_name / edu_label = translation helpers for the current language."""
        x = np.array([vectorize(p)])
        scores = predict_scores(self.model, x)[0]

        items = []
        for c, score in zip(CAREERS, scores):
            if c.get("only") == "f" and p.get("gender") == "m":      # women-only roles
                continue
            ex = expert_score(p, c)
            reasons = []
            for kind, val in ex["reasons"]:
                if kind == "skills":
                    reasons.append(f"{tr('rSkills')}: {', '.join(skill_name(k) for k in val)}")
                else:
                    reasons.append(tr(REASON_KEYS[kind]))
            notes = []
            for n in ex["notes"]:
                if n == "edu":
                    notes.append(tr("nEdu").replace("{e}", edu_label(c["edu"])))
                else:
                    notes.append(tr("n" + n.capitalize()))              # nReloc / nNet / nCost
            items.append(dict(
                c=c, score=float(score), pct=int(min(97, max(5, round(score)))),
                gap=ex["gap"], missing=ex["missing"][:4], reasons=reasons, notes=notes,
            ))

        items.sort(key=lambda r: -r["score"])
        return dict(
            best=[r for r in items if r["gap"] <= 0][:6],       # can start now
            future=[r for r in items if r["gap"] > 0][:3],      # need more study first
        )
