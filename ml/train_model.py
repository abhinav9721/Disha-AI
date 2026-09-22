"""
Trains the career-suitability model.

Task   : multi-output regression.  Input = 65 profile features (education, stream, skills, interests,
         goal, internet, relocation, budget, area ...).  Output = 57 numbers, one per career:
         "how suitable is this career for this person" on a 0-100 scale.  The app shows it as the match %.
Models : Ridge regression (linear baseline), Random Forest, MLP neural network, and an ensemble
         (average of Random Forest + MLP).  The best one on the held-out test set is saved.
Metrics: MAE (average error in points), R2, hit@1 (is the model's top career a truly suitable one?)
         and precision@3 (how many of its top 3 are truly suitable?).  "Suitable" = true score >= 55.
         A "popularity" baseline (same answer for everybody) shows how much the ML really adds.

Run:  python -m ml.train_model                  (creates the synthetic dataset and trains)
      python -m ml.train_model --data my.csv    (train on your own real data, same columns)
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.careers import CAREERS
from ml.features import FEATURE_NAMES
from ml.generate_dataset import DATASET_PATH, make_dataset
from ml.model_utils import SoftVotingEnsemble, predict_scores

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.joblib")
METRICS_PATH = os.path.join(HERE, "metrics.json")
CAREER_IDS = [c["id"] for c in CAREERS]
TARGET_COLS = [f"score_{i}" for i in CAREER_IDS]
SUITABLE = 55            # a score of 55 or more counts as "suitable" when measuring hit@1 / precision@3


def evaluate(P, Y):
    """P = predicted scores, Y = true scores (both n x careers, 0-100)."""
    fit = Y >= SUITABLE
    top1 = np.argmax(P, axis=1)
    top3 = np.argsort(-P, axis=1)[:, :3]
    return dict(
        mae=float(np.abs(P - Y).mean()),
        r2=float(r2_score(Y, P)),
        hit_at_1=float(fit[np.arange(len(Y)), top1].mean()),
        precision_at_3=float(np.take_along_axis(fit, top3, axis=1).mean()),
    )


def train(n_samples=24000, data_csv=None, quiet=False):
    log = (lambda *a: None) if quiet else print
    t0 = time.time()

    if data_csv:
        df = pd.read_csv(data_csv)
        log(f"loaded {data_csv}: {len(df)} rows")
    else:
        log(f"generating {n_samples} synthetic profiles ...")
        df = make_dataset(n_samples)
        df.to_csv(DATASET_PATH, index=False)
        log(f"saved {DATASET_PATH}")

    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    Y = df[TARGET_COLS].to_numpy(dtype=float)                       # 0-100
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)
    log(f"train={len(Xtr)}  test={len(Xte)}  features={X.shape[1]}  careers={Y.shape[1]}")

    candidates = {
        "Ridge (linear)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=120, min_samples_leaf=2, n_jobs=-1, random_state=42),
        "MLP Neural Network": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=300, random_state=42)),
    }

    results, fitted = {}, {}
    baseline = np.tile(Ytr.mean(axis=0), (len(Xte), 1))
    results["Popularity baseline"] = evaluate(baseline, Yte)
    for name, model in candidates.items():
        s = time.time()
        model.fit(Xtr, Ytr / 100.0)                                 # models learn score / 100
        fitted[name] = model
        results[name] = evaluate(predict_scores(model, Xte), Yte)
        log(f"  trained {name:<20} in {time.time() - s:5.1f}s")

    ens = SoftVotingEnsemble([fitted["Random Forest"], fitted["MLP Neural Network"]])
    fitted["Ensemble (RF + MLP)"] = ens
    results["Ensemble (RF + MLP)"] = evaluate(ens.predict_scores(Xte), Yte)

    log("\n%-22s %8s %7s %8s %14s" % ("model", "MAE", "R2", "hit@1", "precision@3"))
    for name, m in results.items():
        log("%-22s %8.2f %7.3f %8.3f %14.3f" % (name, m["mae"], m["r2"], m["hit_at_1"], m["precision_at_3"]))
    log("(MAE is in points out of 100; the synthetic data itself has about 3 points of noise)")

    best = min(fitted, key=lambda n: (results[n]["mae"], -results[n]["precision_at_3"]))
    log(f"\nbest model: {best}")

    bundle = dict(model=fitted[best], name=best, career_ids=CAREER_IDS, feature_names=FEATURE_NAMES,
                  sklearn_version=sklearn.__version__, metrics=results)
    joblib.dump(bundle, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(dict(best=best, sklearn=sklearn.__version__, results=results), f, indent=2)
    log(f"saved {MODEL_PATH}  (total {time.time() - t0:.0f}s)")
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV with the same feature + score_* columns (default: make synthetic data)")
    ap.add_argument("--n", type=int, default=24000, help="number of synthetic profiles")
    a = ap.parse_args()
    train(n_samples=a.n, data_csv=a.data)
