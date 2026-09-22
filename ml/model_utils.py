"""Small helpers shared by training and the web app."""
import numpy as np


def predict_scores(model, X):
    """
    Returns an (n_samples, n_careers) array of predicted suitability scores on a 0-100 scale.
    (Models are trained on score/100, so the output is multiplied back by 100 here.)
    """
    if hasattr(model, "predict_scores"):
        return model.predict_scores(X)
    return np.clip(np.asarray(model.predict(X)) * 100.0, 0.0, 100.0)


class SoftVotingEnsemble:
    """Average of the predictions of several models."""

    def __init__(self, models):
        self.models = models

    def predict_scores(self, X):
        return np.mean([predict_scores(m, X) for m in self.models], axis=0)
