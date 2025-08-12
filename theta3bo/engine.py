import numpy as np
import matplotlib.pyplot as plt

class BOSession:
    """
    Minimal skeleton (no heavy GP deps). Intended as integration surface:
    - Backend selection (exact GP / SVGP / SAASBO / TPE) with logging
    - Acquisition plumbing (qNEI/qEI switch w/ hysteresis), async fantasization hooks
    - Replicate policy, failures, resume, feature importance
    Real model training is expected to be done with your GP/BO stack (e.g., BoTorch).
    """
    def __init__(self, spec, feature_names=None):
        self.spec = spec
        self.backend = None
        self.backend_reason = None
        self.feature_names = feature_names or []
        self.model = type("DummyModel", (), {"lengthscales": np.ones(len(self.feature_names))})()
        self.hysteresis_state = "qNEI"
        self.logs = {"decisions": []}

    # ---- backend selection ----
    def set_backend(self, n, p, config):
        k_cats = sum(1 for pdef in self.spec.get("parameters", []) if pdef.get("type")=="categorical")
        if n <= 5000 and p <= 100 and k_cats <= 20:
            self.backend, self.backend_reason = "exact_gp", "n<=5k & p<=100 & cats<=20"
        elif n > 5000:
            self.backend, self.backend_reason = "svgp_or_love", "n>5k"
        elif p > 100 or k_cats > 20:
            self.backend, self.backend_reason = "saasbo", "p>100 or many categoricals"
        else:
            self.backend, self.backend_reason = "tpe_fallback", "gp_unstable"
        self.logs["backend"] = self.backend
        self.logs["backend_reason"] = self.backend_reason
        return self.backend

    # ---- noise metric ----
    def estimate_noise_ratio(self, y, groups):
        import numpy as np
        grp = {}
        for val, gid in zip(y, groups):
            grp.setdefault(gid, []).append(val)
        stds = [np.std(vals, ddof=1) for vals in grp.values() if len(vals) >= 2]
        if not stds:
            return float("inf")
        med_std = float(np.median(stds))
        iqr = float(np.subtract(*np.percentile(y, [75, 25])))
        return med_std / max(iqr, 1e-12)

    # ---- acquisition plumbing (skeleton) ----
    def fit_surrogate(self, X, y, noise=None, priors=None):
        # Hook to your GP training; store learned lengthscales to self.model.lengthscales
        return self

    def fit_constraints(self, X, C):
        # Hook to your constraint models (real & GP-probit)
        return self

    def acquire(self, batch_size=8, constrained=True, trust_region=None):
        # Placeholder: return empty list (implementation delegated to your BO stack)
        return []

    def acquire_async(self, pending, fantasy_seed=None):
        # Record seed for reproducibility
        self.logs["fantasy_seed"] = fantasy_seed
        return []

    # ---- replicates / failures ----
    def replicate_policy(self, X, pred_var, noise_est, k_min=2, k_max=5):
        # Simple heuristic: higher of pred_var or noise_est gets more reps
        scores = np.asarray(pred_var) + np.asarray(noise_est)
        ranks = np.argsort(-scores)
        n = len(scores)
        reps = {int(i): int(np.clip(k_min + (k_max-k_min)*rank/(max(n-1,1)), k_min, k_max))
                for rank, i in enumerate(ranks)}
        return reps

    def handle_failed_trial(self, x, policy="infeasible_or_impute"):
        return {"x": x, "status": "failed", "policy": policy}

    # ---- updates ----
    def update(self, X_new, y_new, noise_new=None, C_new=None):
        return self

    # ---- diagnostics ----
    def feature_importance(self):
        ell = np.asarray(self.model.lengthscales, dtype=float)
        imp = 1.0 / np.clip(ell, 1e-12, None)**2
        return dict(zip(self.feature_names, imp.tolist()))

    def plot_feature_importance(self, top_k=None, ax=None):
        imp = self.feature_importance()
        items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
        if top_k is not None:
            items = items[:top_k]
        labels, vals = zip(*items) if items else ([], [])
        if ax is None:
            fig, ax = plt.subplots()
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("importance = 1/ℓ^2")
        ax.set_title("θ‴ v2.1 — Feature importance")
        return ax

    # ---- resume ----
    def resume(self, run_id, pending, fantasy_seed=None):
        self.logs["resumed_from"] = run_id
        self.logs["fantasy_seed"] = fantasy_seed
        return self
