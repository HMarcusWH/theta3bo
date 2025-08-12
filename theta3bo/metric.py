import numpy as np
import json
import os

class Theta3V2:
    """
    Canonical θ‴ v2.1 metric utilities:
    - Encode & normalize (softclip)
    - Pairwise distance ρ^2 and similarity S
    - Output transforms
    - Jitter utility & deterministic mode toggles
    - Simple config validation against a JSON Schema
    """
    def __init__(self, spec: dict):
        self.spec = spec
        self.softclip_c = spec.get("softclip_c", 5.0)
        self.parameters = spec.get("parameters", [])
        self.sigmas = self._collect_sigmas()
        self.feature_names = self._build_feature_names()
        self.anchor_enabled = spec.get("anchor", {}).get("enabled", True)

    def _collect_sigmas(self):
        # Per encoded feature; for one-hot/embeds default to 1
        sigmas = []
        for p in self.parameters:
            t = p["type"]
            if t == "periodic":
                s = p.get("sigma", 1.0)
                # sin, cos both scaled by 1 (or s if provided as meaningful scale)
                sigmas += [1.0, 1.0]
                if p.get("magic_angle") is not None:
                    sigmas += [float(s)]
            elif t == "categorical":
                # one-hot → σ=1 per level
                choices = p.get("choices", [])
                sigmas += [1.0]*max(1, len(choices))
            else:
                sigmas += [float(p.get("sigma", 1.0))]
        return np.asarray(sigmas, dtype=float)

    def _build_feature_names(self):
        names = []
        for p in self.parameters:
            t = p["type"]
            n = p["name"]
            if t == "periodic":
                names += [f"{n}_sin", f"{n}_cos"]
                if p.get("magic_angle") is not None:
                    names += [f"{n}_abs_delta_magic"]
            elif t == "categorical":
                for c in p.get("choices", ["<unk>"]):
                    names += [f"{n}::{c}"]
            else:
                names += [n]
        return names

    # ---- encoding ----
    def encode(self, v: dict) -> np.ndarray:
        enc = []
        for p in self.parameters:
            name, t = p["name"], p["type"]
            if t == "periodic":
                theta = float(v[name])
                enc += [np.sin(theta), np.cos(theta)]
                if p.get("magic_angle") is not None:
                    enc += [abs(theta - float(p["magic_angle"]))]
            elif t == "categorical":
                choices = p.get("choices", [])
                x = [0.0]*len(choices)
                if name in v and v[name] in choices:
                    x[choices.index(v[name])] = 1.0
                enc += x if x else [0.0]
            else:
                enc += [float(v[name])]
        return np.asarray(enc, dtype=float)

    def softclip(self, z, c=None):
        c = float(self.softclip_c if c is None else c)
        return c * np.tanh(z / c)

    def normalize(self, x, x_ref=None):
        # divide by sigmas, then softclip
        if x_ref is None:
            x_ref = np.zeros_like(x)
        z = (x - x_ref) / np.clip(self.sigmas, 1e-12, None)
        return self.softclip(z)

    # ---- metric ----
    def rho2(self, x, x_prime, ell):
        d = self.normalize(x, x_prime)
        li2 = np.clip(np.asarray(ell, dtype=float), 1e-12, None)**2
        return float(np.sum((d**2) / li2))

    def S(self, x, x_prime, ell):
        return float(np.exp(-0.5 * self.rho2(x, x_prime, ell)))

    # ---- y transforms ----
    def transform_y(self, y, mode="zscore"):
        y = np.asarray(y, dtype=float)
        if mode == "zscore":
            mu, sd = float(np.mean(y)), float(np.std(y) + 1e-12)
            return (y - mu)/sd, {"mode": mode, "mu": mu, "sd": sd}
        elif mode == "boxcox":
            from scipy.stats import boxcox
            y_shift = y - np.min(y) + 1e-9
            bc, lam = boxcox(y_shift)
            return bc, {"mode": mode, "lam": float(lam), "shift": float(np.min(y) - 1e-9)}
        elif mode == "huber":
            delta = 1.0
            med = np.median(y); mad = np.median(np.abs(y - med)) + 1e-12
            z = (y - med)/(1.4826*mad)
            hub = np.where(np.abs(z) <= delta, z, delta*np.sign(z))
            return hub, {"mode":"huber","median":float(med),"mad":float(mad),"delta":delta}
        else:
            return y, {"mode":"identity"}

    def inverse_transform_y(self, yhat, state=None):
        if not state or state.get("mode") in (None, "identity"):
            return yhat
        mode = state["mode"]
        if mode == "zscore":
            return yhat * state["sd"] + state["mu"]
        elif mode == "boxcox":
            lam = state["lam"]
            shift = state["shift"]
            if abs(lam) < 1e-6:
                return np.exp(yhat) + shift
            return np.maximum(0, (lam*yhat + 1)**(1/lam)) + shift
        elif mode == "huber":
            # Huber is not strictly invertible; return as-is
            return yhat
        return yhat

    # ---- misc ----
    def add_kernel_jitter(self, K, base=1e-6, min_jit=1e-12, max_jit=1e-3, growth=10.0):
        eps = max(min_jit, base * float(np.mean(np.diag(K)) + 1e-12))
        while True:
            try:
                np.linalg.cholesky(K + np.eye(K.shape[0])*eps)
                break
            except np.linalg.LinAlgError:
                eps *= growth
                if eps > max_jit:
                    raise
        return eps

    def set_deterministic(self, enabled=True, cpu_only=True, blas_threads=1):
        if not enabled:
            return
        os.environ["OMP_NUM_THREADS"] = str(blas_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
        os.environ["MKL_NUM_THREADS"] = str(blas_threads)

    def set_anchor_enabled(self, enabled: bool):
        self.anchor_enabled = bool(enabled)

    def validate_config(self, schema: dict):
        # Minimal checker: ensure required fields exist; print diff-like hints
        missing = []
        for r in schema.get("required", []):
            if r not in self.spec:
                missing.append(r)
        if missing:
            raise ValueError(f"Config missing required fields: {missing}")
        return True
