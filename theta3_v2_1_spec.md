# θ‴ v2.1 — Final Specification & Full Documentation

> **Context & deprecation.** This version replaces the earlier **multiplicative θ‴** design (norm × distance × multiple cosine terms × ω × post‑hoc normalization **N**) with a single **quadratic form** and its **bounded exponential**, with importance learned via ARD. The old multiplicative chain and normalization step are shown in the legacy “Ultimate Guide…” (see the main formula and normalization step on p. 4). That entire construct is now **deprecated**.

---

## 0) Design goals

- **Mathematical clarity:** one canonical form (quadratic or its exponential), no duplicated penalties.  
- **Numerical stability:** dimensionless inputs, smooth clamp, explicit jitter, PSD kernels, positivity by construction.  
- **Learnable importance:** ARD/length‑scales (not hand‑tuned weights).  
- **Reproducibility:** replicates, constraints, held‑out validation, deterministic mode, seeds.  
- **Portability:** same engine across **materials**, **superconducting devices**, **ads/SEO**, **trading**.  
- **Scalability:** exact GP → SVGP/LOVE → SAASBO/TPE fallback with logged auto‑selection.

---

## 1) Canonical mathematics

### 1.1 Encoded, dimensionless inputs
Let raw parameters \(v=[v_1,\dots,v_d]\). Encode to \(\phi(v)\in\mathbb{R}^p\):

- **Continuous:** pass through.  
- **Periodic angle \(\theta\):** \([\sin\theta,\cos\theta]\) (no wrap).  
- **Categorical:** one‑hot; for high cardinality → learned embedding (scaled to unit variance) (see §2.1).  
- **Context/dynamics:** time since start, batch/tool IDs, regime flags as numeric/one‑hot.  
- *(Optional for moiré systems):* include \(\lvert\theta-\theta^\*\rvert\) (normalized) if a “magic angle” \(\theta^\*\) is known; ARD will learn its relevance.

Choose a **feature scale** \(\sigma_i>0\) per encoded feature. If physics scale is unavailable, estimate \(\sigma_i\) from warm‑start **IQR/1.349**, cap to reasonable bounds, **re‑estimate once** after 50–80 points (cap change to **2×**) and then **freeze** (identifiability: \(\sigma\) vs. \(\ell\), see below).

**Smooth clamp (gradient‑safe):**
\[
\mathrm{softclip}(z)=c\tanh\!\big(z/c\big),\quad c\in[3,8]\ (default\ 5).
\]
Apply elementwise to normalized deltas
\[
\tilde\Delta_i(x,x') \leftarrow \mathrm{softclip}\!\Big(\frac{\phi_i(x)-\phi_i(x')}{\sigma_i}\Big).
\]
If early saturation is observed, grid‑search \(c\in[3,8]\) during warm‑start, then **freeze \(c\)**.

> **Identifiability note (\(\sigma\) vs. \(\ell\)).** Because \(\sigma\) (feature scale) and \(\ell\) (ARD length‑scale) both rescale axes, **freeze \(\sigma\)** after the single re‑estimation pass; interpretability/adaptivity lives in \(\ell\).

### 1.2 Distance & similarity (choose one default)
- **ARD distance (pairwise):**
\[
\rho^2(x,x')=\sum_{i=1}^{p}\frac{\tilde\Delta_i(x,x')^2}{\ell_i^2},\quad \ell_i>0.
\]
- **Bounded similarity (default for kernels & reporting):**
\[
S(x,x')=\exp\!\Big(-\tfrac{1}{2}\rho^2(x,x')\Big)\in(0,1].
\]
**No extra** \(1/\lVert\cdot\rVert^2\), cosine stacks, \(\omega\), or post‑hoc **N**. Angular/dynamic structure is handled by features; importance is learned via \(\ell_i\).

### 1.3 Kernel, mean & positivity
- **Default kernel (exact or approximate GP):**  
  \(k(x,x')=\sigma_f^2\,S(x,x')+\sigma_n^2\,\mathbf{1}[x=x']\).
- **Additive kernel (opt‑in):**  
  \(k=\sum_g k_g\) where each \(k_g\) is an ARD‑RBF on a feature group (continuous / periodic / categorical).  
  **PSD guarantee:** each \(k_g\) must be PSD; the sum preserves PSD.  
- **Prior mean with anchor + annealing:**  
  \(m(x)=\mu_0+\mu_1^{(t)}S(x,x^\*)\), with \(\mu_1^{(t)}=\mu_1 e^{-t/\tau}\) (or linear decay). Rule‑of‑thumb: \(\tau\approx\) **1–2 warm‑start batches**. **Runtime toggle:** set `anchor.enabled=false` mid‑run to drop anchor influence.
- **Positivity of parameters:** optimize \(\log \ell_i\), \(\log \sigma_f\), \(\log \sigma_n\) **or** use softplus transforms. **Record transformed values** in logs.

---

## 2) Implementation guide

### 2.1 Feature encoding & scales
- Periodic → \([\sin\theta,\cos\theta]\) (scales 1). Optional \(|\theta-\theta^\*|\).  
- Categorical → one‑hot; **high‑cardinality → learned embeddings** (8–32 dims), **L2** regularization; optional **dropout p=0.1**; always **rescale to unit variance** before inclusion in \(\phi(v)\).  
- Continuous → physics/ops scale (e.g., twist 0.05°, carrier density \(2\times10^{11}\,\mathrm{cm}^{-2}\), strain 0.2%).  
- **Auto‑estimate \(\sigma\)** from IQR/1.349 if unknown; **freeze** after one guarded update (≤2× change).

### 2.2 Target transform & heavy tails
- Default output scaling: **z‑score** (fit on training y; unscale predictions for reporting).  
- Heavy‑tail toggle: **Student‑t likelihood** (e.g., \(\nu=4\)) or **Huberized** transform (δ≈1.0) for fat‑tailed noise (trading/ads).

### 2.3 Learnable importance & noise modeling
- **ARD priors:** \(\log\ell_i \sim \mathcal{N}(0,1)\) (tune if domain hints exist).  
- \(\log\sigma_f,\log\sigma_n \sim \mathcal{N}(0,1)\) (or center \(\log\sigma_n\) on small fraction of target range).  
- **Heteroskedastic paths:**  
  (a) **FixedNoiseGP** with replicate‑estimated variances (default).  
  (b) **Latent noise GP** when variance clearly depends on \(x\) (detected via residual analysis/ANOVA).  
- **Replicates:** 2–5 technical reps per setting; **adaptive**: allocate more reps where predictive variance or estimated noise is high; less where stable.

### 2.4 Constraints & calibration
- **Real‑valued:** GP on \(g_j(x)\); PoF \(=\mathbb{P}\{g_j(x)\le 0\}\).  
- **Boolean:** GP‑**probit** (Bernoulli), with **class weighting** for imbalance; compute PoF analytically or via MC.  
- **Calibration sanity checks:** Report **Brier score** and **reliability curve**; if miscalibrated (e.g., ECE > 0.03), auto‑enable **Platt scaling** (or isotonic) and log.

### 2.5 Missing/failed observations
Mark failed trials with status; choose policy:
- **Infeasible** (count against PoF) **or**  
- **Impute with very high noise** variance.  
Expose `max_retries` and `cooldown` (batches) for scheduling.

### 2.6 Privacy & logging
- **Redaction hook:** hash sensitive IDs (ads/trading) with a salted hash; never write raw identifiers.  
- **Required log schema** (extend as needed):
```
{ raw_params, encoded_features, sigma, softclip_c,
  kernel_type, additive_groups,
  lengthscales, sigma_f, sigma_n,
  output_scaling, likelihood,
  acquisition_type, acquisition_value,
  jitter_epsilon, backend, backend_reason,
  seeds:{opt,warmstart,fantasy}, deterministic:{enabled,threads},
  noise_ratio, hysteresis_state,
  time, tool_id, batch,
  constraint_models, feasibility_probs,
  redaction, run_id, resumed_from }
```

---

## 3) BO engine

### 3.1 Surrogate & scalability
- **Default:** GP with ARD‑RBF \(S\), heteroskedastic likelihood, trained by MLL with priors above.  
- **Kernel jitter (always):** add nugget \(\epsilon=\max(10^{-12},\ 10^{-6}\cdot\text{mean}(\mathrm{diag}K))\). If Cholesky fails, multiply \(\epsilon\) by 10 up to \(10^{-3}\). **Log \(\epsilon\)** used.  
- **Backend auto‑select (log decision):**  
  - **Exact GP** if \(n\le 5{,}000\), \(p\le 100\), \(\#\text{categoricals}\le 20\).  
  - **SVGP** or **LOVE** if \(n>5{,}000\). SVGP default \(M=\min(1000,\ 0.2n)\), K‑means++ init on \(\phi(v)\). Log \(M\) and seed.  
  - **SAASBO** if \(p>100\) or many categoricals.  
  - **TPE/SMAC** fallback if GP training is unstable.
- **Deterministic mode (audits):** CPU‑only, fixed BLAS threads; log `{deterministic:true, threads:1}`.

### 3.2 Acquisition (single, constrained, multi‑objective)
- **Single‑objective:** Default **qNEI** under material noise; **auto‑switch** to **qEI** when noise is small.  
  Noise metric: \(\text{noise\_ratio}=\frac{\mathrm{median}_x\,\widehat{\mathrm{std}}[y\vert x]}{\mathrm{IQR}(y)}\).  
  Hysteresis: enter qEI if \< 0.04 for **two** consecutive iterations; revert to qNEI if \> 0.06. **Log decisions.**
- **Constrained:** multiply by PoF, \(\mathrm{qEI}_\mathcal F=\mathrm{qEI}\times\prod_j\mathrm{PoF}_j\).  
  **Safe mode:** SafeOpt / Constrained‑EI with feasibility prior and confidence sets; expose `safety_margin`.
- **Multi‑objective:** **qEHVI with box constraints**, “no‑preference” default. Reference point: warm‑start **10th percentile**; if ill‑posed (few/identical points), fall back to bounds or historical 5th percentiles. **Log method.**

**Acquisition optimizer defaults**  
20 multi‑starts (Sobol + incumbent); **L‑BFGS‑B** for continuous with box bounds.  
Mixed spaces: continuous relaxation + one‑hot rounding **or** a mixed‑integer local search (TR‑aware).  
Log **best‑start index** and **seed**.

### 3.3 Batching, trust‑region & async
- Batch size 4–16.  
- **Trust region:** init ~**20%** of each param range around incumbent; expand/shrink by **±20–50%** based on improvement & feasibility. **Reset** to 20% around best feasible if TR falls below **5%** without improvement across \(K\) proposals.  
- **Async with fantasization:** propose without waiting; **record fantasy seed(s)** for reproducibility.  
- **Server‑side dedup (race‑safe):** at submission, **de‑dup within \(\varepsilon\)** of existing/pending encoded points to prevent concurrent near‑duplicates.

### 3.4 Warm start & loop
- **Warm start:** 40–80 Sobol/LHS; include safe points; place replicates on a subset.  
- **Loop:** ingest → encode → normalize/softclip → transform y → fit surrogate → optimize acquisition (constraints/safety) → propose batch (or async) → run/record → append → repeat.  
- **Near‑duplicates:** if within an \(\varepsilon\)‑ball of queued/running encoded points, treat as a **replicate** (not a new arm).

### 3.5 Nonstationarity
- **Time feature + exponential decay** \(w_t=e^{-\lambda\Delta t}\) (default \(\lambda=0.01\)).  
- **Rolling‑window retrain** over last \(K\) batches.  
- Consider **change‑points** for regime flips (ads/markets).

---

## 4) Minimal APIs & schemas

### 4.1 Config (YAML/JSON)
(See `configs/example_config.json` in the bundle for a complete example.)

### 4.2 Pythonic interface (summary)
- `Theta3V2`: encode, normalize (softclip), \(\rho^2\), \(S\), target transforms, jitter utility, deterministic toggle, anchor toggle, config validation.  
- `BOSession`: backend selection & logging, surrogate/constraints fit (placeholders in skeleton), acquisition (sync/async), replicate policy, failure handling, update, feature importance (+ plot), deterministic resume.

---

## 5) Worked examples
Two notebooks are included in `/examples`: **Nickelate_Tc.ipynb** and **Ads_CPA.ipynb** showing config, warm‑start, constraints, async, trust‑region, seeds, and diagnostics (calibration & feature importance).

---

## 6) Test plan & acceptance
(Outlined in the README and CI stub; includes PSD & jitter checks, auto‑switch hysteresis, async reproducibility, calibration, resume idempotency, redaction.)

---

## 7) Migration notes (old θ‴ → v2.1)
Remove the multiplicative stack (cosines, \(\omega\), **N**) and replace with one quadratic + bounded exponential with ARD. Normalize in input space; treat dynamics as inputs; steer only via an annealed anchor mean; adopt BO + constraints + replicates + OOS validation. 

---

## 8) Quick reference (defaults)
- softclip c=5; warm‑start 64; batch 8.  
- Priors: \(\log\ell_i,\log\sigma_f,\log\sigma_n \sim \mathcal{N}(0,1)\).  
- Surrogate: GP‑ARD‑RBF + heteroskedastic; jitter \(10^{-6}\cdot \mathrm{mean}(\mathrm{diag}K)\) escalating to \(10^{-3}\) if needed.  
- Backend: auto (Exact GP ≤5k; SVGP/LOVE if n>5k; SAASBO if p>100; TPE fallback).  
- Acquisition: qNEI (auto→qEI with 4/6% hysteresis); × PoF; SafeOpt optional.  
- Optimizer: 20 multistarts; L‑BFGS‑B (continuous); relax+round or MILS (categoricals).  
- Nonstationarity: time feature + λ=0.01 and/or window.  
- Replicates: min 2, max 5, adaptive.  
- Determinism: CPU‑only, threads=1 for audits.  
- Privacy: redaction on; hash salt via env var.
