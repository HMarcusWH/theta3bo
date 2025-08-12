# theta3bo v2.1.0

A minimal, portable implementation scaffold for the **θ‴ v2.1** metric and BO workflow.

## What’s inside
- `theta3_v2_1_spec.md` — the full spec and documentation.
- `theta3bo_schema_v2_1_0.json` — JSON Schema for config validation.
- `theta3bo/` — package skeleton (`Theta3V2`, `BOSession`, CLI).
- `configs/example_config.json` — a ready-to-edit config.
- `examples/` — minimal notebooks for Nickelate Tc and Ads CPA.

## Deprecation
This replaces the old **multiplicative θ‴** (norm × distance × multiple cosines × ω × post‑hoc N). See the legacy guide (p.4) for the deprecated chain; v2.1 uses one **quadratic form** and its **bounded exponential** with **ARD**.

## Quickstart
```bash
python -m theta3bo.cli validate --config configs/example_config.json --schema theta3bo_schema_v2_1_0.json
```

## Deterministic mode
CPU-only with fixed BLAS threads for audits; see spec §3.1.

## License
MIT.
