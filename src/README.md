# src

| File | Purpose |
| --- | --- |
| `space.py` | Enumerate configs from `configs/space.yaml` |
| `shapes.py` | Load shapes and splits |
| `measure.py` | Correctness check, timing, compile feedback for one (shape, config) |
| `sweep.py` | Measure everything, resumable after preemption |
| `ttgir.py` | Op counts from Triton's TTGIR |
| `features.py` | Feature sets A (config + analytic) and B (+ compiler feedback) |
| `models.py` | Random forest, XGBoost regressor and ranker |
| `search.py` | Replay search: random, heuristic, guided |
| `stats.py` | Bootstrap CIs, regret, evaluations-to-95% |
| `agent/` | Phase 2 (weeks 5–8) |
