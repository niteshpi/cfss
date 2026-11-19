# cfss: Compiler-Feedback-Guided Schedule Search

Most GPU kernel autotuners choose a schedule (tile sizes, warps, pipeline stages) by compiling and timing many candidates and reading only the measured latency. The compiler, however, already knows a lot about each candidate before it ever runs: whether it fits in shared memory, how many registers it needs, whether it spills, and how its IR is structured.

This project tests whether that compile-time information can make schedule search cheaper and more reliable. The setting is a Triton GEMM kernel, studied with a pre-compile verifier and a learned cost model.

## Research question

If schedule search uses what the compiler reports (static resource use and IR structure), not just measured latency:

- does it find near-optimal schedules with fewer GPU runs?
- does a static pre-compile verifier cut wasted compiles without rejecting good schedules?

## Hypotheses

| ID | Hypothesis | Measured by |
| --- | --- | --- |
| H1 | Compiler-feedback features (registers, spills, shared memory, occupancy, TTGIR op counts) predict latency better than config + shape features alone | Spearman ρ, top-5 recall on held-out shapes |
| H2 | Cost-model-guided search reaches 95% of the best latency in fewer measured runs than random search | Evaluations-to-95%, regret vs. runs |
| H3 | A cost model trained on some shape families still ranks schedules well on unseen families | ρ on held-out families |
| H4 | A static verifier rejects invalid schedules before compilation with few false rejections | False-reject rate, % of compile/launch failures caught |

## Approach

1. **Schedule representation.** Each candidate is described as a `sched.gemm_schedule` op in a small MLIR dialect (built with xDSL). The schedule is kept separate from the Triton backend that executes it.
2. **Static verifier.** Named rules check each schedule before compiling: shared-memory capacity, threads per block, accumulator registers, minimum `tl.dot` tile, and oversized tiles. Every rule is scored for false accepts and false rejects against real compile/launch outcomes.
3. **Compile feedback.** For each compiled kernel the pipeline records register count, spills, shared memory and op counts from Triton's TTGIR (itself an MLIR dialect).
4. **Exhaustive measurement.** Every schedule is measured on every shape once: a correctness check against an FP32 reference, then timed runs with an L2 flush between them. This gives the true best schedule per shape.
5. **Cost models.** A random forest, an XGBoost regressor and an XGBoost ranker are trained on two feature sets: A (config + shape + analytic features) and B (A + compiler feedback).
6. **Replay search.** Random search, a hand-picked heuristic list and cost-model-guided search (static and with online refit) are compared by replaying the measured table under the same run budget, over multiple seeds.

## Benchmark

**Schedule space (1,440 configs):**

| Parameter | Values |
| --- | --- |
| BLOCK_M | 16, 32, 64, 128, 256 |
| BLOCK_N | 32, 64, 128, 256 |
| BLOCK_K | 32, 64, 128 |
| num_warps | 2, 4, 8 |
| num_stages | 2, 3, 4, 5 |
| GROUP_M | 1, 8 |

**Workloads (48 GEMM shapes, FP16 inputs, FP32 accumulation):**

| Family | Count | Split |
| --- | --- | --- |
| Square (256 to 8192) | 6 | train |
| LLM projection shapes | 12 | train |
| Random rectangular, multiples of 64 | 22 | 18 train, 4 validation |
| Non-aligned rectangular | 4 | test |
| Skinny decode (M = 16, 32) | 4 | test |

The test set holds out two whole shape families. No model or search setting is tuned on test shapes.

**Baselines:** cuBLAS (`torch.matmul`), the Triton tutorial's autotune list, and the best single fixed config.

## Experiments

| ID | Question |
| --- | --- |
| E1 | How much headroom does per-shape tuning give over fixed configs and the heuristic list? |
| E2 | Does guided search need fewer runs than random search? |
| E3 | Which cost model ranks schedules best? |
| E4 | Does the model transfer to unseen shape families? |
| E5 | Do compiler-feedback features help (set A vs set B)? |
| E6 | How accurate is the verifier, and how much compile time does it save? |
| E7 | (optional) Does the cost model transfer to a second GPU? |

Results are reported with medians and 95% bootstrap confidence intervals. Only measured speedups are credited, never predicted ones.

## Roadmap

| Phase | Weeks | Focus |
| --- | --- | --- |
| 1 | 1–4 | Kernel, verifier, full measurement sweep, cost models, replay search, experiments E1–E7, report |
| 2 | 5–8 | Agents in the loop: LLM agents that use the verifier and cost model as tools, and agents that propose new verifier rules and features, scored against the measured table |
| Later | — | Port the schedule dialect and verifier to spatial dataflow NPUs (e.g. AMD XDNA via MLIR-AIE) |

## Repository layout

| Path | Contents |
| --- | --- |
| `configs/` | Schedule space, shapes + splits, measurement protocol |
| `kernels/` | Triton GEMM kernel |
| `mlir/` | `sched` dialect, verifier rules, MLIR text emit/parse |
| `src/` | Measurement, sweep, features, cost models, search, statistics |
| `src/agent/` | Phase 2: replay oracle, agent tools, prompts, harness evolution |
| `slurm/` | Cluster batch scripts |
| `scripts/` | Environment check, GPU probe, plotting |
| `tests/` | pytest suite |
| `notebooks/` | Exploratory analysis |
| `results/` | Raw measurements (not in git), processed tables, figures |
| `literature/` | Paper reviews |
| `report/` | Weekly updates, methodology, paper, slides |
| `ENV.md` | Hardware, driver and software versions, dataset checksums |

## Status

Skeleton only: directories, empty source files and READMEs. Nothing is implemented yet.

## Setup

Requirements: Python ≥ 3.11 and an NVIDIA GPU.

1. PyTorch, installed from the PyTorch index that matches the local CUDA driver. Triton ships with the torch wheel and is not installed separately.
2. Remaining dependencies: `pip install -r requirements.txt`.
3. GPU, driver and package versions are recorded in `ENV.md`.

## Key references

CAKE (arXiv:2608.12629) · Ansor (OSDI 2020) · TenSet (NeurIPS 2021) · Triton (MAPL 2019) · MetaSchedule (NeurIPS 2022)
