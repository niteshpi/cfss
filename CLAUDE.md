# CLAUDE.md: handoff for the cfss project

Read this first. It summarizes the planning so far (Cowork, Nov 17 to 19, 2026; Claude Code, Nov 20 to 23, 2026) so any session can continue the work.

## Who and what

- **Researcher:** Nitesh Kumar Pipralia, MS ECE student at UW–Madison (2026–28). The advisor is Prof. Suman.
- **Background:** 6 years at AMD's AI group. Worked on the Windows AI compiler for Ryzen AI XDNA NPUs (MLIR), ONNX Runtime operator fusion and pattern matching, tiling and dataflow scheduling, L1/L2 buffer allocation, BF16/BFP16, and TVM on AMD AI Engines.
- **Starting paper:** CAKE: Compiler–Agent Co-Design for Frontier Kernel Evolution (Ye et al., NVIDIA/CMU, arXiv:2608.12629, read in full Nov 20). Agents write a typed, hardware-explicit IR. A verifier and a calibrated cost model give targeted feedback, and the harness itself evolves: failures become new verifier rules or features, but only after human review. Gaps it leaves open (confirmed from section 8 and Appendix C): NVIDIA only, the cost model is calibrated only for B200/H100, no ablations, verifier false positives/negatives acknowledged but never measured, and harness evolution still needs a human at merge gates.

## Chosen project (phase 1, 4 weeks)

**Title:** Compiler-Feedback-Guided Schedule Search: Verifying and Ranking Triton GEMM Schedules Before Execution.

A small, GPU-only version of CAKE's verifier + cost-model loop. A Triton GEMM autotuner checks configs with a static verifier before compiling, then ranks them with a cost model. The test is whether this beats random search and existing pruning on unseen shapes.

**Hypotheses (revised Nov 22 after the prior-work check):**

- **H1:** compiler-reported features (set B: registers, spills, shared memory, occupancy, TTGIR op counts) predict latency better than config features (set A) and statically estimated hardware features (set A+, following "Hardware-Aware Features for CUTLASS Kernel Selection", arXiv 2609.35587), and still win when compile time is counted.
- **H2:** guided search reaches 95% of the best latency in fewer GPU runs than random search.
- **H3:** the cost model generalizes to held-out shape families (Spearman ρ ≥ 0.7).
- **H4:** the static verifier has a false-reject rate under 2% and catches at least 80% of compile and launch failures. Measuring this is the strongest phase 1 claim: nobody reports it.

**Proposed add-on (pending Prof. Suman's agreement): certified cascaded autotuning.** Use Triton's compile stages as a sequence of filters: 0 config only (verifier), 1 TTGIR, 2 full compile (ptxas: registers, spills), 3 GPU run. Each cut is set by conformal prediction so it carries a guarantee ("with 95% confidence a config within 3% of best survives"). Questions: C1 cost vs quality (compile + run time), C2 coverage on new shape families and whether 3-shape recalibration repairs it, C3 which stage earns its cost. Caveats: compile stages see only the config (M, N, K are runtime args), so stages 1–2 are per-config filters whose savings are largest in cold starts; stopping Triton after TTGIR may need internals (check in week 1; fallback is a two-level cascade). New files: `src/conformal.py`, `src/cascade.py` (week 3).

**Benchmark (to be frozen at the end of week 2; grown Nov 23 for conformal calibration):**

- **Space:** BLOCK_M {16,32,64,128,256} × BLOCK_N {32,64,128,256} × BLOCK_K {32,64,128} × num_warps {2,4,8} × num_stages {2,3,4,5} × GROUP_M {1,8} = 1,440 configs.
- **Shapes (78):**
  - 6 square, 256 to 8192 (train)
  - 12 LLM projections: M ∈ {128,512,2048,8192} × (N,K) ∈ {(4096,4096), (14336,4096), (4096,14336)} (train)
  - 42 random aligned shapes (multiples of 64 in [256, 8192]): 18 train, 4 val, 20 calibration
  - 9 non-aligned shapes (exact sizes TBD): 3 recalibration, 6 test
  - 9 skinny decode shapes: M ∈ {8,16,32} × (N,K) ∈ {(4096,4096), (14336,4096), (4096,14336)}: 3 recalibration, 6 test (M values still open)
  - Totals: 36 train, 4 val, 20 calibration, 6 recalibration, 12 test. 112,320 measured points. Never train, tune or calibrate on test shapes. Recalibration shapes are only for the per-family recalibration experiment (C2).
- **Protocol:**
  1. Run the verifier and log its verdict, but compile anyway so its false rejects can be counted.
  2. Compile stage by stage (TTIR, TTGIR, full build); record each stage's time, save the TTGIR, and record n_regs, n_spills, shared memory.
  3. Check correctness against torch.matmul in FP32 (atol = rtol = 1e-2). Input dtype (FP16/BF16/FP32) is still open; with FP32 inputs tl.dot defaults to TF32.
  4. Time it: 10 warmup + 50 timed runs with CUDA events, a 256 MB L2 flush between runs, and store median/p10/p90.
  5. Append one JSON line per (shape, config) right away, so a preempted job can resume.
- **Approach:** measure everything once, then score every search strategy by replaying the table (regret, evaluations-to-95%, and total time including compile), with 10 seeds and bootstrap CIs.
- **Baselines:** cuBLAS (torch.matmul, same precision mode as the Triton kernels), the Triton tutorial's autotune list, the best fixed config, Triton's own `early_config_prune` + `perf_model` + `top_k` pruning, and a tritonBLAS-style analytic pick (arXiv 2512.04226).
- **Experiments:** E1 tuning headroom, E2 search comparison, E3 model comparison, E4 unseen shape families, E5 feature ablation (A vs A+ vs B), E6 verifier false positives and false negatives, E7 (optional) cross-GPU or NVIDIA→AMD transfer via TTGIR features, C1–C3 cascade (if adopted).
- **Report:** an 8-page workshop-style paper plus an appendix.

**Time budget:** 20 h/week, about 3.3 h/day over 6 days.

**Weekly gates:**

- Week 1: a correct, timed GEMM on Euler; partial-compile (TTGIR) check.
- Week 2: verifier done and the full dataset measured.
- Week 3: models, cascade and search done; methodology frozen on day 18.
- Week 4: experiments, report and talk.

## Phase 2 (weeks 5 to 8): agents in the loop

An LLM proposing configs is **not novel**. Prior work includes Helion's LLM autotuner (PyTorch, Jun 2026), TritonTune (NeurIPS 2026), Reasoning Compiler (NeurIPS 2025), Agentic Auto-Scheduling (PACT 2025), Autocomp and TritonForge.

- **Level 1 (baseline):** the agent proposes configs.
- **Level 2:** the agent calls verify/predict/compile_feedback as tools. Ablate what it sees: latency only, compiler feedback, or anonymized knob names (the AutoTuring check, arXiv 2609.19387, done there only on a simulator).
- **Level 3 (most novel):** the agent evolves the harness: verifier rules, cost-model features and (if the cascade is adopted) new stage filters, scored automatically against the measured table. A decision-tree rule miner is the required baseline.

Phase 2 starts with a 2-hour novelty check (also search "conformal autotuning" and "multi-fidelity compilation" on OpenReview, Scholar, MLSys/CGO/PACT 2026). Agents query a replay oracle with a run budget, so phase 2 needs no new GPU time. Token cost needs a budget agreed with the advisor.

**Long-term direction:** move the `sched` dialect and verifier to AMD XDNA NPUs via MLIR-AIE/IRON. Compare against KernelCraft and TauLang.

## Environment

- **Compute:** UW CAE **Euler** cluster (`ssh <netid>@euler.engr.wisc.edu`), Slurm `research` partition. Jobs can be preempted and are requeued, so code must checkpoint/resume. GPUs are selected via features (`--constraint` / `--prefer`). Exact GPU models and `--gres` spelling are **unknown**: discover them with `sinfo -o "%P %f"` and `sinfo -N -o "%N %P %f %G"`, then record them in ENV.md. Use one GPU model for all experiments.
- **Python:** `module load conda/miniforge && bootstrap-conda` (don't run `conda init`); Python 3.12; PyTorch 2.15 with cu132 wheels (`--index-url https://download.pytorch.org/whl/cu132`), falling back to cu128 if the driver is older than 580; the Triton version the torch wheel pins; then `pip install -r requirements.txt`. xDSL provides the `sched` dialect.
- **Slurm scripts:** set `TRITON_CACHE_DIR=/tmp/$USER/triton-$SLURM_JOB_ID`. Nsight Compute may be blocked; nothing depends on it.
- **Laptop:** an RTX 3060 (sm_86), for prototyping only.
- **Cloud sessions:** the `cfss` environment allows arxiv.org and export.arxiv.org. GitHub, pytorch.org and neurips.cc pages cannot be opened from it.

## Repo state (`cfss/`)

Skeleton exists: directories, empty code/config files and per-folder READMEs. Key files to implement, in order:

1. `configs/space.yaml`, `configs/shapes.yaml`, `configs/measure.yaml`
2. `kernels/matmul.py` (tutorial kernel, all knobs exposed, masked for non-aligned shapes; returns the compiled kernel with `n_regs`, `n_spills`, `metadata.shared`, `asm['ttgir']`; check these names against the installed Triton; also try stopping after TTGIR)
3. `src/space.py`, `src/shapes.py`, `src/measure.py`, `src/sweep.py` (resumable JSONL)
4. `mlir/verify.py`: R1 shared-memory capacity, R2 threads ≤ 1024, R3 accumulator registers, R4 tl.dot minimum tile ≥ 16, R5 tile oversize (warning)
5. `mlir/emit.py`: generic-form `"sched.gemm_schedule"` MLIR text
6. `mlir/sched_dialect.py` (xDSL, week 2)
7. `src/ttgir.py`, `src/features.py`, `src/models.py`, `src/conformal.py`, `src/cascade.py`, `src/search.py`, `src/stats.py` (week 3)
8. `src/agent/*` (phase 2)

## Working conventions

- Build incrementally and keep it small. The researcher wants to review each step, so don't generate the whole codebase at once.
- Keep `results/raw/` out of git. Never credit a predicted speedup, only measured ones.
- The proposal will keep changing. Update this file when decisions change.

## Where the full material lives

- **Project plan** (claude.ai doc): "CFSS: Project Plan and Two-Week Progress", https://claude.ai/artifact/M6zwLzLZBG5AeAFYiG27yJ. Problem, approach, benchmark, cascade add-on, roadmap, 26-item status tracker, phase 2, risks.
- **Paper review** (claude.ai doc): "CFSS Paper Review", https://claude.ai/artifact/ABdvnoUJmyXgYK36LpJmHU. About 30 papers and tools, closest prior work, open gaps and novel directions.
- **Original 4-week plan** (Cowork artifact): https://claude.ai/artifact/KdJ9RvVCDFDeqg6g7T3hvV.
- **Project notes** (claude.ai project "Compiler research"): `claude/cake-paper-notes-and-directions.md`.

## Open decisions

- Input dtype (FP16, BF16 or FP32).
- Skinny decode M values; exact non-aligned sizes.
- Cascade guarantee target (95% / within 3%) and whether the cold-start scenario convinces the advisor.
- Phase 2 token budget.
- "Prism" in the reading list could not be identified; needs a title or link.
