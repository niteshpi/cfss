# configs

- `space.yaml`: schedule parameters (BLOCK_M/N/K, num_warps, num_stages, GROUP_M), 1,440 raw configs.
- `shapes.yaml`: the 48 GEMM shapes with family and split (36 train / 4 val / 8 test).
- `measure.yaml`: timing protocol (warmup, reps, L2 flush, tolerances).

All three are frozen at the end of week 2 and not changed afterwards.
