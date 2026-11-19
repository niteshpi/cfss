# mlir

- `sched_dialect.py`: the `sched.gemm_schedule` op in xDSL.
- `verify.py`: named static rules (shared memory, threads, registers, tile sizes), each scored for false accepts and false rejects.
- `emit.py`: write and read the schedule as MLIR text.
