# ENV.md: compute environment

## Euler GPU inventory (from `sinfo`, Nov 24 2026)

Partitions with GPUs: `interactive`, `research` (all types below), `instruction` (RTX 4000 Ada only).

| GPU | Arch (sm) | Nodes | GPUs | Slurm gres | Feature tags |
| --- | --- | --- | --- | --- | --- |
| H100 | Hopper (sm_90) | euler29, euler30 | 8 (4 per node) | `gpu:h100` | `hopper,ent_gpu` |
| A100 | Ampere (sm_80) | euler17, euler19 | 8 (4 per node) | `gpu:a100` | `ampere,ent_gpu` |
| RTX A4500 | Ampere (sm_86) | euler05–08 | 32 (8 per node) | `gpu:rtxa4500` | `ampere,quadro_gpu` |
| RTX 4000 Ada | Ada (sm_89) | euler01–04, 09, 10 | 48 (8 per node) | `gpu:rtx4000ada` | `ada,quadro_gpu` |
| RTX 2080 Ti | Turing (sm_75) | euler16 | 2 | `gpu:rtx2080ti` | `turing` |

No AMD GPUs on Euler.

Pin the GPU model with the gres type, not `--constraint=ampere` (that tag covers both A100 and A4500):

```
srun -p research --gres=gpu:a100:1 -t 01:00:00 --pty bash
```

## Choice for experiments

- **Primary (all phase 1 measurements):** A100 (`gpu:a100`), pending a check that `ent_gpu` nodes are open to this account.
- **Fallback primary:** RTX A4500 (`gpu:rtxa4500`), same sm_86 architecture as the RTX 3060 laptop.
- **Cross-GPU (E7):** H100, plus A4500 or RTX 4000 Ada.
- **Development:** RTX 3060 laptop or A4500.
- **Not used:** RTX 2080 Ti (sm_75, outdated for current Triton).

## Still to record

- Access to `ent_gpu` nodes (try the `srun` above).
- Partition time limits: `sinfo -p research -o "%P %l"`.
- Driver version (`nvidia-smi` on an A100 node), CUDA, torch and Triton versions (`scripts/check_env.py`).
