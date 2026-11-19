# slurm

Cluster batch scripts. The GPU `--gres` / `--constraint` spelling is confirmed with `scripts/probe_gpu.sh` before jobs are submitted. Jobs on shared partitions can be preempted, so every job resumes from `results/raw/`.
