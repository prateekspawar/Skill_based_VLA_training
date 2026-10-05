# Notes for Claude

- Read `README.md` first: section 1 is the current architecture, section 7 the latest results, section 8 the open work.
  Update those sections whenever the architecture or results change.
- Two Python environments: simulation in `~/isaacsim` (scripts that import Isaac Lab), policies/perception in
  `~/lerobot-env`. They communicate through `so101_lang/serve/policy_server.py`.
- Evaluations must be seeded (`--seed`) and use ≥ 50 episodes; unseeded 20-episode runs are dominated by noise.
- `pkill -f <pattern>` can kill the calling shell when the pattern appears in its own command line; kill by PID.
- Large assets (`datasets/`, `checkpoints/`, `models/`, `outputs/`) are git-ignored.
