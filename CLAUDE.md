# Notes for Claude

- Read `docs/NOTES.md` first (the technical notebook; `README.md` is the public project page): section 1 is the current
  architecture, section 7 the latest results, section 8 the open work.
  Update those sections whenever the architecture or results change.
- Two Python environments: simulation in `~/isaacsim` (scripts that import Isaac Lab), policies/perception in
  `~/lerobot-env`. They communicate through `so101_lang/serve/policy_server.py`.
- Evaluations must be seeded (`--seed`) and use ≥ 50 episodes; unseeded 20-episode runs are dominated by noise.
- `pkill -f <pattern>` can kill the calling shell when the pattern appears in its own command line; kill by PID.
- Large assets (`datasets/`, `checkpoints/`, `models/`, `outputs/`) are git-ignored.
- Launch long jobs detached (`setsid nohup <cmd> > log 2>&1 < /dev/null &`): background jobs started through the
  session are killed when the Claude session ends (this interrupted a 20k-step training once). Cron reminders are
  session-only too and must be recreated in a new session.
- Never edit a shell script while it is running (bash reads scripts incrementally; an edit corrupted a running eval
  driver once). Wait for it to finish, or copy it under a new name for the change.
