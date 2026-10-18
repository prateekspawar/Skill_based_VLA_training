---
name: nvidia-driver-autoupdate
description: Unattended Ubuntu NVIDIA driver upgrades break Isaac Sim rendering (all-black cameras) until reboot
metadata:
  node_type: memory
  type: project
  originSessionId: d2915f31-430d-4df2-9436-ea337f837e9a
  modified: 2026-10-10T07:23:07.794Z
---

On 2026-10-10 Ubuntu auto-upgraded the NVIDIA driver (595.91 → 595.99) while the old kernel module stayed loaded.
Symptoms: `nvidia-smi` says "Driver/library version mismatch", torch CUDA still works, but Isaac Sim renders all-black
camera frames, so every find fails (detector scores 0.0, find pose errors ~180 mm).

**Why:** a whole evaluation can silently produce garbage results while all processes look healthy.

**How to apply:** if find/detection suddenly fails everywhere or frames are black, check `nvidia-smi` and
`/var/log/dpkg.log` for a driver upgrade first; the fix is a reboot (user's action). Consider asking the user to hold
the driver packages (`apt-mark hold`) during long runs. Related: [[plate-colours-change]].
