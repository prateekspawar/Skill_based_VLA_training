# so101_lang — language-driven SO-101 manipulation (simulation-first, local)

Natural-language pick-and-place on a single SO-101 arm, built and evaluated in Isaac Sim 6.1 / Isaac Lab 3.0
on one RTX 5080 (16 GB). Everything runs locally.

Design goals (from the project owner):
- **Stereo "head" camera** (single mount, two cameras) placed like a human head relative to the arm.
- **Planner LLM** turns the user's goal into calls to a fixed **skill list**; it may only emit those skills.
- **VLA** executes one skill at a time and outputs a **robot-agnostic end-effector pose + gripper opening**.
- **Robot-specific adapter + IK** turns that into joint angles, so the VLA is reusable across arms.

---

## 1. Architecture (as used now)

```
 user goal: "put the red cube on the pink plate"
        │
        ▼
 ┌──────────────────────────────┐   NOT YET WIRED IN. Evaluation currently uses an *oracle planner*
 │ Planner LLM (Ollama,         │   (scripts/eval_policy.py) that emits the two skills for the episode.
 │ qwen3-vl:8b)                 │   Function list = language.skills_schema().
 │ → [pick(object="red cube"),  │
 │    place(target="pink plate")]│
 └──────────────┬───────────────┘
                │ per skill: step text  "pick up the red cube"  +  object name
                ▼
 ┌──────────────────────────────┐   so101_lang/perception/
 │ Perception                    │   • OWLv2 (open-vocab detector) finds the named object in the LEFT image
 │  stereo pair ─► FFS depth     │   • Fast-FoundationStereo (TensorRT FP16, 320 px, ~9 ms) → metric depth
 │  ─► target 3-D point          │   • box/mask pixels + depth → 3-D point in the HEAD frame
 │  (head frame)                 │   • targets: static (once at episode start)  or  SAM2 tracking
 └──────────────┬───────────────┘
                │ target point (x, y, z)
                ▼
 ┌──────────────────────────────┐   SmolVLA (450 M, LeRobot 0.6.1) via plugin "smolvla_depth"
 │ VLA executor                  │   inputs : left + right 512² images, step text,
 │                               │            13-D state = TCP pose (pos + rot6d) + gripper opening + target xyz
 │                               │   output : 50-step chunk of 10-D actions (TCP pose + gripper opening),
 │                               │            head frame, learned as offsets from the current state
 └──────────────┬───────────────┘   (first 10 actions executed per query, then re-query)
                │ 10-D head-frame action
                ▼
 ┌──────────────────────────────┐   so101_lang/robots/so101.py
 │ SO101Adapter (robot-specific) │   head→base calibration, TCP offset on the gripper link,
 │                               │   gripper opening → joint angle
 └──────────────┬───────────────┘
                ▼
   SO-101 pose IK (Isaac Lab damped least squares, 5 arm joints) → joint targets → simulated robot
```

Process layout (two Python environments that talk over a local socket):

| Process | Environment | Runs |
|---|---|---|
| Simulator + evaluation loop | `~/isaacsim` (Isaac Sim 6.1, Isaac Lab 3.0) | `scripts/eval_policy.py`, environment, adapter, IK |
| Policy server | `~/lerobot-env` (LeRobot 0.6.1, PyTorch 2.11 cu130, TensorRT 10.16) | `so101_lang/serve/policy_server.py`: VLA, OWLv2, SAM2, stereo depth |
| Planner LLM (not wired in yet) | Ollama in `~/.local/ollama` | `qwen3-vl:8b` |

### Frames and conventions
- **Head frame:** origin midway between the two optical centres, +X forward (viewing direction), +Y left,
  +Z up. All VLA state, actions and target points are in this frame. Quaternions are xyzw.
- **TCP:** grasp point between the jaw pads = gripper link + (0.009, 0, -0.090) m in the gripper-link frame.
- **10-D state/action:** `[tcp.x, tcp.y, tcp.z, r11, r21, r31, r12, r22, r32, gripper.open]`
  (rotation as the first two matrix columns; gripper 0 = closed, 1 = open). The 13-D state adds `target.x/y/z`.
- **Relative actions:** the first 9 action dims are trained as `action_norm − state_norm` and the state is added
  back at inference (`relative_action_dims=9`); exactly invertible.

### Skills (`so101_lang/language.py`)
| Skill | Parameter (allowed values) | Canonical step text |
|---|---|---|
| `pick` | `object`: red / green / blue / yellow cube | "pick up the {object}" |
| `place` | `target`: white / pink plate | "place it on the {target}" |
| `find` | `object`: any cube or plate | "find the {object}" |

### Skill models and the pass/fail verdict
- One model per skill (pick, place, find) on a **shared frozen SmolVLA backbone**; the server swaps only the
  ~100 M trainable tensors (`serve/skills.py`, `policy_server --skills pick=... place=... find=...`).
- State is 19-D: 10-D TCP + 9-D target pose (position + rot6d, head frame; find gets zeros). Yaw is not folded.
- **Action is 12-D:** 10-D TCP command + `[done, success]` (`data/status.py`): running `[0,0]`, succeeded `[1,1]`
  (last 5 frames of a successful skill), failed `[1,0]` (from failure detection on; the robot then holds still).
  The executor ends a skill when `done > 0.5` for 2 consecutive actions and passes the verdict to the
  orchestrator. A failed verdict means stop and hold still.
- Timeouts are judged by the executor (skill time limit / no progress). A memoryless policy cannot tell
  "stuck for 3 s" from "just started", so no stall demos are generated.
- Failure demos come from the expert with injected errors; the outcome is checked in the simulator:
  - pick: grasp miss (2–3.5 cm offset), slip (gripper opens while lifting);
  - place: released 9–12 cm off the plate, dropped during the carry (`scripts/gen_fail_demos.py`);
  - find: target absent, i.e. the object is moved out of view (`gen_find_demos.py --absent_prob`).
- Pipeline: `scripts/run_stage_verdict.sh`. Data:
  - 2000 success episodes plus 400 pick/place failures → `s512_all9s` → `s512_pick9s` / `s512_place9s`;
  - 1000 find successes plus 200 absent → `s512_find9s`.
- Eval: `eval_policy.py --verdict` scores every verdict against simulator truth.

Training phrasings and held-out phrasings per skill are defined there; `HELDOUT_PAIRS` = (yellow, pink), (green, white).

---

## 2. Simulation (`so101_lang/envs/`)
- Task `SO101-PickPlace-Lang-IK-v0` (IK-controlled) and `SO101-PickPlace-Lang-Joint-v0` (joint-controlled, unused now).
- Scene: Isaac Lab's SO-101 (`so101_new_calib`) seated on a table, 4 coloured 3 cm cubes, 2 plates; random
  non-overlapping layout and (cube, plate) task per episode.
- **Stereo head:** two parallel 512×512 pinhole cameras, 6 cm baseline, 32 mm focal length (~36° FOV),
  centre at (−0.12, 0.20, 0.45) m looking at (0.21, −0.01, 0) — over the arm's "left shoulder".
  Calibration (intrinsics, head-in-base) is saved with every dataset as `meta/stereo_calibration.json`.
- Success: target cube resting on the target plate, gripper open, cube nearly still.

## 3. Data (`scripts/gen_demos.py`, `so101_lang/expert/`, `so101_lang/data/`)
- **Scripted expert** (`expert/pick_place_sm.py`): top-down grasp state machine on the IK action
  (approach → descend → close → lift → carry → lower → release), about 97–98 % success.
- **DART noise:** a smooth Ornstein–Uhlenbeck offset (σ ~ U(0, 12 mm) per episode) on the *executed* command;
  recorded labels stay clean, so the data contains recoveries.
- Frames are labelled per skill (`pick` until lifted, then `place`) with that skill's step text.
- Raw recordings: `datasets/raw/s512_dart_1000.h5` (1000 episodes, 98 GB).
- LeRobot datasets: `datasets/lerobot/s512_dart_none` (10-D state) and `datasets/lerobot/s512_dart_target`
  (13-D state with target point, built by `data/add_target_point.py`, videos linked rather than copied).

## 4. VLA training (`so101_lang/policies/smolvla_depth/`, `train/`)
- LeRobot plugin `smolvla_depth` = SmolVLA + optional native depth branch (point-map tokens, zero-gated)
  + optional relative actions. Train through `python -m so101_lang.train_cli` (registers the plugin).
- **Current checkpoint (chosen for future use):** `checkpoints/d2_pointmap_20000/checkpoints/last/pretrained_model`
  - Same data and recipe as stage 1 plus the native depth branch (head-frame point map from Fast-FoundationStereo
    → 64 zero-gated tokens). Performs on par with the no-depth model (see the depth comparison) at ~10 ms more
    latency; kept so depth is available to the policy. Serve with `--depth fast_foundationstereo`.
- **Stage-1 checkpoint (no depth input):** `checkpoints/mix_target_20000/checkpoints/last/pretrained_model`
  - 20k steps from the SmolVLA base on `s512_mix_target`: 1000 static + 1000 moving-target episodes (cube slides
    during pick in ~35 %, plate slides during carry in ~37 %), 13-D state with SAM2-style targets
    (`data/build_targets.py`: detection noise, 2.5 cm deadband, SAM2 update rate, near-gripper freeze).
- Previous checkpoint: `checkpoints/s512_target_ft/checkpoints/last/pretrained_model`
  - Fine-tuned 2,500 steps on `s512_dart_target` from `checkpoints/s512_rel_none_16000` (16k steps, no target).
  - Depth branch off, relative actions on (9 dims), cameras left→camera1, right→camera2, camera3 empty.
- SmolVLA's vision-language backbone is frozen (`train_expert_only=True`); about 100 M parameters train.

## 5. Perception (`so101_lang/perception/`)
| Module | What it does |
|---|---|
| `stereo_depth.py` | Depth from the stereo pair. Default `fast_foundationstereo` (NVIDIA Fast-FoundationStereo ONNX, TensorRT FP16, 320 px: 2.2 mm median error vs ground truth, ~9 ms). Also SGBM (classical, noisy on textureless table). Colormap encoding and point-map (`depth_to_pointmap`) helpers. |
| `locate.py` | `TargetLocator`: OWLv2 detection ("a small red cube", "a round flat white plate") + depth in the box centre → head-frame 3-D point; `point_from_mask` with robot self-filter. |
| `sam2_tracker.py` | SAM2.1-small streaming video tracking, seeded with OWLv2 boxes. Robot self-filter (drops pixels within 3.5 cm of the arm's keypoints), mask-growth and jump checks, occlusion hold, and a 2.5 cm displacement deadband: published target = initial detection + tracked displacement. |

Targets are perceived **once at episode start** (before the arm occludes the scene), optionally followed by SAM2.
Pick targets freeze when the gripper is within 5 cm; place targets within 6 cm.

## 6. Evaluation (`scripts/eval_policy.py` + `serve/policy_server.py`)
- Seeded: layouts, pushes and the VLA's sampling noise (per episode and skill) are reproducible, so variants are
  paired. Without seeding, two identical 20-episode runs differed in 10/20 outcomes — use ≥ 50 episodes.
- `--target {none, oracle, perception, sam2, oracle_tracked}`, `--push` (slide the target cube 3–6 cm over 0.5 s
  during pick), `--heldout_pairs`, `--sam2_every`.
- Logs per episode: pick/place success, gripper-to-target distance and nearest cube at the first close, target
  error at close, push distance, SAM2 statuses.

```bash
# policy server (lerobot-env), then evaluation (isaacsim env)
~/lerobot-env/bin/python -m so101_lang.serve.policy_server --checkpoint checkpoints/d2_pointmap_20000/checkpoints/last/pretrained_model --depth fast_foundationstereo
OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes 50 --seed 0 --target sam2
```

---

## 7. Results so far (fair runs: seeded, 50 paired episodes)

| Setup | Pick | Full pick + place |
|---|---|---|
| Language only, no target point (any resolution / training length tried) | ~3 % | ~0 % |
| Target point, **static perception**, stationary objects | 38 % | 26 % |
| Target point, **SAM2 tracking**, stationary objects | 38 % | 24 % |
| Teleport push (unrealistic), any target source incl. oracle | 0 % on pushed episodes | — |
| Sliding push (3–6 cm over 0.5 s), **static perception** | 12 % | 6 % |
| Sliding push, **SAM2 tracking** | **28 %** | **18 %** |
| Sliding push, **oracle tracking** (true position refreshed every query) | 12 % | 10 % |
| **Stage 1 model** (static + moving demos, SAM2-style targets, 20k steps), SAM2, stationary | **64 %** | **44 %** |
| **Stage 1 model**, SAM2, sliding push | **50 %** | **34 %** (pushed episodes: 18/43 pick, 12/43 full) |

Depth comparison (same 2000 episodes, SAM2-style targets, 20k steps, SAM2 evaluation, 50 paired seeded episodes):

| Model | Stationary pick / full | Sliding push pick / full | Paired full-success wins vs baseline (stationary; push) | Latency |
|---|---|---|---|---|
| Baseline, no depth (`mix_target_20000`) | 64 % / 44 % | 50 % / 34 % | — | ~113 ms |
| D1, Fast-FoundationStereo depth image as camera3 (`d1_depthcam_20000`) | 62 % / 52 % | 42 % / 34 % | 12 vs 8; 2 vs 2 | ~126 ms |
| D2, native point-map tokens (`d2_pointmap_20000`) | 66 % / 46 % | 44 % / 32 % | 8 vs 7; 8 vs 9 | ~123 ms |

Neither depth variant is a significant improvement: all paired differences are within chance (e.g. 12 vs 8
discordant episodes for D1 stationary). With a perception-provided 3-D target point the VLA already gets the
geometry it needs; extra depth input adds ~10–13 ms per query without a measurable gain.

Key findings:
1. Without a target point, the VLA ignores the instruction's colour word (swapping the colour moves predictions
   ~3 mm vs ~26 mm for images) and closes 4–25 cm from the target. A perception-provided 3-D target fixed grounding.
2. Perception is accurate (cube ~2 mm, plate ~7–9 mm median), but about 1 in 10 episodes starts from a wrong
   OWLv2 detection (~200+ mm off); tracking cannot fix that.
3. SAM2 tracking costs nothing on stationary scenes once the self-filter, checks and deadband are in.
4. The VLA was trained only with targets that stay constant within a skill. Under sliding pushes SAM2 tracking
   (one deadbanded update after the cube moves; target 24 mm off at close vs 45 mm static) clearly beats static
   perception, while *continuously* refreshed perfect targets (oracle) do worse than SAM2 — the VLA copes with one
   step change but not with a target that keeps changing. Training with moving targets is the next fix.
5. Depth input to the VLA (colormap depth camera or native point-map tokens) gives no significant gain over the
   no-depth model once the target point is provided (see the depth comparison table). Depth stays in perception.

### Skill models with pass/fail verdict (2026-10-07; 50 seeded episodes, SAM2 targets, 9-D target pose)
Pick, place and find run as separate models on one shared frozen backbone (swap 0.8 ms). The final models are
`pick9fm_15000` and `place9m_15000`: cube yaw folded by the cube's 90° symmetry, and failure-demo motion masked.
Find is `find9s_15000`.

| Run | Pick | Full pick+place | Verdict accuracy (pick / place / find) |
|---|---|---|---|
| D2 combined model (reference) stationary / push | 66 / 44 % | 46 / 32 % | – |
| direct, skills end on simulator truth, stationary | 70 % | 38 % | – |
| direct, push | 46 % | 30 % | – |
| skills end on the model's verdict (current-frame read), stationary | 70 % | 42 % | 83 / 59 / – % |
| same, push | 40 % | 18 % | 80 / 43 / – % |
| **find → pick → place**, verdict-ended, stationary | **76 %** | **44 %** | 85 / 57 / 73 % |

**Find pipeline v2 (150 seeded episodes, ±4 pts; 2026-10-07).** Each object is tracked from its own find view, the
cube keeps tracking during find(plate) and the return home, and the same trackers run through pick and place.
find is retried only when both the find model and the detector name check say "not found". A fail verdict needs
two consecutive readings. Stationary: **pick 73 %, full 48 %** (place after a good pick 66 %). Push: pick 29 %,
full 20 %.
- Stationary pick failures (41): grasp missed with a good target 21, cube tracker drift before pick 6, wrong cube
  at find 5, target wrong during pick 3, never closed 5, find failed 1.
- Push pick failures (106): grasp missed 59, target wrong during pick 25, wrong cube at find 13.
- Stationary place failures (37): never released 14, wrong plate target 12, released off the plate 6, dropped 3.
- find acceptance: 300/314 accepted, 19–32 accepted with a pose >30 mm off, 12–16 correct finds rejected.
  The name check fails on plates (pale colours) about 90 times per run; plate colours will be changed.

Findings:
1. **Find works as intended.** Wrong-target pick failures fall from 13 to 5 per 50 episodes. Find pose error is
   1 mm (cube) / 7 mm (plate) median. Pick reaches 76 %, the best so far, even though the skills end on the model's
   own verdict.
2. **Symmetric-object yaw must match perception.** A pick model trained on unfolded cube yaw drops from 64 % to
   18 % when given the folded yaw that perception can measure. Targets are now folded by `language.YAW_SYMMETRY_DEG`
   (cube 90°, plate 360°; an asymmetric object keeps its full yaw).
3. **Failure demos must not teach their motion.** Unmasked, place succeeded after only 38 % of good picks
   (D2: 70 %): the model imitated releases beside the plate and drops mid-carry. With the motion before failure
   detection masked (`data/mask_failure_motion.py`, NaN entries ignored by the loss) it is 54 %. Place is still
   the weakest skill.
4. **Read the verdict from the current frame** (`--verdict_from first`). Later entries of a chunk are predictions
   up to 0.5 s ahead.
5. **Verdict errors are mostly false passes.**
   - Pick passes 7–10 misses per 50 episodes and almost never fails a real pick.
   - About half of place's false passes follow a false pick pass: place then runs with an empty gripper, a case
     never seen in training. The others are real place failures, or success reported before release.
   - Find wrongly reports "not found" for 19/100 objects it did find. All its training failures are
     "object absent", so it never learnt partial views.

## 8. Not done yet / next steps
- Wire in the LLM planner (Ollama qwen3-vl:8b, `skills_schema()` as the function list, validation, re-planning).
- Full training run with the target point (current model is a 2,500-step fine-tune).
- Better first detection (confidence / LLM verification).
- Verdict quality:
  - add place-with-empty-gripper failures and post-release hold frames (labelled success) to the place data;
  - add "partly visible / occluded but present" find successes;
  - more varied pick-failure demos (wrong-cube grasps, edge grasps) to cut false passes.
- Place is the weakest skill (54 % after a good pick vs D2 70 %): more place data, or initialise place from D2.
- Wire the verdicts into the planner: retry / re-find on failure.
- ~~Train the VLA to follow moving targets~~ done in stage 1 (combined pick + place model). Separate per-skill models
  were considered and dropped: the combined model is the one carried forward. (Tooling remains:
  `hdf5_to_lerobot --skill`, `build_targets --skill`, `policy_server --checkpoint_place`, `run_stage_per_skill.sh`.)
- ~~Rerun the depth comparison on the working recipe~~ done: no significant gain from either depth variant.
- Real robot: hand-eye calibration of the head (`head_in_base`), standalone IK, real stereo camera.

## 9. File map
```
so101_lang/
  language.py                  object vocabulary, skill registry, skills_schema(), templates
  sim_app.py                   Isaac Lab launch helpers (isaacsim env)
  envs/                        task config, MDP terms, stereo head calibration (head.py)
  expert/pick_place_sm.py      scripted demonstrator (+ DART noise)
  robots/frames.py, so101.py   rigid-transform maths; SO101Adapter
  data/                        HDF5 → LeRobot converter; add_target_point
  perception/                  stereo_depth, locate (OWLv2), sam2_tracker
  policies/smolvla_depth/      LeRobot plugin (depth branch, relative actions)
  serve/policy_server.py       VLA + perception server
  train_cli.py                 lerobot-train with the plugin registered
scripts/                       data generation, replay, evaluation, benchmarks, experiment drivers
train/                         training launchers
tests/                         frame-maths tests
models/ (git-ignored)          Fast-FoundationStereo ONNX + TensorRT cache, smolvla_depth base config
datasets/, checkpoints/, outputs/ (git-ignored)
```

Licences of downloaded models: SmolVLA base, OWLv2, SAM2.1 — Apache-2.0; Fast-FoundationStereo — NVIDIA Open
Model Agreement (model card: research and evaluation use); Qwen3-VL — Apache-2.0.
