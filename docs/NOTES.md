# Technical notes — so101_lang (architecture, data, training, perception, evaluation, results log)

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
 user goal: "put the red cube on the green plate"
        │
        ▼
 ① Planner        so101_lang/planner.py (eval_policy.py --planner llm; default: oracle sequences). The LLM
                  (Ollama qwen3-vl:8b-instruct, no training) only states goals "cube X on Y", constrained to a
                  JSON schema of the scene's objects; code orders them (supports first), turns them into
                  pick / place / stack calls and validates them; find runs before every call. Re-planning
                  (--replan): after each sub-task open, go home, check every goal (perception or simulator
                  truth) and run the next unmet goal whose support is in place, ≤ 3 attempts per goal; the
                  LLM is not asked again.
        │
        ▼
 ② find(object)   per object:
                  a. find skill model moves the arm until it judges the view clear (own verdict, ≤ 6 s)
                  b. perception: 9-D pose of the object (position + rot6d, head frame)
                  c. fused (default): b again from 3 preset viewing poses (up_back, right_low, back_low) with all
                     trackers paused; the plausible detections that agree within 2 cm → median position
                  d. back home: a SAM2 tracker per object, anchored at the fused position
                     (object not visible from home → its position stays static)
        │ target pose (9-D) per skill
        ▼
 ③ pick  ④ place  skill models on one shared frozen SmolVLA backbone
                  inputs : left + right 512² images, point-map tokens (stereo depth), step text,
                           19-D state = TCP pose (10-D) + target pose (9-D)
                  output : 50-step chunk of 12-D actions = TCP pose + gripper opening + [done, success]
                  first 10 actions executed per query (~120 ms), then re-query; SAM2 updates the target every
                  3 steps, the pick target freezes within 3 cm of the gripper. A skill ends when done > 0.5 on
                  2 consecutive current-frame readings, or after 16 s. Grasp check after pick: gripper
                  opening < 0.05 (closed on nothing) or no pass verdict → open, go home, re-detect the cube, pick
                  again (up to 2 retries). Release gate: no gripper opening (> 0.05) while the TCP is more
                  than 4 cm sideways / 6.5 cm above the perceived plate target.
        │ 10-D head-frame action
        ▼
 ⑤ SO101Adapter   head → base calibration, TCP offset, gripper opening → joint angle (so101_lang/robots/so101.py)
        ▼
   SO-101 pose IK (Isaac Lab damped least squares, 5 arm joints) → joint targets → simulated robot
```

**Perception** (`so101_lang/perception/`, runs in the policy server):
1. OWLv2 proposes candidate boxes for the request ("red cube") and for the noun.
2. **Ranked detection** (`verify.py`) scores each box: detector score for request and noun, soft stereo size
   likelihood, and the learned colour posterior of the object's own pixels (`color_model.json`: how much more it
   looks like the requested colour than like any other colour name). The best box wins.
3. **Plausibility gate:** the winner must fit the request in size and colour; otherwise find looks again.
4. 3-D point from the mask + Fast-FoundationStereo depth (TensorRT FP16, ~9 ms, ~2 mm error), arm pixels removed
   with robot keypoints from FK (8 mm / 1° hand-eye error simulated). Yaw from the top face, folded by the
   object's symmetry (`language.YAW_SYMMETRY_DEG`: cube 90°).
5. **SAM2 tracker, state-based:** *locked* publishes the clear-view detection; switches to *following* only after
   two consistent trusted measurements > 1.5 cm away; locks again once still. Trusted = mask 0.7–1.4× its find
   size and < 15 % of its pixels on the robot.

Process layout (two Python environments that talk over a local socket):

| Process | Environment | Runs |
|---|---|---|
| Simulator + evaluation loop | `~/isaacsim` (Isaac Sim 6.1, Isaac Lab 3.0) | `scripts/eval_policy.py`, environment, adapter, IK |
| Policy server | `~/lerobot-env` (LeRobot 0.6.1, PyTorch 2.11 cu130, TensorRT 10.16) | `so101_lang/serve/policy_server.py`: skill models (shared 350 M backbone + 101 M per skill, 0.8 ms swap), OWLv2, SAM2.1-small, Fast-FoundationStereo |
| Planner LLM | Ollama in `~/.local/ollama` | `qwen3-vl:8b-instruct` (on the CPU next to the simulator: with it on the GPU the policy server runs out of memory) |

Current models (served by default): `checkpoints/pick9rc2_15000`, `place9rc2_15000`, `find9rc_15000`. Previous:
`pick9rc_15000` / `place9rc_15000` (needed `--place_release open`). Old white/pink-plate models for
comparison: `pick9fm_15000`, `place9m_15000`, `find9s_15000`. Earlier combined pick+place model (13-D state,
best push result): `d2_pointmap_20000`.

### Frames and conventions
- **Head frame:** origin midway between the two optical centres, +X forward (viewing direction), +Y left,
  +Z up. All VLA state, actions and target poses are in this frame. Quaternions are xyzw.
- **TCP:** grasp point between the jaw pads = gripper link + (0.009, 0, -0.090) m in the gripper-link frame.
- **10-D TCP state/action:** `[tcp.x, tcp.y, tcp.z, r11, r21, r31, r12, r22, r32, gripper.open]`
  (rotation as the first two matrix columns; gripper 0 = closed, 1 = open). Skill models add the 9-D target pose
  (find gets zeros) to the state and `[done, success]` to the action.
- **Relative actions:** the first 9 action dims are trained as `action_norm − state_norm` and the state is added
  back at inference (`relative_action_dims=9`); exactly invertible.
- **Virtual head** (`so101_lang/robots/virtual_head.py`): the models always see the training head pose. A real
  camera mounted elsewhere is given as `SO101_CAM_OFFSET="dx,dy,dz,roll,pitch,yaw"` (m, deg, Rz·Ry·Rx; the real
  camera's pose in the training head frame, i.e. hand-eye calibration result relative to the training pose).
  The policy server measures in the real camera frame (stereo, OWLv2/SAM2, with the real `head_in_base`) and
  converts the point map, found points and poses into the training frame; inputs given in the training frame
  (robot points, prior/base/true points) go the other way. `SO101_VIRTUAL_HEAD=0` turns this off (the real frame
  is passed straight to the models). No retraining; the image itself is not re-rendered.

### Skills (`so101_lang/language.py`)
| Skill | Parameter (allowed values) | Canonical step text |
|---|---|---|
| `pick` | `object`: red / green / blue / yellow cube | "pick up the {object}" |
| `place` | `target`: {colour} plate, colours from `PLATE_PALETTE` (white, pink, red, blue, green, yellow, orange, purple, brown) | "place it on the {target}" |
| `find` | `object`: any cube or plate | "find the {object}" |

Training phrasings and held-out phrasings per skill are defined there; `HELDOUT_PAIRS` = (yellow, pink), (green, white).

### Planner (`so101_lang/planner.py`, `scripts/eval_planner.py`, `tests/planner_cases.json`)
- Reply = goals `{cube_words, cube, on_words, on}`; `cube` / `on` are enums of the scene's objects plus "gripper"
  (keep holding) and "table" / "unspecified" (so the LLM can say a request is impossible; always refused).
- Code: ambiguity check (the user's own words, e.g. a bare "the plate" with two plates → refuse), ordering (held
  cube first, supports before what goes on them, cycles refused), conversion to calls, precondition check
  (`validate`). An invalid reply is sent back once with the error.
- Offline test set, 50 instructions (simple, reworded, multi-step, quantifiers, holding a cube, impossible):

  | Model / design | Correct | Time per plan (median / max, GPU) |
  |---|---|---|
  | `qwen3-vl:8b` (always reasons), skill calls | 47/50 | 12 s / 121 s |
  | `qwen3-vl:8b-instruct`, skill calls | 45/50 | 0.4 s / 2.6 s |
  | **`qwen3-vl:8b-instruct`, goals + code ordering + ambiguity check** | **48/50** | **0.4 s / 1.3 s** |

  Misses: "blue at the bottom, then green, then red on top" built upside down (unstable between runs); "each cube
  on the plate of its colour" sent red to the pink plate. The prompt was tuned on these 50 instructions, so the
  number is optimistic. **Held-out set** (`tests/planner_cases_heldout.json`, 50 new instructions written after
  the tuning, orange / brown plates, never used for tuning): **47/50** (simple 8/8, reworded 10/10, multi 6/8,
  quantifier 7/8, holding 4/4, impossible 12/12). Misses: "put green on the brown plate and then stack blue on
  it" refused (bare "it"; letting pronouns through made it stack on the plate instead, so refusal stays);
  "red at the bottom and blue above it" stacked upside down (same weakness as the tuning set's tower);
  "every cube on the plate of its own colour" (red / yellow plates) refused. On the CPU a plan takes 5–35 s
  (median 9 s).
- Re-planning does not use the skills' verdicts as proof (pick passes 7–10 misses per 50, place verdict 87 %):
  goals are checked in the scene. Perception check (`--check perception`): `locate9` of each goal's cube and
  support from home (preset views for objects not plausibly seen), cube within the plate radius and < 3 cm above
  the plate, or within 2 cm of the base cube and one cube height (± 1.2 / 1.5 cm) above it; an object never seen
  falls back on the goal's last verdict. Every check is scored against the simulator.
  - Plate positions come from the fused multi-view find (median 8 mm error): a single locate from home put two
    different plates at the same wrong spot, and a cube on a plate pulls the plate's detection away. Cubes are
    located live at each check (a cube on a plate is seen 1–2 cm low: plate pixels in its depth), so the window
    is cube centre within the plate radius and 2.5 cm below to 3.2 cm above the plate point.
  - Check accuracy on 8 debug episodes (two plates, objects seen): 88 % (single-view plates) → 98 % (fused
    plates); every error was a false "not met", which made the executor pick up a correctly placed cube and
    re-place it (once 11 cm off the plate).

### Pass/fail verdict
- `[done, success]` per action (`data/status.py`): running `[0,0]`, succeeded `[1,1]` (last 5 frames of a
  successful skill), failed `[1,0]` (from failure detection on; the robot then holds still).
- A failed verdict or a timeout stops the episode (hold still). The orchestrator does not act on verdicts yet.
- Timeouts are judged by the executor: a memoryless policy cannot tell "stuck for 3 s" from "just started".

---

## 2. Simulation (`so101_lang/envs/`)
- Task `SO101-PickPlace-Lang-IK-v0` (IK-controlled) and `SO101-PickPlace-Lang-Joint-v0` (joint-controlled, unused now).
- Scene: Isaac Lab's SO-101 (`so101_new_calib`) seated on a table, 4 coloured 3 cm cubes, 2 plates; random
  non-overlapping layout and (cube, plate) task per episode.
- **Plate colours:** two distinct colours from `PLATE_PALETTE` per episode (default; it includes the cube colours).
  `SO101_RANDOM_PLATE_COLORS=0` gives the fixed white/pink plates, `SO101_PLATE_COLORS=red,blue` a fixed override.
- **Stereo head:** two parallel 512×512 pinhole cameras, 6 cm baseline, 32 mm focal length (~36° FOV),
  centre at (−0.12, 0.20, 0.45) m looking at (0.21, −0.01, 0) — over the arm's "left shoulder".
  Calibration (intrinsics, head-in-base) is saved with every dataset as `meta/stereo_calibration.json`.
- Success: target cube resting on the target plate, gripper open, cube nearly still.

## 3. Data (`scripts/run_stage_randcolor.sh`: generation → training, resumable)
- **Scripted expert** (`expert/pick_place_sm.py`): top-down grasp state machine on the IK action
  (approach → descend → close → lift → carry → lower → release), about 97–98 % success.
- **DART noise:** a smooth Ornstein–Uhlenbeck offset (σ ~ U(0, 12 mm) per episode) on the *executed* command;
  recorded labels stay clean, so the data contains recoveries.
- Demonstrations (raw HDF5 in `datasets/raw/rc_*.h5`, all with random plate colours):
  - 1000 static + 1000 moving-target pick/place (`gen_demos.py`; cube slides during pick / plate during carry);
  - 400 failures (`gen_fail_demos.py`): pick miss / slip, place released off the plate / dropped;
  - 1000 find + 200 with the target absent (`gen_find_demos.py`).
- Processing (`so101_lang/data/`): `hdf5_to_lerobot` (point map + status labels) → merge (`pointmap_merge`) →
  `build_targets` (SAM2-style targets: detection noise, deadband, update rate, near-gripper freeze; 9-D pose,
  folded yaw, empirical yaw noise) → `skill_subset` → `mask_failure_motion` (motion before failure detection set
  to NaN, ignored by the loss).
- Datasets: `datasets/lerobot/s512_rc_all9` (all pick/place frames), `s512_pick9rc`, `s512_place9rc`,
  `s512_find9rc` (videos linked to `s512_rc_all_nopm`, not copied).

## 4. VLA training (`so101_lang/policies/smolvla_depth/`, `scripts/run_stage_skills.sh`)
- LeRobot plugin `smolvla_depth` = SmolVLA + native depth branch (head-frame point map from Fast-FoundationStereo
  → 64 zero-gated tokens) + relative actions. Train through `python -m so101_lang.train_cli` (registers the plugin).
- One model per skill, 15k steps each (~2.4 h on the RTX 5080). SmolVLA's vision-language backbone is frozen
  (`train_expert_only=True`), so the skills share it; about 100 M parameters train per skill.

## 5. Perception (`so101_lang/perception/`)
| Module | What it does |
|---|---|
| `stereo_depth.py` | Depth from the stereo pair. Default `fast_foundationstereo` (NVIDIA Fast-FoundationStereo ONNX, TensorRT FP16, 320 px: 2.2 mm median error vs ground truth, ~9 ms). Also SGBM. Point-map (`depth_to_pointmap`) helpers. |
| `locate.py` | `TargetLocator`: detection + depth → head-frame point; `locate9` → 9-D pose (yaw from the top face); `point_from_mask` with robot self-filter. |
| `verify.py` | Ranked detection (`detect_ranked`) and the plausibility gate. |
| `sam2_tracker.py` | SAM2.1-small streaming tracking seeded with the detection box; `Sam2MultiTracker` = one session per object, added at its own find; state-based locked / following logic (`SAM2_MODE=state`, default). |

## 6. Evaluation (`scripts/eval.sh` → `scripts/eval_policy.py` + `serve/policy_server.py`)
- `eval_policy.py` defaults are the current pipeline: find → pick → place, fused find, 9-D SAM2 targets,
  verdict-ended skills (current frame), 16 s per skill, 50 episodes.
- Seeded: layouts, pushes and the VLA's sampling noise (per episode and skill) are reproducible, so variants are
  paired. Without seeding, two identical 20-episode runs differed in 10/20 outcomes — use ≥ 50 episodes.
- `--push` slides the target cube 3–6 cm over 0.5 s during pick. Variants: `--find_views retry|all|fused`,
  `--pipeline direct` (no find), `--no-verdict` (skills end on simulator truth), `--target_dim 3` (13-D models).
- Simulator episode limit (`--episode_length_s`): 2 × step_timeout + 2 s, plus 120 s for the find pipeline.
  A time-out auto-resets the scene mid-episode; each run reports how many episodes it hit (must be 0).
- Logs per episode: success, find attempts and pose errors, target error at close, nearest cube, verdicts,
  tracker statuses; the summary breaks down pick misses (yaw / offset), place failure causes and verdict accuracy.

```bash
# server + 150-episode stationary and push runs with the current models (outputs/eval/)
setsid nohup scripts/eval.sh still push > logs/eval.log 2>&1 < /dev/null &
# or by hand: server (lerobot-env), then evaluation (isaacsim env)
~/lerobot-env/bin/python -m so101_lang.serve.policy_server
OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes 150 --seed 0
```

**Interactive demo** (`scripts/demo.sh`; needs `ollama serve`): `eval_policy.py --interactive --replan --check
perception --viz kit` with the arm's wrist mount removed (`SO101_NO_CAMERA_MOUNT=1`) and a visual model of the
stereo head on a post (`SO101_SHOW_HEAD=1`, `stereo_head_rig_cfgs()`; behind the optical centres and inside the
5 cm near clip, so the stereo images are unchanged). Each round: new random scene; instruction from the
"SO-101 demo" panel (`demo_ui.py`) or the terminal; the LLM plans in a thread while the app keeps rendering; the
panel shows the left camera with the perceived targets, the plan with each goal's status and attempt, and the
step running. The viewport camera is free (right mouse + WASD, Alt + left mouse). Videos: `outputs/demo/`.

---

## 7. Results so far (seeded, paired episodes; newest at the end)

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

**Tracker comparison, find pipeline (150 seeded episodes each; 2026-10-07).** Pick starts with the gripper closed
(as in training). Robot keypoints for the self-filter come from FK with an 8 mm / 1° hand-eye calibration error.
No simulator ground truth reaches the tracker.

| Tracker | Stationary pick / full | target err at close | Push pick / full | err right after push |
|---|---|---|---|---|
| displacement deadband 2.5 cm (old) | 73 / 48 % | 2 mm | 29 / 20 % | 21 mm |
| deadband 1 cm, freeze 3 cm | 72 / 43 % | 15 mm | **43 / 25 %** | 14 mm |
| **state-based** (locked / following on trusted measurements) | **79 / 52 %** | 2 mm | 35 / 23 % | 17 mm |
| state-based, looser trust while following (mask 0.4–2×, robot share <50 %) | 72 / 50 % | 2 mm | 33 / 23 % | 17 mm |

The state tracker publishes the clear-view detection while locked. It follows only after two consistent
trusted measurements more than 1.5 cm away, and locks again once still. A measurement is trusted when the mask
is 0.7–1.4× its find-view size and <15 % of its pixels lie on the robot. While the cube is still sliding it
rejects many measurements (mask size), so it follows pushes more slowly than the 1 cm deadband. Accepting those
measurements while following does not help: they are the gripper-occluded ones, which drag the target. Strict
trust is kept.
Stationary pick misses are now dominated by the cube-angle estimate: target yaw error median 18° in misses vs
2° in successful picks.

**Plate colours (2026-10-08).** The same first 25 stationary and 25 push episodes were rerun with the plates
recoloured by slot (white → red, pink → blue; `SO101_PLATE_COLORS=red,blue`). The models were not retrained.
- Place success after a good pick: **19/21 (90 %) vs 18/29 (62 %)** with white/pink. On the 11 stationary episodes
  picked in both runs, place succeeded 11/11 vs 7/11. The place model handles colour names it never saw, because
  the target comes from perception.
- Pick dropped (stationary 20 → 15, push 9 → 6): "blue cube" / "red cube" detections now land on the same-coloured
  plate, 7 cube targets 106–246 mm off at find. Plate colours must not repeat cube colours.

**Perception for plates that share colours with cubes (red/blue plates, same 25 stationary + 25 push scenes;
2026-10-08).**

| Variant | Stationary pick / full | Push pick / full | wrong cube at find | find failed |
|---|---|---|---|---|
| white/pink plates (reference) | 20 / 12 | 9 / 6 | 4 | 1 |
| red/blue, plain OWLv2 | 15 / 14 | 6 / 5 | 10 | 0 |
| ranked detection + plausibility gate, retry from home | 14 / 11 | 8 / 7 | 0 | 8 |
| **ranked + gate, retry from preset views (default)** | **15 / 13** | 6 / 5 | 2 | 2 |
| ranked + gate, always all views | 9 / 1 | 5 / 0 | 5 | 4 |

- **Ranked detection** (`perception/verify.py`) scores every candidate box: detector score for the request and for
  the noun, soft stereo size likelihood, and colour similarity of the object's own pixels. The best match wins.
  If even that match does not fit the request in size and colour, find retries from the next preset viewing pose
  (up_back, right_low, back_low). The gate is reliable: 0-1 wrong accepts and no wrong rejects per run.
- A rejected find is almost always the arm hiding the object, or arm pixels inside its box (rank_debug: 9 of 13
  occluded). Retrying from home repeats the occlusion; a different viewpoint fixes it.
- Always touring all views hurts: the arm sweeping the scene makes the cube and plate trackers drift (21/50
  episodes more than 3 cm off before pick). Oblique views also give plausible but mispositioned matches.
- Remaining red/blue pick gap: grasp misses with a correct target (8 vs 2). About half are cube-angle errors; the
  rest look like a visual shift, since pick was trained only with white/pink plates. Fix: vary plate colours in
  the training data.
- An attribute contrast via OWLv2 prompts ("a blue plate" vs "a blue cube") was tried and is off: OWLv2 is not
  reliable at fine attribute distinctions.

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

### Random plate colours: retrained skills, retry vs fused find (2026-10-09; 150 seeded episodes)
Each episode draws two plate colours from `language.PLATE_PALETTE` (`SO101_RANDOM_PLATE_COLORS=1`). The new
models (`pick9rc`, `place9rc`, `find9rc`, 15k steps each) were trained on 2,400 random-colour pick/place episodes
(incl. 400 failures, motion masked) + 1,200 find episodes, with empirical cube-yaw noise on the targets. The old
models (`pick9fm`, `place9m`, `find9s`) only saw white/pink plates. Same 150 scenes in every run
(`scripts/run_eval_randcolor2.sh`, outputs in `outputs/stage_rc_eval2/`).

| Models, find | Stationary pick / full | place after pick | Push pick / full (pushed: pick / full) | grasp misses | find failed |
|---|---|---|---|---|---|
| old, retry views | 61 % / 42 % | 69 % | — | 32 | 25 |
| new, retry views | 57 % / 41 % | 72 % | 28 % / 19 % (14/72, 10/72) | 35 | 28 |
| **new, fused views** | **73 % / 45 %** | 62 % | 31 % / 22 % (15/93, 13/93) | **14** | 25 |

Paired outcomes (same scenes): new vs old models, pick 19 wins / 24 losses, full 21 / 22, no difference.
Fused vs retry find, pick **27 / 3** (sign test p < 10⁻⁵), full 20 / 14 (not significant).

Findings:
1. **Retraining on random colours did not help.** On random colours the old models drop from 79 / 52 % (white/pink)
   to 61 / 42 %, and the retrained models score the same (57 / 41 %). The remaining gap is not the plates looking
   new to pick/place.
2. **Fused find is the clear pick gain** (+16 points): grasp misses fall from 35 to 14. With retry views the
   misses close a median 68 mm away from the cube (target error at close ~0, so the gripper is not following
   its target); with fused views 14 mm. Most of the gain is lost at place: 26 of 43 place failures are "never
   released" (retry: 13 of 24). Cause not yet analysed.
3. **Grasp misses are not perception yaw errors.** Target yaw error is 2–7° median for misses and ~2° for good
   picks. The jaw-cube angle at close is 21–28° for misses. The new models also close less aligned on successful
   picks (jaw-cube 14–17° vs 3° for the old models): the yaw noise added to training targets made them rely less
   on the target angle.
4. **Find fails in 1 of 6 episodes** with every variant (24–28 / 150, mostly plates), after three viewpoint
   retries. Find's own "not found" verdict is almost never right (0–3 of 44–65 reports per run).
5. **Pushes remain unsolved**: 15 % pick on pushed episodes. Wrong target at close (> 30 mm) in 28–44 episodes
   per push run, so the tracker loses the cube during the slide.

**Evaluation bug fixed (time-out reset).** `eval_policy.py` set the simulator episode limit to
2 × step_timeout + 2 = 18 s, written before the find stage existed. When find + pick + place ran longer, the
environment auto-reset mid-episode: new layout and (random colours) new plate colours, so the rest of the
episode ran on a different scene. Find-pipeline runs now use 138 s and log `simulator time-out resets
mid-episode`. With retry views it hit **1 / 50** episodes (2 %), so earlier find-pipeline results in this
README hold within that margin. Fused views tour six poses and hit it in every episode; the fused numbers from
before the fix (crashes, 147 mm targets) were this bug.

### Fixes from the failure analysis (2026-10-10; fused find, 150 seeded episodes, same scenes)
Each fix alone on stationary scenes, then all three together (now the defaults):

| Variant | Stationary pick / full | place after pick | find failed | Push pick / full (pushed: pick) |
|---|---|---|---|---|
| 9 Oct baseline: 8 s per skill, HSV colour rule | 73 % / 45 % | 62 % | 25 | 31 % / 22 % (15/93) |
| 16 s per skill | 75 % / 57 % | 76 % | 24 | — |
| 16 s + open the gripper after a place success verdict | 75 % / 64 % | 86 % | 24 | — |
| learned colour model (8 s) | 85 % / 55 % | 65 % | 4 | — |
| **all three (defaults)** | **86 % / 72 %** | **84 %** | 6 | 35 % / 33 % (20/109) |

Paired full-success wins / losses vs the previous row's base: 16 s 24 / 7, forced release 18 / 7, colour model
26 / 11 (pick 23 / 5). Seeded reruns still differ in ~8 % of pick outcomes (GPU non-determinism), so differences of
a few episodes are noise.

Findings:
1. **"Never released" was mostly timing and labels, not the place model.** Of 26 (fused, 8 s): 11 were a good pick
   whose verdict came too late (cube lifted within 4 s, but the "done" reading lagged > 3 s in 15 episodes), so
   place never ran; 5 were place running out of 8 s; 9 were place reporting success before opening.
2. **The place success label sat before the release.** Demos ended when the success term first fired, with the
   gripper ~27 % open; the last-5-frames success label covered closed-gripper frames, so the model says "success"
   just before releasing. Executor workaround: `--place_release open` (default until the `*9rc2` models). Data fix: `gen_demos
   --post_release_frames 5` records through the retreat; new demos `rc2_*` are being generated.
3. **Plate colour failures were the hand-set colour rules.** Brown failed 94 % of finds: the bright lighting renders
   brown as light tan, which the HSV rule scored 0.00; orange scored higher as yellow, pink as white. The learned
   model (`perception/color_model.json`, `scripts/fit_color_model.py`) scores a candidate by its posterior over all
   colour names (plus table / outlier classes), fitted on rendered training pixels. Brown finds 19 → 0 failures.
4. Remaining stationary failures (42): released off the plate 12, find failed 6, pick misses ~15, never released 4.
   Push: pick on pushed episodes 20/109 — the tracker / pick under motion is now the largest gap.

**Push study (2026-10-11; defaults, same 150 push scenes, cube slid 3–6 cm during pick).**

| Tracker | pushed episodes | picked |
|---|---|---|
| state-based (default) | 109 | 20 (18 %) |
| state-based, follows after 1 trusted measurement > 1 cm | 107 | 17 (16 %) |
| displacement deadband 1 cm | 109 | 19 (17 %) |

- The tracker is not the bottleneck. The gripper closes a median 0.7 s after the push ends; in 44 episodes the
  target had caught up before the close, yet only 13 were picked (gripper ~30 mm from the cube at close).
- **The pick model closes on schedule instead of re-aiming:** its moving-target demos only moved the cube during
  the approach (gripper > 6 cm away), never during the descent.
- Fix in the data: `gen_demos --late_pick_move` slides the cube during the descent (jaws ≥ 4 cm above it); the
  expert follows it (and lifts straight up to re-approach if it moves faster than the expert,
  `ExpertCfg.recover_dist`). 500 such demos (`rc2_latepush_500`) for the retrain.

### Retrain on post-release / late-push demos (`pick9rc2`, `place9rc2`; 2026-10-11; 150 seeded episodes, same scenes)
Data (`scripts/run_stage_rc2.sh`): `rc2_static_1000` + `rc2_moving_1000` + `rc2_latepush_500` (recorded through the
release and the expert's retreat; in the late-push set the cube slides during the pick descent) + `rc_fail_400`.
Pick targets with 3° Gaussian yaw noise (instead of the empirical noise of `*9rc`). The place success label covers
every frame after the gripper is fully open (`data/relabel_release.py`: the expert's retreat mostly ran into its 4 s
timeout, leaving ~90 identical hover frames with only the last 5 labelled). Find unchanged (`find9rc`).

| Models | Stationary pick / full | place after pick | place failures | place verdict acc. | jaw–cube yaw at close (picked) | Push pick / full | pushed picked |
|---|---|---|---|---|---|---|---|
| `*9rc` (defaults until now) | 86 % / 72 % | 84 % | 21 | 76 % | 16.6° | 35 % / 33 % | 20/109 |
| **`*9rc2`** | 83 % / **77 %** | **93 %** | **9** | **87 %** | **4.1°** | 31 % / 29 % | 23/119 |
| `*9rc2`, no forced release | 83 % / 77 % | 93 % | 9 | 87 % | — | — | — |

- **Place releases on its own now.** With and without the forced release the run is identical (0 forced
  releases needed); "never released" 4 → 1, place verdict 87 % accurate with no timeouts.
- **Grasp alignment fixed** by the smaller yaw noise (jaws 4° off the cube faces vs 17°). Pick is unchanged
  within noise (10 wins / 14 losses), full success 77 vs 72 % (28 / 20, not yet significant).
- **Late-push demos did not help pushes** (scenes pushed in both runs: 21 vs 20 picked of 106). 500 demos in which
  the expert simply re-aims during the descent were not enough for the pick model to re-aim.
- Remaining stationary failures (34): pick misses 15, find failed 8, place 9.

### Grasp check and pick retry in the executor (2026-10-12; `*9rc2`, 150 seeded episodes, same scenes)
Traces of pushed episodes (`--video_episodes`): the pick model descends and closes on its learned schedule even when
the target is already correct 30–57 mm away; the target froze at the first close, and the model then reported
success with an empty gripper. The gripper opening tells a miss apart reliably: 0.00–0.01 when closed on nothing
(104 of 107 failed picks), ~0.10 when holding a cube. The executor now checks it after pick and, on a miss (or no
pass verdict), opens, goes home, re-detects the cube (home view, else the preset views; fresh tracker) and picks
again, up to twice (`--pick_retries 2`, `--grasp_min_opening 0.05`, defaults).

| Executor | Stationary pick / full | Push pick / full | pushed picked | episodes retried (still / push) |
|---|---|---|---|---|
| no retry | 83 % / 77 % | 31 % / 29 % | 23/119 (19 %) | — |
| **grasp check + retry (default)** | **91 % / 85 %** | **87 % / 80 %** | **105/116 (91 %)** | 19 / 90 |

Paired vs no retry: stationary pick 12 wins / 0 losses, full 14 / 3; push pick 83 / 0, full 79 / 3.
- After a miss the cube is usually still and visible, so the second attempt succeeds (80 of 90 push episodes that
  retried were picked). Re-detection failed in 3 / 6 episodes.
- The model's lack of re-aiming during the descent remains (most pushed episodes need a retry); retries cost time,
  not success.
- Remaining stationary failures (23): place 10, find failed 6, pick 7.

### Release gate in the executor (2026-10-13; defaults, 150 seeded episodes, same scenes)
Place failures after a good pick (grasp check + retry, no gate; 20 in 300 episodes): early release mid-carry with
a correct plate target 6, released at the plate edge 5, plate target drift 4, cube dropped during the carry 5. The
gate (`--release_gate 0.04 0.065`, default) replaces any opening command above 0.05 with "closed" while the TCP is
more than 4 cm sideways or 6.5 cm above the perceived plate target (base frame); the expert releases ~3 cm above
the centre. A 0.5 threshold was too lenient: a partly opened gripper already drops the cube.

| Executor | Stationary pick / full | Push pick / full | place failures after pick (still / push) |
|---|---|---|---|
| grasp check + retry | 91 % / 85 % | 87 % / 80 % | 10 / 10 |
| **+ release gate (default)** | **91 % / 91 %** | **87 % / 87 %** | **1 / 1** |

Paired full success vs no gate: stationary 11 wins / 2 losses, push 14 / 4. The gate held the gripper closed in
12 / 9 episodes, all of which were then placed. It also removed the carry drops (the model loosens the grip
before it is over the plate) and most edge releases (it can only release close to the target).

### Granularity study: experiments 1-4 (2026-10-15 to 2026-10-18)
Monolithic vs task-level vs skill-level experts at equal data and compute. Paper draft in `paper/`, per-episode
results and paired sign tests in `paper/data/` (`summary.csv`). Drivers: `scripts/run_exp1_mono.sh`,
`run_exp2_stack.sh`, `run_exp2b.sh`, `run_exp3_compose.sh`, `run_exp4_task_experts.sh`.

Numbering follows the paper (order of presentation): 1 = monolith vs skills, 2 = task vs skill experts, 3 = adding
a skill, 4 = composed tasks. (Driver scripts keep their original names: run_exp4_task_experts.sh = Exp. 2,
run_exp2_stack.sh / run_exp2b.sh = Exp. 3, run_exp3_compose.sh = Exp. 4.)

| Exp. | Comparison | Result |
|---|---|---|
| 1 | one model (`mono9rc2_30000`, 30k steps) vs pick + place experts (2 x 15k), same frames | stationary full 93 vs 91 % (p = 0.22); push 81 vs 87 % (p = 0.12); no consistent difference |
| 2 | task experts (one per instruction, ~115 episodes, 4 x 7.5k steps) vs skill experts | seen 56-72 vs 90-98 %, unseen (nearest expert) 48-58 vs 94 %; all p <= 0.02 |
| 3 | adding `stack` (300 demos, 5k steps) | stack expert 53 % (100 demos 33 %), from place 59 %; place expert reused 0 %; monolith FT on stack only 0 % and forgets pick+place (0 %); monolith + replay (290 old episodes) 68 %, keeps 92 %; stacking release gate: no gain |
| 4 | unseen compositions, oracle sequences, skills vs monolith + replay | two plates 86 vs 88 %, plate-stack 42 vs 62 % (p = 0.08), tower 16 vs 18 %; 2nd step: 2nd plate 91/94 %, stack on a plated cube 45/65 %, stack on a 2-cube tower 26/26 % |

- Splitting by skill does not improve individual skills (Exp. 1); its value is that every expert learns from all
  the data (Exp. 2) and that skills are added in isolation (Exp. 3). The monolith with replay transfers more
  between skills (stacking) but needs the old data and re-validation on every change.
- Steps compose almost independently; no later step disturbed an earlier one. Composite success follows the weakest
  skill: stacking, which also does not extrapolate to target heights absent from its demos (cube on a plate, tower).
- Bugs found on the way (fixed): composed episodes restarted later sub-tasks from the previous pose instead of home
  (0 % composed), SAM 2 sessions accumulated GPU memory over composed episodes (server OOM), the LeRobot merge turns
  masked NaN actions into nulls (`data/restore_nan.py`).

### Wrist camera mount removed (2026-10-29; 37 of 50 seeded stationary episodes, same scenes as `eval_gate`)
The Isaac SO-101 USD's `Sensor` variant adds a wrist camera mount (mesh + collision) to the gripper; it is in every
training image (about 4 % of the head view). `SO101_NO_CAMERA_MOUNT=1` selects the `none` variant. Without the
mount: 34/37 full vs 33/37 with it (flips both ways), so the real robot does not need it. Run stopped at 37 episodes;
`outputs/eval_nomount/`. Print files for the real setup (SO-101 gripper, a stereo plate for two
Arducam IMX219 boards, cubes, plates) are in `hardware/prints/`.

### Camera pose offset with the virtual head (2026-10-30; 50 seeded stationary episodes, same scenes as `eval_gate`)
The rendered stereo head is moved away from the training pose (`scripts/run_camera_offset.sh`,
`outputs/camera_offset/`); the models are unchanged. Baseline on the same 50 scenes: 46/50.

| Offset | Virtual head | Full pick+place | Lost / gained vs baseline | Find error, cube / plate (median) |
|---|---|---|---|---|
| small: (-10, 15, 10) mm, (1, 2, -2) deg | on | 47/50 | 1 / 2 | 1 / 7 mm |
| large: (-25, 35, 20) mm = 47 mm, (2, 4, -4) deg = 6.05 deg | on | 47/50 | 2 / 3 | 1 / 6 mm |
| large | off | 3/50 | 43 / 0 (sign p < 1e-12) | 2 / 7 mm |

With the virtual head a camera mounted up to about 5 cm and 6 deg off the training pose costs nothing, provided the
hand-eye calibration measures where it is. Without it the policies fail (pick 41/50 but place 3/50): perception
still finds the objects, but the state, targets and point map arrive in a frame the models never saw. The real head
need not match the training pose exactly; it needs to be calibrated.

### LLM planner + re-planning in the simulator (2026-10-31; 50 seeded episodes per task, same scenes as Exp. 4)
`scripts/run_replan_eval.sh`, `outputs/replan/`. Skills `*9rc2`, `find9rc`, `stack9_place300_5000`; at most 3
attempts per goal. Baseline: the oracle sequence run once, no re-planning (Exp. 4 scenes, `outputs/exp3`).

| Task | Oracle, no re-planning | **LLM plan + re-planning, perception checks** | Oracle + re-planning, truth checks |
|---|---|---|---|
| two plates (`pairs`) | 43/50 | **44/50** (goals 94/100, 116 skill runs) | 47/50 (97/100, 110 runs) |
| plate, then stack (`plate_stack`) | 21/50 | **20/50** (65/100, 162 runs) | 31/50 (78/100, 146 runs) |
| tower | 8/50 | stopped (stacking set aside) | stopped |

- The LLM plan was right in 100/100 episodes (CPU, median 14 s, max 33 s per plan).
- Perception goal checks: 97 % right on `pairs` (332 checks, 24 with an object unseen), 94 % on `plate_stack`
  (424, 59 unseen); the errors are mostly false "not met" (8 and 19), which cost a re-run of a met goal.
- Re-planning recovers failed pick-and-place steps; it cannot fix the stack skill: in 22 of 50 `plate_stack`
  LLM episodes the stack goal failed all 3 attempts. Stack verdict accuracy 44-47 % (about half its passes are
  false), so checking goals in the scene rather than trusting verdicts is what makes the retries work at all.
- With truth checks `plate_stack` gains 10 episodes over the baseline, with perception checks none: the perception
  checks at the stack height (unseen cubes, false "not met") waste the retries the stack skill needs.
- The first `tower` run crashed: SAM2's video session kept every frame and its outputs on the GPU (~22 MB per
  frame), and a long tower sub-task with three cubes ran the server out of memory. The session now keeps them on
  the CPU (`sam2_tracker.py`; identical masks, GPU growth 2.2 GB → 0.19 GB per 100 frames). Tower runs were then
  stopped to concentrate on plain pick-and-place.

## 8. Not done yet / next steps
Best system (defaults: `*9rc2`, grasp check + retry, release gate): stationary 91 / 91 %, push 87 / 87 %.
Granularity study (experiments 1-4) complete; paper draft in `paper/` (authors and reference checks open).
1. **Stacking** at target heights outside the demos (cube on a plate, towers): add such demos, or a height-aware
   target representation; the stack skill bounds every composed task that uses it.
2. **LIBERO** (Object / Spatial, then Long and PRO): standard benchmark numbers for the skill-expert architecture
   (environment wrapper with depth, demo segmentation into skills, Franka adapter).
3. **Real robot:** a small SO-101 study (hand-eye calibration of the head, standalone IK, real stereo camera; refit
   the colour model and the gripper-opening thresholds on real data). Camera pose: virtual head (done, above);
   calibration targets in `hardware/calibration/`. Setup: Raspberry Pi 5 (`pi5-eth`, 10.0.0.4 on a direct cable)
   with two OV5647 cameras on its CSI ports (22-to-15 cables; the Arducam IMX219 CamArray kit never delivered
   image data) and the servo bus adapter on USB (`/dev/ttyACM0`); the PC runs perception, policies and IK and talks
   to small hosts on the Pi over wired Ethernet (ZeroMQ, as LeRobot's LeKiwi). Real images must be rectified to the
   training intrinsics (512 x 512, fx 781.87 px, 60 mm baseline).
   Status 2026-11-01:
   * stereo: `real/stereo_host.py` (left = cam1, 1296 x 972 at 30 fps, libcamera software sync, 0 ms L-R skew;
     640 x 480 measured ~30 ms exposure-to-PC, 33 Mbit/s);
   * exposure locked (`stereo_config.json` `exposure`; `"auto"` restores AE/AWB): both cameras at 33 ms, analogue
     gain 8, colour gains [1.157, 1.645], the auto values under the room lamp (~20 lux, 3600 K; AE was at its limit,
     so more light would allow a shorter exposure). The host's meta reports each camera's exposure, gain and colour
     gains; `controls {json}` changes them live. Re-lock when the lighting changes;
   * stereo calibration (`scripts/calibrate_stereo.py`, 72 ChArUco pairs, result `real/stereo_calib.json`): RMS
     0.51 px, rectified rows within 0.28 px (median), **baseline 62.98 mm** (plate design 60), relative rotation
     2.6°, fx ~1236 px = **55° horizontal FOV** (the seller's 72° is wrong; matches the v1 camera's 53.5°). The
     sim's 36° view is a ~800 px centre crop resized to 512. The 8-coefficient rational lens model folds back
     before the image corners; the standard 5-coefficient model fits equally well (0.003 px worse);
   * **the camera modules do not hold their left-right pose**: rows drifted 7.6 px, 9.8 px (after tightening the
     screws), then ~57 px (~2° tilt) within one session. The OV5647 sensor sits on a daughterboard held by foam
     tape; the modules are hot-glued now (gluing itself moved them ~0.2°; re-check rows after any handling). `calibrate_stereo.py solve
     --intrinsics <json>` re-solves only the left-right pose; the hand-eye solve re-measures it from its views;
   * hand-eye (`scripts/calibrate_handeye.py`, result `real/handeye.json`): the 90 mm board on the **moving jaw**
     (its opening is a sixth joint, so the steep head view can see the board face-on; on the fixed jaw it stayed
     55-65° oblique and mostly edge-on). The planner picks roll + opening to face the camera, rejects poses where
     the arm hides the board and keeps the gripper, board and paths clear of table, base and head. Current result
     (after hot-gluing the sensor modules; 21 views): **4.2 px RMS**; joint zero offsets shoulder_lift −0.1°,
     elbow_flex −3.6°, **wrist_flex −4.4°** (true = reading + offset; ±0.7-0.9° by bootstrap; the run before
     gluing gave +2.5 / −2.0 / −6.0, so treat them as ±2°). Head vs the training head (bootstrap ±3-5 mm,
     ±0.3-0.7°): `SO101_CAM_OFFSET=0.0351,-0.0229,-0.0024,-1.99,-1.44,1.76`. Repeat visit: 0.3 mm / 0.06° not
     explained by the joints (the arm settles ~1° differently by approach direction). Stereo left-right pose
     re-measured from the same views (live rows −1.4 px median; a 22 mm board refresh would tighten it). Isaac's
     SO-101 USD differs from the URDF (base body frame, no 2.8° wrist tilt): `real/kinematics.py` (numpy FK,
     `SIM_BASE_FROM_BASE`) matches Isaac to 0 mm;
   * end-to-end checks: `scripts/check_table_plane.py` (stereo depth into the base) puts the floor under the
     robot's table at −276 mm, matching the measured table height, flat to 2.1 mm but tilted 2.5° (lower toward
     −y; a level reading of the table top is still to do). `scripts/check_tip_overlay.py` on 6 poses outside the
     hand-eye fit: the FK skeleton and the sim jaw meshes land on the real arm and jaw (by eye; with the board on
     the jaw it reports corner errors); the zero offsets visibly improve the gripper position;
   * gripper: **0 % = closed** (LeRobot's range_min). At 5.5 % the fingertips are ~10 mm apart (= 6.1° at ~95 mm);
     closing below ~2 % the servo starts to lag and draw current (jaws meet). Real opening deg = % x 1.117 against
     the sim's 0-100° (1.745 rad), so sim opening 1.0 = 89.5 %. `arm_host.py` clamps the gripper to
     `gripper_range_pct` [0, 95] (was a hard-coded 5-95, which could not close the jaws);
   * 12 V supply sagged to 8.9 V on one large multi-joint move (safety stop); moves run at 10°/s now;
   * arm: LeRobot calibration done (`so101_follower`), all six joint directions match `so101_new_calib`, middle
     pose reads ~0° except wrist_flex ~+8° (offset in software if it matters);
   * powered hold and ±5° steps on every joint pass with `real/arm_host.py`. LeRobot's P = 16 leaves ~3° of
     gravity sag on shoulder_lift and elbow_flex; P = 32 on those two joints (in `real/arm_config.json`) brings it
     to ~1.3° without oscillation. LeRobot's SOFollower rewrites one P for all joints on connect, so a LeRobot-based
     client must re-apply the per-joint gains afterwards.
4. Re-aiming during the descent after late pushes (retries cover it today); find failures (blue cube not proposed).
5. LLM planner + re-planning: done for pick-and-place (results above). Interactive demo: `scripts/demo.sh`.
   Stacking (towers, cube on a plated cube) set aside for now.

## 9. File map
```
so101_lang/
  language.py                  object vocabulary, plate palette, skill registry, skills_schema(), templates
  planner.py                   planner LLM (goals -> ordered, validated skill calls), next_goal() re-planning
  sim_app.py                   Isaac Lab launch helpers (isaacsim env)
  demo_ui.py                   demo window in Kit: robot view with perceived targets, plan, current step
  envs/                        task config, MDP terms (plate colours), stereo head calibration (head.py)
  expert/pick_place_sm.py      scripted demonstrator (+ DART noise)
  robots/frames.py, so101.py   rigid-transform maths; SO101Adapter
  data/                        HDF5 → LeRobot converter, point-map merge, build_targets, skill_subset,
                               mask_failure_motion, status labels
  perception/                  stereo_depth, locate (OWLv2, locate9), verify (ranked detection), sam2_tracker
  policies/smolvla_depth/      LeRobot plugin (depth branch, relative actions)
  serve/policy_server.py       skill models + perception server; serve/skills.py shared-backbone swapping
  real/                        Pi-side hosts, copied to the Pi as ~/so101_real: arm_host.py (+ arm_cmd.py client,
                               arm_config.json: per-joint gains, caps, safety limits, applied at every start),
                               stereo_host.py (+ stereo_config.json); stereo_calib.json, handeye.json (the head's
                               calibrations); kinematics.py (numpy FK from the URDF + the sim asset's differences)
  train_cli.py                 lerobot-train with the plugin registered
scripts/
  run_stage_randcolor.sh       data generation → conversion → targets → per-skill datasets → training
  run_stage_skills.sh          train skill models (used by the stage driver)
  eval.sh                      server + seeded evaluation runs (still / push)
  eval_planner.py              planner LLM on tests/planner_cases.json (no simulator)
  run_replan_eval.sh           LLM planner + re-planning on the composed tasks
  demo.sh                      interactive demo: type an instruction, watch it in the Isaac Sim window
  run_eval_randcolor2.sh       the 2026-10-09 comparison (new vs old models, retry vs fused find)
  eval_policy.py               closed-loop evaluation;  gen_demos / gen_fail_demos / gen_find_demos.py: recorders
  calibrate_stereo.py          real head: ChArUco capture window (auto-saves diverse still pairs) + stereo solve
  calibrate_handeye.py         real head in the arm base: drives the arm (planned, path-checked poses) + solve
  *.py                         diagnostics and benchmarks (depth, detection, SAM2, latency, sensitivity)
train/                         training launchers
tests/                         frame-maths tests, planner instruction set
hardware/                      prints/ (all STLs), so101_gripper/ (sim gripper source files), stereo_head/
                               (parametric stereo plate), objects/ (cube + plate generator), calibration/
                               (ChArUco boards, hand-eye jaw clamp + board plate)
models/ (git-ignored)          Fast-FoundationStereo ONNX + TensorRT cache, smolvla_depth base config
datasets/, checkpoints/, outputs/ (git-ignored)
```

Licences of downloaded models: SmolVLA base, OWLv2, SAM2.1 — Apache-2.0; Fast-FoundationStereo — NVIDIA Open
Model Agreement (model card: research and evaluation use); Qwen3-VL — Apache-2.0.
