"""Paper figures (PDF for LaTeX, PNG for the preview page). Run in ~/lerobot-env from the repository root:

    ~/lerobot-env/bin/python paper/figures/make_figures.py

fig_scene        frames of a successful evaluation episode (left head camera)
fig_architecture system block diagram
fig_organisation monolithic vs task-level vs skill-level experts
fig_results      experiment 4 (task vs skill experts) and experiment 2 (adding a skill)
fig_composed     experiment 3 (composed multi-step tasks), overall and per step
"""

from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42})
# Validated categorical pair (dataviz reference palette, slots 1-2): blue = skill-level / modular, orange = comparison.
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8c8b86"
INK, MUTED, RULE = "#1d2124", "#5d666c", "#c9ccc8"
FILL = {"in": "#eef3f8", "perc": "#e8f4ef", "skill": "#e6eefa", "exec": "#fbefe9", "robot": "#f1f1ef"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- scene
def fig_scene():
    v = iio.imread("outputs/eval_gate/still.mp4")
    frames = [(0, "(a) scene"), (61, "(b) find: preset view"), (401, "(c) pick"), (463, "(d) placed")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95))
    for ax, (i, label) in zip(axes, frames):
        ax.imshow(v[i])
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(RULE)
        ax.set_title(label, fontsize=7.5, color=INK, pad=3)
    fig.subplots_adjust(wspace=0.04)
    save(fig, "fig_scene")


# ---------------------------------------------------------------- block-diagram helpers
def box(ax, x, y, w, h, title, lines=(), fill="#ffffff", edge=INK, tsize=7.5, lsize=6.4, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.08", fc=fill, ec=edge, lw=lw))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=tsize, fontweight="bold", color=INK)
    for k, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.40 - 0.22 * k, ln, ha="center", va="top", fontsize=lsize, color=MUTED)


def arrow(ax, p, q, text=None, color=INK, rad=0.0, tpos=0.5, toff=(0, 0.09), style="-|>", lw=0.9, tsize=6.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if text:
        x = p[0] + (q[0] - p[0]) * tpos + toff[0]
        y = p[1] + (q[1] - p[1]) * tpos + toff[1]
        ax.text(x, y, text, ha="center", va="bottom", fontsize=tsize, color=color, style="italic")


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim), ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- architecture
def fig_architecture():
    fig, ax = canvas(7.0, 3.5, (0, 14), (0, 7))
    L = 5.7  # line font size
    # column A: instruction -> skill sequence
    box(ax, 0.1, 5.3, 2.9, 1.5, "Instruction", ['"put the red cube', 'on the green plate"'], FILL["in"], lsize=L)
    box(ax, 0.1, 1.9, 2.9, 2.95, "Skill sequence", ["planner (oracle here)", "", "find(red cube)", "find(green plate)",
                                                     "pick(red cube)", "place(green plate)"], FILL["in"], lsize=L)
    arrow(ax, (1.55, 5.3), (1.55, 4.85))
    # column B: perception (top), skill experts (bottom)
    box(ax, 3.45, 4.45, 4.9, 2.35, "Perception", ["stereo depth \u2192 point map", "OWLv2 ranked detection:",
                                                   "request, noun, size, learned colour", "find: fuse 4 viewpoints",
                                                   "SAM 2 tracker: locked / following"], FILL["perc"], lsize=L)
    box(ax, 3.45, 0.15, 6.35, 3.6, "Skill experts on one frozen backbone", [], FILL["skill"], edge=BLUE, lw=1.2)
    ax.text(6.62, 3.1, "in: images + point map, step text, TCP pose + 9-D target", ha="center", fontsize=L, color=MUTED)
    ax.text(6.62, 2.85, "out: 12-D actions = TCP pose, gripper, [done, success]", ha="center", fontsize=L, color=MUTED)
    for k, sk in enumerate(["find", "pick", "place", "stack"]):
        x = 3.7 + k * 1.5
        ax.add_patch(FancyBboxPatch((x, 1.45), 1.35, 0.95, boxstyle="round,pad=0,rounding_size=0.06", fc="#ffffff",
                                    ec=BLUE, lw=1.0))
        ax.text(x + 0.675, 2.02, sk, ha="center", va="center", fontsize=7.0, fontweight="bold", color=BLUE)
        ax.text(x + 0.675, 1.68, "101 M", ha="center", va="center", fontsize=5.4, color=MUTED)
    ax.add_patch(FancyBboxPatch((3.7, 0.4), 5.85, 0.75, boxstyle="round,pad=0,rounding_size=0.06", fc="#ffffff", ec=RULE))
    ax.text(6.62, 0.86, "SmolVLA vision-language backbone, 350 M, frozen, shared", ha="center", va="center", fontsize=L)
    ax.text(6.62, 0.58, "active expert swapped in 0.8 ms", ha="center", va="center", fontsize=5.3, color=MUTED)
    # column C: camera, executor, adapter, robot
    box(ax, 10.3, 5.75, 3.6, 1.05, "Stereo head camera", ["2 \u00d7 512\u00b2, 6 cm baseline"], FILL["robot"], lsize=L)
    box(ax, 10.3, 2.7, 3.6, 2.6, "Executor", ["ends skill on verdict or 16 s", "grasp check: gripper opening",
                                               "miss \u2192 home, re-detect, retry", "release gate: open only",
                                               "near the target", "\u2192 next skill"], FILL["exec"], edge=ORANGE, lw=1.1,
        lsize=L)
    box(ax, 10.3, 1.4, 3.6, 0.9, "Robot adapter + IK", ["head-frame TCP \u2192 joints"], FILL["robot"], lsize=L)
    box(ax, 10.3, 0.15, 3.6, 0.85, "SO-101 arm (Isaac Lab)", [], FILL["robot"], tsize=6.8)
    # flows
    arrow(ax, (3.0, 4.6), (3.45, 5.2), "objects", tpos=0.2, toff=(-0.3, 0.12))
    arrow(ax, (3.0, 2.5), (3.45, 2.5), "skill, text", tpos=0.5, toff=(0.0, 0.07))
    arrow(ax, (5.9, 4.45), (5.9, 3.75), "9-D target pose", tpos=0.5, toff=(0.95, -0.05))
    arrow(ax, (10.3, 6.25), (8.35, 6.25), "images", tpos=0.5, toff=(0, 0.07))
    arrow(ax, (10.6, 5.75), (9.6, 3.75), "images", tpos=0.45, toff=(-0.45, 0.0))
    arrow(ax, (9.8, 3.25), (10.3, 3.6), "actions", tpos=0.5, toff=(0.0, 0.12))
    arrow(ax, (12.1, 2.7), (12.1, 2.3))
    arrow(ax, (12.1, 1.4), (12.1, 1.0))
    save(fig, "fig_architecture")


# ---------------------------------------------------------------- organisations
def fig_organisation():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    titles = ["(a) Monolithic", "(b) Task-level experts", "(c) Skill-level experts"]
    for ax, t in zip(axes, titles):
        ax.set_xlim(0, 6), ax.set_ylim(0, 5), ax.axis("off")
        ax.set_title(t, fontsize=8, fontweight="bold", color=INK, pad=2)
    a = axes[0]
    box(a, 0.2, 3.6, 5.6, 0.9, "all instructions", [], FILL["in"], tsize=6.6)
    box(a, 1.2, 1.7, 3.6, 1.2, "one model", ["all skills, all data"], "#fdf1ec", edge=ORANGE, lw=1.1, tsize=7, lsize=5.8)
    arrow(a, (3.0, 3.6), (3.0, 2.9))
    a.text(3.0, 0.9, "trains on: all 2,900 episodes", ha="center", fontsize=6.0, color=INK)
    a.text(3.0, 0.45, "new skill: retrain all (replay)", ha="center", fontsize=6.0, color=MUTED)

    def experts(ax, items, fc, ec, bold):
        for k, (n, sub) in enumerate(items):
            x = 0.15 + k * 1.95
            ax.add_patch(FancyBboxPatch((x, 1.7), 1.75, 1.2, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec,
                                        lw=1.0))
            ax.text(x + 0.875, 2.42, n, ha="center", va="center", fontsize=6.3 if bold else 5.7,
                    fontweight="bold" if bold else "normal", color=ec if bold else INK)
            ax.text(x + 0.875, 2.03, sub, ha="center", va="center", fontsize=5.2, color=MUTED)
            arrow(ax, (3.0, 3.6), (x + 0.875, 2.9), lw=0.6)

    b = axes[1]
    box(b, 0.2, 3.6, 5.6, 0.9, "instruction \u2192 router", [], FILL["in"], tsize=6.6)
    experts(b, [("blue\u2192blue", "pick + place"), ("red\u2192yellow", "pick + place"), ("\u2026 36 more", "")],
            "#fdf1ec", ORANGE, False)
    b.text(3.0, 0.9, "trains on: ~115 episodes each", ha="center", fontsize=6.0, color=INK)
    b.text(3.0, 0.45, "new instruction: new expert", ha="center", fontsize=6.0, color=MUTED)
    c = axes[2]
    box(c, 0.2, 3.6, 5.6, 0.9, "instruction \u2192 skills", [], FILL["in"], tsize=6.6)
    experts(c, [("pick", "+ target"), ("place", "+ target"), ("stack \u2026", "+ target")],
            "#e6eefa", BLUE, True)
    c.text(3.0, 0.9, "trains on: its skill in all 2,900", ha="center", fontsize=6.0, color=INK)
    c.text(3.0, 0.45, "new skill: add one expert", ha="center", fontsize=6.0, color=MUTED)
    fig.subplots_adjust(wspace=0.1)
    save(fig, "fig_organisation")


# ---------------------------------------------------------------- results
def fig_results():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw={"width_ratios": [1.35, 1]})
    # (a) experiment 4: full success per instruction
    labels = ["blue\n→ blue", "red\n→ yellow", "blue\n→ white", "red\n→ green", "blue\n→ orange", "red\n→ purple"]
    task = [72, 58, 72, 56, 48, 58]
    skill = [90, 96, 96, 98, 94, 94]
    xs = range(len(labels))
    w = 0.36
    a.bar([x - w / 2 - 0.01 for x in xs], task, w, color=ORANGE, label="task-level experts", zorder=2)
    a.bar([x + w / 2 + 0.01 for x in xs], skill, w, color=BLUE, label="skill-level experts", zorder=2)
    for x, t, s in zip(xs, task, skill):
        a.text(x - w / 2, t + 1.5, f"{t}", ha="center", fontsize=5.6, color=MUTED)
        a.text(x + w / 2, s + 1.5, f"{s}", ha="center", fontsize=5.6, color=MUTED)
    a.axvline(3.5, color=RULE, lw=0.8, ls="--", zorder=1)
    a.text(1.5, 108, "seen instructions (cube \u2192 plate)", ha="center", fontsize=6.4, color=INK)
    a.text(4.5, 108, "unseen", ha="center", fontsize=6.4, color=INK)
    a.set_xticks(list(xs), labels, fontsize=5.8, rotation=0)
    a.set_ylim(0, 115), a.set_yticks([0, 25, 50, 75, 100])
    a.set_ylabel("full success (%)", fontsize=6.6)
    a.grid(axis="y", color=RULE, lw=0.5, zorder=0)
    a.legend(frameon=False, fontsize=6.2, loc="lower left", ncol=2, bbox_to_anchor=(0, -0.36))
    a.set_title("(a) Exp. 2: task-level vs skill-level experts (50 episodes each)", fontsize=7.2, color=INK, loc="left")
    # (b) experiment 2: adding stack
    meth = ["place expert reused", "stack expert, 100 demos", "stack expert, 300 demos", "stack expert, from place",
            "monolith, stack only", "monolith + replay"]
    val = [0, 33, 53, 59, 0, 68]
    col = [GREY, BLUE, BLUE, BLUE, ORANGE, ORANGE]
    note = ["", "", "", "", "old skills: 0%", "old skills: 92%"]
    ys = list(range(len(meth)))[::-1]
    b.barh(ys, val, 0.6, color=col, zorder=2)
    for y, v_, n in zip(ys, val, note):
        b.text(v_ + 2, y, f"{v_}%" + (f"   {n}" if n else ""), va="center", fontsize=5.8, color=MUTED)
    b.set_yticks(ys, meth, fontsize=6.0)
    b.set_xlim(0, 105), b.set_xticks([0, 25, 50, 75, 100])
    b.set_xlabel("stacked (%)", fontsize=6.6)
    b.grid(axis="x", color=RULE, lw=0.5, zorder=0)
    b.set_title("(b) Exp. 3: adding the stack skill (150 episodes)", fontsize=7.2, color=INK, loc="left")
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2, width=0.5, labelsize=6.0, colors=MUTED)
    fig.subplots_adjust(wspace=0.55)
    save(fig, "fig_results")


def fig_composed():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1, 1.25]})
    tasks = ["two cubes\n\u2192 two plates", "plate,\nthen stack", "3-cube\ntower"]
    skills, mono = [86, 42, 16], [88, 62, 18]
    xs = range(3)
    w = 0.36
    a.bar([x - w / 2 - 0.01 for x in xs], skills, w, color=BLUE, label="skill-level experts", zorder=2)
    a.bar([x + w / 2 + 0.01 for x in xs], mono, w, color=ORANGE, label="monolith + replay", zorder=2)
    for x, s_, m_ in zip(xs, skills, mono):
        a.text(x - w / 2, s_ + 1.5, f"{s_}", ha="center", fontsize=5.8, color=MUTED)
        a.text(x + w / 2, m_ + 1.5, f"{m_}", ha="center", fontsize=5.8, color=MUTED)
    a.set_xticks(list(xs), tasks, fontsize=6.0)
    a.set_ylim(0, 105), a.set_yticks([0, 25, 50, 75, 100])
    a.set_ylabel("composed success (%)", fontsize=6.6)
    a.set_title("(a) Exp. 4: unseen composed tasks (50 episodes)", fontsize=7.2, color=INK, loc="left")
    a.legend(frameon=False, fontsize=6.0, loc="upper right")
    # (b) per step: first sub-task, and second given the first succeeded
    steps = ["1st cube\n\u2192 plate", "2nd cube\n\u2192 plate", "cube\n\u2192 plate", "stack on\nplated cube",
             "stack on\ncube", "stack on\n2 cubes"]
    sk = [94, 91, 94, 45, 62, 26]
    mo = [94, 94, 96, 65, 68, 26]
    xs2 = range(6)
    b.bar([x - w / 2 - 0.01 for x in xs2], sk, w, color=BLUE, zorder=2)
    b.bar([x + w / 2 + 0.01 for x in xs2], mo, w, color=ORANGE, zorder=2)
    for x, s_, m_ in zip(xs2, sk, mo):
        b.text(x - w / 2, s_ + 1.5, f"{s_}", ha="center", fontsize=5.4, color=MUTED)
        b.text(x + w / 2, m_ + 1.5, f"{m_}", ha="center", fontsize=5.4, color=MUTED)
    for x in (1.5, 3.5):
        b.axvline(x, color=RULE, lw=0.8, ls="--", zorder=1)
    for x, t in ((0.5, "two plates"), (2.5, "plate, then stack"), (4.5, "tower")):
        b.text(x, 108, t, ha="center", fontsize=6.0, color=INK)
    b.set_xticks(list(xs2), steps, fontsize=5.3)
    b.set_ylim(0, 115), b.set_yticks([0, 25, 50, 75, 100])
    b.set_ylabel("step success (%)", fontsize=6.6)
    b.set_title("(b) per step: 1st sub-task, 2nd given the 1st succeeded", fontsize=7.2, color=INK, loc="left")
    for ax in (a, b):
        ax.grid(axis="y", color=RULE, lw=0.5, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=2, width=0.5, labelsize=6.0, colors=MUTED)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "fig_composed")


if __name__ == "__main__":
    fig_scene()
    fig_architecture()
    fig_organisation()
    fig_results()
    fig_composed()
    print("written:", sorted(p.name for p in OUT.glob("fig_*")))
