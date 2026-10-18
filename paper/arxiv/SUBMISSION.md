# arXiv submission: Skill-Level Experts with Grounded Targets

Upload `skill_level_experts_arxiv.tar.gz` (built by `paper/make_arxiv.sh`; test-compiled with pdflatex only).

## Form fields

**Title**
Skill-Level Experts with Grounded Targets: Modularity and Generality in Language-Conditioned Manipulation

**Authors**
Prateek Pawar

**Abstract** (plain text, 1475 characters; limit 1920)
Vision-language-action (VLA) models are increasingly specialised by attaching small task-specific experts to a shared, frozen backbone. We ask at which granularity such experts should be defined. We compare three ways of organising the same model capacity, data and compute for language-conditioned pick-and-place on a 5-DoF arm in simulation: one monolithic model, task-level experts (one per instruction, as in recent LoRA-expert systems), and skill-level experts (one per primitive such as find, pick, place and stack, composed into tasks), all conditioned on a target pose provided by an open-vocabulary perception and tracking stack. In seeded, paired evaluations of 50-150 episodes per condition we find that (i) splitting by skill does not improve per-skill accuracy over a monolithic model trained on the same data; (ii) skill-level experts generalise far better than task-level experts, on seen (90-98% vs. 56-72% success) and unseen instructions (94% vs. 48-58%), because every skill learns from all episodes; (iii) a new skill can be added in isolation without any access to old data or risk to deployed skills, while naive fine-tuning of a monolithic model erases its previous skills (0%), and fine-tuning with replay retains them but requires the old data and full re-validation; (iv) on unseen multi-step tasks, skills compose almost independently for both organisations, and composite success is bounded by the weakest skill, here stacking onto raised targets.

**Comments**
6 pages, 5 figures, 4 tables. Code, per-episode results and videos: https://github.com/prateekspawar/Skill_based_VLA_training

**Primary category:** cs.RO (Robotics)
**Cross-list:** cs.LG (Machine Learning)

**License:** "arXiv.org perpetual, non-exclusive license" (the default). It keeps the later IEEE submission
(RA-L / ICRA / IROS) straightforward; IEEE allows the preprint to stay on arXiv.

## Before submitting
- First submission to cs.RO: arXiv may ask for an **endorsement** from an established cs.RO author; the form says so
  after you choose the category and gives an endorsement code to send to an endorser.
- Check the generated PDF arXiv shows before confirming (it should match `test_compile.pdf`, 6 pages).
- The source becomes public with the paper; `main.tex` contains no private notes.
