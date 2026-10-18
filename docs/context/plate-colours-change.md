---
name: plate-colours-change
description: User plans to change plate colours (white/pink) to distinct saturated colours (e.g. red/blue) later
metadata:
  node_type: memory
  type: project
  originSessionId: d2915f31-430d-4df2-9436-ea337f837e9a
  modified: 2026-10-07T03:17:40.847Z
---

As of 2026-10-07 the plates are white and pink (`language.PLATE_COLORS`). The pale pink causes OWLv2 to confuse the plates: a wrong plate is picked at the first detection ~22% of the time, which causes most place failures. The user said they will change the plate colours to distinct saturated colours (e.g. red and blue) later.

**Why:** colour-distinct plates fix the plate first-detection errors without detector hacks.
**How to apply:** don't spend effort on plate colour-verification heuristics. When the colours change, check that no cube shares a plate colour (cubes are red/green/blue/yellow), regenerate data, and re-measure place accuracy. Related: [[find-pipeline-gaps]]
