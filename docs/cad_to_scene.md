# CAD to Scene

## Overview
This document will provide information when translating objects from CAD into the standard interface - Open USD - for Isaac Sim.
This tutorial will use the construction of a table in Onshape as an example.

1) First create your object in OnShape

## Known Converter Artifacts (usd-convert-cad / HOOPS)

Our `table.usd`, `CardboardBox.usd`, and `Block.usd` were converted OnShape → JT →
USD with `usd-convert-cad`. Inspecting the resulting stages in Isaac Sim surfaced
three artifacts that trace back to the HOOPS converter **defaults**
(`instancingStyle=2`, `useMaterials=true`, `tessLOD=2` — see the
`omniverse-cad-to-usd` skill). These are asset-level issues; do **not** paper over
them in the env config. The permanent fix is re-conversion with the right options
and/or fixing the CAD export, then re-verifying.

Measured facts (all three: `metersPerUnit=0.001`, Z-up, one bound `Diffuse`
UsdPreviewSurface, geometry under an `instanceable` prim):

| Asset | Imported size (m, X×Y×Z) | Material color |
|---|---|---|
| `table.usd` | 1.000 × 0.250 × **1.275** | `(0.616, 0.812, 0.929)` pale blue |
| `CardboardBox.usd` | 0.111 × 0.111 × 0.185 | same pale blue |
| `Block.usd` | 0.030 × 0.030 × 0.060 | same pale blue |

### 1. Instanceable geometry blocks runtime overrides
Every asset's mesh sits under an `instanceable=True` prim
(`/table/tn__Part1_f5`, `/CardboardBox/CardboardBox`, `/Block/Block`). Isaac Lab
silently skips runtime API edits on instanced prims, so:
- `collision_props` on the cardboard box is dropped → **the box has no collider**
  (log: `Could not perform 'modify_collision_properties' ... instanced prim paths`).
- Any cfg-level `visual_material` color override on these USDs won't bind either.

**Candidate permanent fix (to verify):** re-convert with
`--option instancingStyle=0` (or `1`) so geometry is not instanced, e.g.
`python convert.py Block.jt Block.usd --option instancingStyle=0 --report status.json`.
Then cfg-level collision/material overrides should apply.

### 2. Materials flattened to one placeholder color
Despite `useMaterials=true`, all three USDs carry the *same* pale-blue `Diffuse`
material — the CAD/JT appearance (red/blue blocks, brown box) did not carry
through. Likely the JT export from OnShape does not embed per-body appearance, so
HOOPS assigns a default.

**Candidate permanent fix (to verify):** ensure the OnShape/JT export includes
materials/appearances; or author correct UsdPreviewSurface colors post-conversion;
or bind materials in-scene once the geometry is non-instanceable (see #1).

### 3. Table imports rotated (up-axis mismatch)
`table.usd` is correctly **scaled** (real size 1.0 × 0.25 × 1.275 m) but its
1.275 m extent lands on the **Z / up axis** — it stands tall instead of lying
flat. The CAD was exported Y-up while the USD stage is Z-up.

**Candidate permanent fix (to verify):** export with the correct up-axis or apply
a corrective author-time rotation; confirm the converter's up-axis handling.

### Current scene consequences (intentional, pending re-conversion)
- Arm + lightbox run at ground level (working, unchanged). The **table is a
  visual backdrop only**, not the mounting surface — it cannot sit under the
  lightbox until #3 is fixed.
- The **cardboard box has no collision** (#1); blocks rest on the surface at the
  box footprint rather than being contained.
- Table and box render **pale blue**, not wood/cardboard (#2). Blocks are fine —
  they are `CuboidCfg` primitives colored in-sim, not the CAD `Block.usd`.

## Resources
- OpenUSD Fundamentals: https://docs.isaacsim.omniverse.nvidia.com/6.0.1/omniverse_usd/open_usd.html
- CAD to USD Workflows: https://developer.nvidia.com/blog/building-cad-to-usd-workflows-with-nvidia-omniverse/
