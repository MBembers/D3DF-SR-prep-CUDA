# converter.py — Pipeline Analysis

## Overview

`converter.py` is the pre-processing step of a super-resolution pipeline for radiotherapy (RT) dose data. 
It takes a low-resolution detector readout (coarse cells, ~10 mm) and a MLC configuration file, and produces a fine-grained voxel grid (target ~1 mm) annotated with dose and a Field Scaling Factor (FSF).

---

## Top-level Call Graph (`__main__`)

```
__main__
├── pd.read_csv(data_filename)             ← load low-res detector cells
├── parse_mlc_data(mlc_filename)           ← parse MLC/jaw geometry
├── enrich_df(sorted_df, jaws, mlc_positions, target_resolution, cell_size)
│   ├── [Phase A] Cell Expansion Loop      ← expand each cell → sub-voxels
│   ├── [Phase B] Space Filling Loop       ← fill full 64³ grid
│   │   └── calculate_fsf(x, y, z, ...)   ← called for every detector voxel
│   │       └── angle_between(v1, v2)      ← called for every MLC leaf pair
│   └── renormalize_fsf(output_df)         ← min-max rescale FSF column
├── df.to_csv("output.csv")                ← write result
└── plot_df(df, ...)                       ← 3-D visualisation
    └── add_voxel(ax, ...)                 ← called per non-zero voxel
```

---

## Stage-by-Stage Breakdown

### 1. Input Loading

| Item | Description |
|---|---|
| **File** | `*_d3ddetector_cell.csv` |
| **Columns used** | `X [mm]`, `Y [mm]`, `Z [mm]`, `Cell IdX/Y/Z`, `Dose [Gy]` |
| **Resolution** | Coarse — one row per detector cell (~10 mm spacing) |

The DataFrame is sorted by `(X, Y, Z)` before being passed downstream.

---

### 2. `parse_mlc_data(filename)`

Parses the `.dat` MLC configuration file produced by the treatment planning system.

**Steps:**
1. Read raw text file.
2. Extract `comment_params` — key/value metadata (e.g. beam energy, number of leaves) from `#`-prefixed lines.
3. Parse the first non-comment line as **jaw boundaries**: `(X1, X2, Y1, Y2)` in mm.
4. Parse remaining non-comment lines as **MLC leaf pairs**: `(left_pos, right_pos)` in mm, one pair per leaf row (60 leaves total).

**Returns:** `(jaws: ndarray[4], mlc_positions: ndarray[N×2], comment_params: dict)`

---

### 3. `enrich_df(input_df, jaws, mlc_positions, target_resolution, cell_size)`

This is the **core and most compute-intensive** function. It produces the full output grid.

#### Phase A — Cell Expansion

For each row (coarse cell) in `input_df`:

- Compute `voxel_no_per_cell = cell_size // target_resolution` (e.g. 10 for 1 mm resolution inside a 10 mm cell).
- Use `np.linspace` to generate a uniform grid of `voxel_no_per_cell³` fine sub-voxel coordinates centred on the cell.
- Append each sub-voxel as a dict to `filled_points` list, carrying:
  - Fine coordinates `(x, y, z)`
  - Parent cell IDs `(cell_idx, cell_idy, cell_idz)`
  - Local voxel IDs within the cell `(voxel_idx, voxel_idy, voxel_idz)`
  - Inherited `dose`
- Track overall bounding box (`min/max_voxel_x/y/z`) across all sub-voxels.

**Complexity:** `O(N_cells × voxel_no_per_cell³)` — typically small.

#### Phase B — Space Filling (dominant bottleneck)

Iterates over the **full 64 × 64 × 64 mm space** at `target_resolution` step (64³ = ~262 k points at 1 mm):

For each grid point `(x, y, z)`:

1. **Bounding box check** — if outside the detector bounding box → classify as **air** (Material = 0.0211788, Dose = 0, FSF = 0).
2. **Linear search** through `filled_points` to find a coordinate match.
3. If **match found** → classify as **detector voxel**:
   - Copy cell/voxel IDs and dose from `filled_points[index]`.
   - Call `calculate_fsf(x, y, z, ...)` → assign `FieldScalingFactor`.
   - `del filled_points[index]` (to shrink the search space as matches are consumed).
4. If **no match** → classify as **air**.

**Complexity:** `O(grid_size³ × |filled_points|)` — this nested loop with a linear search is the primary performance bottleneck.

#### Phase C — FSF Renormalisation

Calls `renormalize_fsf(output_df)` before returning.

---

### 4. `calculate_fsf(x, y, z, nx, ny, nz, jaws, mlc_positions)`

Computes the **Field Scaling Factor** for a single voxel — a geometric approximation of beam fluence at that point based on the MLC aperture.

**Steps:**
1. Define the MLC plane position (`mlc_detector_distance = -626.25 mm` above isocentre).
2. Compute reference vector `r_mlc` from voxel to MLC centre.
3. For each of the `N` MLC leaf pairs:
   - Compute position of left and right leaf edges in 3-D.
   - Compute vectors from voxel to each leaf edge.
   - Call `angle_between(r_mlc, r_leaf_left)` and `angle_between(r_mlc, r_leaf_right)`.
   - Accumulate squared angles into `y1MLC` and `y2MLC`.
4. Sum: `fsf = y1MLC + y2MLC`.
5. Normalise by `(nx + 2·ny + 3·nz)`.

**Complexity per call:** `O(N_leaves)` — called once per detector voxel in Phase B.

---

### 5. `angle_between(v1, v2)`

Leaf-level primitive: computes the angle between two 3-D vectors using the dot product formula.

```
arccos( dot(v1,v2) / (|v1| · |v2|) )
```

**Called:** `2 × N_leaves` times per `calculate_fsf` call.

---

### 6. `renormalize_fsf(dataframe)`

Min-max rescales all non-zero FSF values to the fixed range `[0.02, 0.98]`:

```
FSF_norm = (FSF - min) / (max - min) × (0.98 - 0.02) + 0.02
```

Applied once, on the completed output DataFrame.

---

### 7. Output — `df.to_csv("output.csv")`

Writes the completed voxel grid to disk. Each row is one fine voxel with columns:

| Column | Description |
|---|---|
| `Cell IdX/Y/Z` | Parent coarse cell index (-1 for air) |
| `Voxel IdX/Y/Z` | Sub-voxel index within cell (-1 for air) |
| `X/Y/Z [mm]` | Fine voxel centre coordinates |
| `Material` | Density: 0.28947 (tissue) or 0.0211788 (air) |
| `Dose` | Inherited dose from parent cell [Gy] |
| `FieldScalingFactor` | Beam aperture weight (renormalised 0.02–0.98, 0 for air) |

---

### 8. `plot_df(df, ...)` — Visualisation (optional)

Produces a 3-D matplotlib scatter/voxel plot of the `FieldScalingFactor` channel.

- Filters rows with `FieldScalingFactor > 0`.
- For each such row calls `add_voxel(ax, ...)` which renders a coloured cube using `ax.voxels()`.
- Adds a colourbar and axis labels.

**Not part of the data output** — purely diagnostic.

---

## Data Flow Diagram

```
 [*.dat MLC file]          [*_cell.csv detector data]
       │                              │
       ▼                              ▼
 parse_mlc_data()            pd.read_csv() + sort
       │                              │
       │  jaws, mlc_positions         │  low-res cell DataFrame
       └──────────────┬───────────────┘
                      ▼
              enrich_df() - 35.5s on my machine 
              │
              ├─ [Phase A] Cell Expansion
              │   └─ linspace sub-division of each cell
              │      → filled_points list (dict of fine voxels)
              |      → min/max voxel ranges
              │
              ├─ [Phase B] Space Filling  ← BOTTLENECK - 34.98s (98% of enrich_df())
              │   └─ for every point in 64x64x64 grid:
              │       ├─ air?   → zero-fill
              │       └─ hit?   → calculate_fsf() - 22.2s in total
              │                     └─ angle_between()  [×2·N_leaves]
              │
              └─ [Phase C] renormalize_fsf()
                      │
                      ▼
              output DataFrame
              │
              ├─ to_csv("output.csv") 
              └─ plot_df()     
                     
```

---

## Performance Notes (Pre-CUDA Analysis)

| Bottleneck | Why | CUDA Opportunity |
|---|---|---|
| **Phase B outer triple loop** | 64³ points iterated in pure Python | Each grid point is independent → trivially parallelisable |
| **Linear search in `filled_points`** | O(M) per grid point, M up to `N_cells × voxel_no_per_cell³` | Replace with hash-map lookup or sort+binary search; on GPU: build a coordinate lookup table in shared memory |
| **`calculate_fsf` per detector voxel** | Calls `angle_between` 2×60 = 120 times with pure Python loops | Fully data-parallel across voxels; all MLC data fits in constant/shared memory |
| **`angle_between`** | Individual `np.sqrt` + `arccos` per call | Vectorisable over the leaf axis; on GPU: one warp per voxel, one thread per leaf |
| **Phase A cell expansion** | Python triple loop, but small size | Low priority; straightforward to vectorise with `np.meshgrid` even on CPU |
