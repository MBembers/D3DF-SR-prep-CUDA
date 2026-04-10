import pandas as pd
import numpy as np

from time import perf_counter
from typing import Tuple, Union

try:
  from numba import njit, prange  # type: ignore[reportMissingImports]
  NUMBA_AVAILABLE = True
except Exception:
  NUMBA_AVAILABLE = False

  def njit(*args, **kwargs):
    def decorator(func):
      return func
    return decorator

  def prange(*args):
    return range(*args)


# FSF renormalization target range
max_fsf_scaling = 0.98
min_fsf_scaling = 0.02


def parse_mlc_data(filename: str) -> Tuple[np.ndarray, np.ndarray, dict[str, Union[float, int]]]:
  with open(filename) as fp:
    data = fp.read()
  lines = data.strip().split('\n')

  comments = [line for line in lines if line.startswith('#')]
  comment_params = {}
  for comment in comments:
    key = comment.removeprefix('# ').rstrip('0123456789. :')
    value = comment.removeprefix('# ').split(' ', 1)[1]
    if '.' in value:
      try:
        value = float(value)
      except ValueError:
        continue
    else:
      try:
        value = int(value)
      except ValueError:
        continue
    comment_params[key] = value

  lines = [line for line in lines if not line.startswith('#')]

  jaws = np.array(list(map(float, lines[0].split(','))), dtype=np.float64)

  mlc_positions = []
  for line in lines[1:]:
    if ',' in line:
      left, right = map(float, line.split(','))
      mlc_positions.append((left, right))

  return jaws, np.array(mlc_positions, dtype=np.float64), comment_params


@njit(cache=True, fastmath=False)
def angle_between_components(v1x: float, v1y: float, v1z: float,
                             v2x: float, v2y: float, v2z: float) -> float:
  v1_len = np.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
  v2_len = np.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
  denom = v1_len * v2_len
  if denom == 0.0:
    return 0.0
  cos_angle = (v1x * v2x + v1y * v2y + v1z * v2z) / denom
  if cos_angle > 1.0:
    cos_angle = 1.0
  elif cos_angle < -1.0:
    cos_angle = -1.0
  return np.arccos(cos_angle)


@njit(cache=True, fastmath=False)
def calculate_fsf(x: float, y: float, z: float,
                        nx: int, ny: int, nz: int,
                        mlc_positions: np.ndarray) -> float:
  n_leaves = mlc_positions.shape[0]

  leaf_height = 2.5
  mlc_detector_distance = -626.25

  r_mlc_x = -x
  r_mlc_y = -y
  r_mlc_z = mlc_detector_distance - z

  y1_mlc = 0.0
  y2_mlc = 0.0

  current_x = -n_leaves * leaf_height / 2.0 + (leaf_height / 2.0)
  for i in range(n_leaves):
    y_left = mlc_positions[i, 0]
    y_right = mlc_positions[i, 1]

    r_leaf_left_x = current_x - x
    r_leaf_left_y = y_left - y
    r_leaf_left_z = mlc_detector_distance - z

    r_leaf_right_x = current_x - x
    r_leaf_right_y = y_right - y
    r_leaf_right_z = mlc_detector_distance - z

    lambda_left = angle_between_components(
      r_mlc_x, r_mlc_y, r_mlc_z,
      r_leaf_left_x, r_leaf_left_y, r_leaf_left_z,
    )
    lambda_right = angle_between_components(
      r_mlc_x, r_mlc_y, r_mlc_z,
      r_leaf_right_x, r_leaf_right_y, r_leaf_right_z,
    )

    y1_mlc += lambda_left * lambda_left
    y2_mlc += lambda_right * lambda_right

    current_x += leaf_height

  fsf = y1_mlc + y2_mlc
  return fsf / (1.0 * nx + 2.0 * ny + 3.0 * nz)


@njit(cache=True, parallel=True, fastmath=False)
def compute_fsf_for_points(x_vals: np.ndarray,
                           y_vals: np.ndarray,
                           z_vals: np.ndarray,
                           nx: int,
                           ny: int,
                           nz: int,
                           mlc_positions: np.ndarray) -> np.ndarray:
  n = x_vals.shape[0]
  out = np.zeros(n, dtype=np.float64)
  for i in prange(n):
    out[i] = calculate_fsf(
      x_vals[i], y_vals[i], z_vals[i],
      nx, ny, nz,
      mlc_positions,
    )
  return out


def renormalize_fsf(fsf_array: np.ndarray) -> np.ndarray:
  out = fsf_array.copy()
  nonzero_mask = out != 0.0
  if not np.any(nonzero_mask):
    return out

  nonzero = out[nonzero_mask]
  max_fsf = float(np.max(nonzero))
  min_fsf = float(np.min(nonzero))

  if max_fsf == min_fsf:
    out[nonzero_mask] = min_fsf_scaling
    return out

  out[nonzero_mask] = (
    (nonzero - min_fsf) / (max_fsf - min_fsf)
  ) * (max_fsf_scaling - min_fsf_scaling) + min_fsf_scaling
  return out


def _build_subvoxel_offsets(voxel_no_per_cell: int,
                            cell_size: float,
                            target_resolution: float) -> np.ndarray:
  start = -cell_size / 2.0 + target_resolution / 2.0
  end = cell_size / 2.0 - target_resolution / 2.0
  return np.linspace(start, end, voxel_no_per_cell, dtype=np.float64)


def enrich_df(input_df: pd.DataFrame,
                          jaws: np.ndarray,
                          mlc_positions: np.ndarray,
                          target_resolution: int = 1,
                          cell_size: int = 10) -> pd.DataFrame:
  assert (cell_size // target_resolution) % 2 == 0, (
    f"cell_size // target_resolution must be even, but is {cell_size // target_resolution}"
  )

  t0 = perf_counter()

  voxel_no_per_cell = int(cell_size // target_resolution)

  unique_input_xs = np.unique(input_df['X [mm]'].to_numpy(dtype=np.float64))
  unique_input_ys = np.unique(input_df['Y [mm]'].to_numpy(dtype=np.float64))
  unique_input_zs = np.unique(input_df['Z [mm]'].to_numpy(dtype=np.float64))

  cell_no_x = int(unique_input_xs.shape[0])
  cell_no_y = int(unique_input_ys.shape[0])
  cell_no_z = int(unique_input_zs.shape[0])

  patient_cell_material = 0.28947
  air = 0.0211788

  total_space_len = 64
  origin = -total_space_len / 2.0 + target_resolution / 2.0
  n_grid = int(total_space_len // target_resolution)

  # Input columns as contiguous arrays
  cell_idx_arr = input_df['Cell IdX'].to_numpy(dtype=np.int32)
  cell_idy_arr = input_df['Cell IdY'].to_numpy(dtype=np.int32)
  cell_idz_arr = input_df['Cell IdZ'].to_numpy(dtype=np.int32)
  center_x_arr = input_df['X [mm]'].to_numpy(dtype=np.float64)
  center_y_arr = input_df['Y [mm]'].to_numpy(dtype=np.float64)
  center_z_arr = input_df['Z [mm]'].to_numpy(dtype=np.float64)
  dose_arr = input_df['Dose [Gy]'].to_numpy(dtype=np.float64)

  n_cells = center_x_arr.shape[0]
  points_per_cell = voxel_no_per_cell ** 3
  n_filled = n_cells * points_per_cell

  print(f"expanding {n_cells} cells into {n_filled} detector voxels...")

  # Dense lookup arrays: O(1) access per space voxel.
  has_detector = np.zeros((n_grid, n_grid, n_grid), dtype=np.bool_)
  grid_cell_idx = np.full((n_grid, n_grid, n_grid), -1, dtype=np.int32)
  grid_cell_idy = np.full((n_grid, n_grid, n_grid), -1, dtype=np.int32)
  grid_cell_idz = np.full((n_grid, n_grid, n_grid), -1, dtype=np.int32)
  grid_voxel_idx = np.full((n_grid, n_grid, n_grid), -1, dtype=np.int16)
  grid_voxel_idy = np.full((n_grid, n_grid, n_grid), -1, dtype=np.int16)
  grid_voxel_idz = np.full((n_grid, n_grid, n_grid), -1, dtype=np.int16)
  grid_dose = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)

  offsets = _build_subvoxel_offsets(voxel_no_per_cell, float(cell_size), float(target_resolution))

  # Bounding box in index space for fast in-range mask.
  min_ix = n_grid
  min_iy = n_grid
  min_iz = n_grid
  max_ix = -1
  max_iy = -1
  max_iz = -1

  # Building the lookup arrays
  for c in range(n_cells):
    cx = center_x_arr[c]
    cy = center_y_arr[c]
    cz = center_z_arr[c]

    for vx in range(voxel_no_per_cell):
      x = cx + offsets[vx]
      ix = int(round((x - origin) / target_resolution))
      if ix < 0 or ix >= n_grid:
        continue
      for vy in range(voxel_no_per_cell):
        y = cy + offsets[vy]
        iy = int(round((y - origin) / target_resolution))
        if iy < 0 or iy >= n_grid:
          continue
        for vz in range(voxel_no_per_cell):
          z = cz + offsets[vz]
          iz = int(round((z - origin) / target_resolution))
          if iz < 0 or iz >= n_grid:
            continue

          has_detector[ix, iy, iz] = True
          grid_cell_idx[ix, iy, iz] = cell_idx_arr[c]
          grid_cell_idy[ix, iy, iz] = cell_idy_arr[c]
          grid_cell_idz[ix, iy, iz] = cell_idz_arr[c]
          grid_voxel_idx[ix, iy, iz] = vx
          grid_voxel_idy[ix, iy, iz] = vy
          grid_voxel_idz[ix, iy, iz] = vz
          grid_dose[ix, iy, iz] = dose_arr[c]

          if ix < min_ix:
            min_ix = ix
          if iy < min_iy:
            min_iy = iy
          if iz < min_iz:
            min_iz = iz
          if ix > max_ix:
            max_ix = ix
          if iy > max_iy:
            max_iy = iy
          if iz > max_iz:
            max_iz = iz

  t1 = perf_counter()
  print(f"lookup built in {t1 - t0:.2f}s")

  if max_ix == -1:
    raise RuntimeError('No detector voxels were mapped into output space.')

  n_total = n_grid ** 3
  x_coords = origin + np.arange(n_grid, dtype=np.float64) * target_resolution
  y_coords = origin + np.arange(n_grid, dtype=np.float64) * target_resolution
  z_coords = origin + np.arange(n_grid, dtype=np.float64) * target_resolution

  # Materialize full output arrays once.
  out_cell_idx = np.full(n_total, -1, dtype=np.int32)
  out_cell_idy = np.full(n_total, -1, dtype=np.int32)
  out_cell_idz = np.full(n_total, -1, dtype=np.int32)
  out_voxel_idx = np.full(n_total, -1, dtype=np.int16)
  out_voxel_idy = np.full(n_total, -1, dtype=np.int16)
  out_voxel_idz = np.full(n_total, -1, dtype=np.int16)
  out_material = np.full(n_total, air, dtype=np.float64)
  out_dose = np.zeros(n_total, dtype=np.float64)
  out_fsf = np.zeros(n_total, dtype=np.float64)

  out_x = np.empty(n_total, dtype=np.float64)
  out_y = np.empty(n_total, dtype=np.float64)
  out_z = np.empty(n_total, dtype=np.float64)

  # Collecting coordinates of detector voxels for FSF computation.
  detector_linear_indices = []
  detector_x = []
  detector_y = []
  detector_z = []

  idx = 0
  for ix in range(n_grid):
    x = x_coords[ix]
    in_x = min_ix <= ix <= max_ix
    for iy in range(n_grid):
      y = y_coords[iy]
      in_xy = in_x and (min_iy <= iy <= max_iy)
      for iz in range(n_grid):
        z = z_coords[iz]

        out_x[idx] = x
        out_y[idx] = y
        out_z[idx] = z

        if in_xy and (min_iz <= iz <= max_iz) and has_detector[ix, iy, iz]:
          out_cell_idx[idx] = grid_cell_idx[ix, iy, iz]
          out_cell_idy[idx] = grid_cell_idy[ix, iy, iz]
          out_cell_idz[idx] = grid_cell_idz[ix, iy, iz]
          out_voxel_idx[idx] = grid_voxel_idx[ix, iy, iz]
          out_voxel_idy[idx] = grid_voxel_idy[ix, iy, iz]
          out_voxel_idz[idx] = grid_voxel_idz[ix, iy, iz]
          out_material[idx] = patient_cell_material
          out_dose[idx] = grid_dose[ix, iy, iz]

          detector_linear_indices.append(idx)
          detector_x.append(x)
          detector_y.append(y)
          detector_z.append(z)

        idx += 1

  detector_linear_indices = np.array(detector_linear_indices, dtype=np.int64)
  detector_x = np.array(detector_x, dtype=np.float64)
  detector_y = np.array(detector_y, dtype=np.float64)
  detector_z = np.array(detector_z, dtype=np.float64)

  t2 = perf_counter()
  print(f"space arrays prepared in {t2 - t1:.2f}s")

  if detector_linear_indices.shape[0] > 0:
    # JIT-accelerated FSF for detector voxels only.
    fsf_vals = compute_fsf_for_points(
      detector_x,
      detector_y,
      detector_z,
      cell_no_x,
      cell_no_y,
      cell_no_z,
      mlc_positions,
    )
    out_fsf[detector_linear_indices] = fsf_vals

  t3 = perf_counter()
  print(f"fsf computed in {t3 - t2:.2f}s")

  out_fsf = renormalize_fsf(out_fsf)

  output_df = pd.DataFrame({
    'Cell IdX': out_cell_idx,
    'Cell IdY': out_cell_idy,
    'Cell IdZ': out_cell_idz,
    'Voxel IdX': out_voxel_idx,
    'Voxel IdY': out_voxel_idy,
    'Voxel IdZ': out_voxel_idz,
    'X [mm]': out_x,
    'Y [mm]': out_y,
    'Z [mm]': out_z,
    'Material': out_material,
    'Dose': out_dose,
    'FieldScalingFactor': out_fsf,
  })

  t4 = perf_counter()
  print(f"final dataframe built in {t4 - t3:.2f}s")
  print(f"total enrich time: {t4 - t0:.2f}s")
  return output_df


if __name__ == '__main__':
  target_resolution = 1
  cell_size = 10

  data_filename = 'data/new/prostate_imrt_beam0_cp74_d3ddetector_cell.csv'
  mlc_filename = 'data/new/prostate_imrt_beam0_cp74.dat'

  if not NUMBA_AVAILABLE:
    print('warning: numba is not installed, running without JIT acceleration')

  raw_df = pd.read_csv(data_filename)
  jaws, mlc_positions, _ = parse_mlc_data(mlc_filename)

  sorted_df = raw_df.sort_values(by=['X [mm]', 'Y [mm]', 'Z [mm]'])

  start = perf_counter()
  output_df = enrich_df(
    sorted_df,
    jaws,
    mlc_positions,
    target_resolution=target_resolution,
    cell_size=cell_size,
  )
  end = perf_counter()

  output_df.to_csv('outputs/output_numba.csv', index=False)
  write_time = perf_counter() - end
  print(f'output generated in {end - start:.2f} seconds -> outputs/output_numba.csv')
  print(f'write time: {write_time:.2f} seconds')
