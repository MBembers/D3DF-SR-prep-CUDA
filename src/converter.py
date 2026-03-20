import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib.axes import Axes

phantom_counts_cmap = cm.magma

import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple, Union
from time import perf_counter

import src.plotting as plotting
import src.plotting2 as plotting2


## MLC ##

def parse_mlc_data(filename: str) -> Tuple[np.ndarray, np.ndarray, dict[str, Union[float, int]]]:
  with open(filename) as fp:
    data = fp.read()
  lines = data.strip().split('\n')
  
  # read data from comments
  comments = [line for line in lines if line.startswith('#')]
  comment_params = {}
  for comment in comments:
    key = comment.removeprefix("# ").rstrip("0123456789. :")
    value = comment.removeprefix('# ').split(" ", 1)[1]
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
  
  # remove comments, no longer needed
  lines = [line for line in lines if not line.startswith("#")]
  
  # Parse jaws data
  jaws = np.array(list(map(float, lines[0].split(','))))
  
  # Parse MLC positions
  mlc_positions = []
  for line in lines[1:]:
    if ',' in line:
      left, right = map(float, line.split(','))
      mlc_positions.append((left, right))
  
  return jaws, np.array(mlc_positions), comment_params

## OUTPUT GENERATION ##

def enrich_df(input_df: pd.DataFrame, jaws: np.ndarray, mlc_positions: np.ndarray, target_resolution: int=1, cell_size: int=10) -> pd.DataFrame:
  # otherwise "air" will "stick out" of grid 
  assert (cell_size//target_resolution) % 2 == 0, f"cell_size // target_resolution must be even, but is {cell_size//target_resolution}"

  unique_input_xs = sorted(input_df['X [mm]'].unique())
  unique_input_ys = sorted(input_df['Y [mm]'].unique())
  unique_input_zs = sorted(input_df['Z [mm]'].unique())

  voxel_no_per_cell = int(cell_size//target_resolution)
  cell_no_x = len(unique_input_xs)
  cell_no_y = len(unique_input_ys)
  cell_no_z = len(unique_input_zs)

  patient_cell_material = 0.28947
  air = 0.0211788

  # assuming cube space
  total_space_len = 64 # mm
  space_x_range = (-total_space_len+total_space_len//2, total_space_len-total_space_len//2)
  space_y_range = (-total_space_len+total_space_len//2, total_space_len-total_space_len//2)
  space_z_range = (-total_space_len+total_space_len//2, total_space_len-total_space_len//2)

  print("iterating over input...")
  filled_points = []

  # initial values are out of range, for correct min/max calculation
  min_voxel_x =  total_space_len
  max_voxel_x = -total_space_len
  min_voxel_y =  total_space_len
  max_voxel_y = -total_space_len
  min_voxel_z =  total_space_len
  max_voxel_z = -total_space_len

  # ANALYSIS
  analysis = {}
  
  for _, row in input_df.iterrows():
    # remember parameters from current row
    cell_idx = row["Cell IdX"]
    cell_idy = row["Cell IdY"]
    cell_idz = row["Cell IdZ"]
    dose = row["Dose [Gy]"]

    # generate new coord ranges for voxel entries
    filled_xs = np.linspace(
      row["X [mm]"] - cell_size/2 + target_resolution/2, 
      row["X [mm]"] + cell_size/2 - target_resolution/2, 
      voxel_no_per_cell
    )
    filled_ys = np.linspace(
      row["Y [mm]"] - cell_size/2 + target_resolution/2, 
      row["Y [mm]"] + cell_size/2 - target_resolution/2, 
      voxel_no_per_cell
      )
    filled_zs = np.linspace(
      row["Z [mm]"] - cell_size/2 + target_resolution/2, 
      row["Z [mm]"] + cell_size/2 - target_resolution/2, 
      voxel_no_per_cell
    )

    # studying this code...
    # print(f"filled_xs: {filled_xs}")
    # print(f"filled_ys: {filled_ys}")
    # print(f"filled_zs: {filled_zs}")
    # print()

    for voxel_idx, x in enumerate(filled_xs):
      for voxel_idy, y in enumerate(filled_ys):
        for voxel_idz, z in enumerate(filled_zs):
          filled_points.append({
            "x": x, 
            "y": y, 
            "z": z, 
            "cell_idx": int(cell_idx), 
            "cell_idy": int(cell_idy), 
            "cell_idz": int(cell_idz), 
            "voxel_idx": voxel_idx, 
            "voxel_idy": voxel_idy, 
            "voxel_idz": voxel_idz, 
            "dose": dose
          })
          # update voxel ranges
          min_voxel_x = min(min_voxel_x, x)
          max_voxel_x = max(max_voxel_x, x)
          min_voxel_y = min(min_voxel_y, y)
          max_voxel_y = max(max_voxel_y, y)
          min_voxel_z = min(min_voxel_z, z)
          max_voxel_z = max(max_voxel_z, z)
  
  print(f"iterated over input, generated {len(filled_points)} filled points.")
  print(f"filling {total_space_len}x{total_space_len}x{total_space_len} space...")
  print(f"voxel x range: {min_voxel_x} - {max_voxel_x}")
  print(f"voxel y range: {min_voxel_y} - {max_voxel_y}")
  print(f"voxel z range: {min_voxel_z} - {max_voxel_z}")
  raw_output = []
  x_iterstart = space_x_range[0] + target_resolution/2
  x_iterend   = space_x_range[1] - target_resolution/2 + 1
  y_iterstart = space_y_range[0] + target_resolution/2
  y_iterend   = space_y_range[1] - target_resolution/2 + 1
  z_iterstart = space_z_range[0] + target_resolution/2
  z_iterend   = space_z_range[1] - target_resolution/2 + 1

  start_big_loop = perf_counter()
  for x in np.arange(x_iterstart, x_iterend, target_resolution):
    print(f"x = {x}", end="\r", flush=True)
    for y in np.arange(y_iterstart, y_iterend, target_resolution):
      for z in np.arange(z_iterstart, z_iterend, target_resolution):
        # filter out points outside voxel ranges
        if x < min_voxel_x or x > max_voxel_x or \
           y < min_voxel_y or y > max_voxel_y or \
           z < min_voxel_z or z > max_voxel_z:
          # "air" case
          raw_output.append({
            "Cell IdX": -1,
            "Cell IdY": -1,
            "Cell IdZ": -1,
            "Voxel IdX": -1,
            "Voxel IdY": -1,
            "Voxel IdZ": -1,
            "X [mm]": x,
            "Y [mm]": y,
            "Z [mm]": z,
            "Material": air,
            "Dose": 0,
            "FieldScalingFactor": 0
          })
          continue
        # check if current coords match any entry in filled points
        for index, entry in enumerate(filled_points):
          analysis["matching_filled_points_checks"] = analysis.get("matching_filled_points_checks", 0) + 1
          if x == entry['x'] and y == entry['y'] and z == entry['z']:
            break
        else:

          index = -1
        if index != -1:
          # coords match entry, "detector" case
          start_fsf_calc = perf_counter()
          raw_output.append({
            "Cell IdX": filled_points[index]["cell_idx"],
            "Cell IdY": filled_points[index]["cell_idy"],
            "Cell IdZ": filled_points[index]["cell_idz"],
            "Voxel IdX": filled_points[index]["voxel_idx"],
            "Voxel IdY": filled_points[index]["voxel_idy"],
            "Voxel IdZ": filled_points[index]["voxel_idz"],
            "X [mm]": x,
            "Y [mm]": y,
            "Z [mm]": z,
            "Material": patient_cell_material,
            "Dose": filled_points[index]["dose"],
            "FieldScalingFactor": calculate_fsf(x, y, z, cell_no_x, cell_no_y, cell_no_z, jaws, mlc_positions)
          })
          end_fsf_calc = perf_counter()
          analysis["fsf_calc_count"] = analysis.get("fsf_calc_count", 0) + 1
          analysis["fsf_calc_time"] = analysis.get("fsf_calc_time", 0) + (end_fsf_calc - start_fsf_calc)
          # for speedup, delete for multiprocessing code
          del filled_points[index]
        else:
          # coords do not match entry, "air" case
          raw_output.append({
            "Cell IdX": -1,
            "Cell IdY": -1,
            "Cell IdZ": -1,
            "Voxel IdX": -1,
            "Voxel IdY": -1,
            "Voxel IdZ": -1,
            "X [mm]": x,
            "Y [mm]": y,
            "Z [mm]": z,
            "Material": air,
            "Dose": 0,
            "FieldScalingFactor": 0
          })
  end_big_loop = perf_counter()
  analysis["big_loop_time"] = end_big_loop - start_big_loop
  output = pd.DataFrame(raw_output)
  # renormalize 
  start_renorm = perf_counter()
  renormalized_output = renormalize_fsf(output)
  end_renorm = perf_counter()
  analysis['fsf_renormalization_time'] = end_renorm - start_renorm
  print(analysis)
  return renormalized_output

## FSF ##

max_fsf_scaling = 0.98
min_fsf_scaling = 0.02

def calculate_fsf(x: np.float64, y: np.float64, z: np.float64, nx: int, ny: int, nz: int, jaws: np.ndarray, mlc_positions: np.ndarray) -> np.float64:
  """
  Assumptions: 
  - MLC_leaf positions are calculated as (x_leaf, y_leaf+leaf_height/2, mlc_detector_distance) V
  - angles are calculated in radians (0, np.pi)
  - y1MLC == y2MLC (???) ?
  """
  *_, Y1, Y2 = jaws
  n_leaves = len(mlc_positions)

  leaf_height = 2.5 # Fixed for simplified model
  mlc_detector_distance = -626.25 # In mm, above isocentre
  mlc_centre = np.array((0, 0, mlc_detector_distance))
  r_mlc = mlc_centre - np.array((x, y, z))

  y1MLC = 0
  y2MLC = 0

  current_x = -n_leaves*leaf_height/2 + (leaf_height/2)
  for y_left, y_right in mlc_positions:
    
    # WARNING: swapped X and Y dimensions to meet expected criteria
    r_leaf_left =  (mlc_centre + np.array((current_x,  y_left, 0))) - np.array((x, y, z))
    r_leaf_right = (mlc_centre + np.array((current_x, y_right, 0))) - np.array((x, y, z))

    lambda_leaf_left = angle_between(r_mlc, r_leaf_left)
    lambda_leaf_right = angle_between(r_mlc, r_leaf_right)

    y1MLC += lambda_leaf_left*lambda_leaf_left
    y2MLC += lambda_leaf_right*lambda_leaf_right

    current_x += leaf_height
  fsf = y1MLC + y2MLC
  fsf_norm = fsf/(1*nx + 2*ny + 3*nz)
  return fsf_norm

def renormalize_fsf(dataframe):
  print("renormalizing fsf...")
  max_fsf = dataframe['FieldScalingFactor'].max()
  min_fsf = dataframe[dataframe['FieldScalingFactor'] != 0]['FieldScalingFactor'].min()
  dataframe['FieldScalingFactor'] = (dataframe[dataframe['FieldScalingFactor'] != 0]['FieldScalingFactor'] - min_fsf) / (max_fsf-min_fsf) * (max_fsf_scaling-min_fsf_scaling) + min_fsf_scaling
  return dataframe

def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.float64:
  """ Returns the angle in radians between vectors v1 and v2"""
  v1_len = np.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
  v2_len = np.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
  return np.arccos(np.dot(v1, v2)/(v1_len*v2_len))

## MAIN ##

if __name__ == "__main__":
  target_resolution = 1 # mm
  cell_size = 10 #mm

  # data_filename = "cp-0_d3ddetector_cell.csv"
  data_filename = "data/new/prostate_imrt_beam0_cp74_d3ddetector_cell.csv"
  raw_df = pd.read_csv(data_filename)
  print(f"raw data read from {data_filename}...")

  # mlc_filename = "1. prostate_imrt_beam0_cp0.dat"
  mlc_filename = "data/new/prostate_imrt_beam0_cp74.dat"
  jaws, mlc_positions, _ = parse_mlc_data(mlc_filename)
  print(f"mlc data parsed, jaws: {jaws}, mlc_positions number: {len(mlc_positions)}")

  sorted_df = raw_df.sort_values(by=['X [mm]', 'Y [mm]', 'Z [mm]'])
  print("cell data sorted, generating output...")
  
  start = perf_counter()
  df = enrich_df(sorted_df, jaws, mlc_positions, target_resolution, cell_size, )
  end = perf_counter()
  print(f"output generated in {end-start:.2f} seconds.")

  df.to_csv("outputs/output.csv", index=False)

  # plotting
  if False:
    # for testing
    unique_xs_raw = np.array(sorted(raw_df['X [mm]'].unique()))
    unique_ys_raw = np.array(sorted(raw_df['Y [mm]'].unique()))
    unique_zs_raw = np.array(sorted(raw_df['Z [mm]'].unique()))

    start_plotting = perf_counter()
    plotting.plot_df(df, unique_xs_raw, unique_ys_raw, unique_zs_raw, target_resolution, output_filename="outputs/output.png")
    end_plotting = perf_counter()
    print(f"output plotted in {end_plotting-start_plotting:.4f} seconds.")

    start_plotting2 = perf_counter()
    plotting2.plot_df(df, unique_xs_raw, unique_ys_raw, unique_zs_raw, target_resolution, output_filename="outputs/output2.png")
    end_plotting2 = perf_counter()
    print(f"output plotted with plotting2 in {end_plotting2-start_plotting2:.4f} seconds.")
