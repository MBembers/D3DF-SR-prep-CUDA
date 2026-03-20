import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

phantom_counts_cmap = cm.magma

import matplotlib.pyplot as plt
import numpy as np

## PLOTTING ##

def add_voxel(ax, x_center: float, y_center: float, z_center: float, voxel_side_len: float, dose: float, min: float, max: float) -> None:
  x = [x_center - voxel_side_len / 2, x_center + voxel_side_len / 2]
  y = [y_center - voxel_side_len / 2, y_center + voxel_side_len / 2]
  z = [z_center - voxel_side_len / 2, z_center + voxel_side_len / 2]
  xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

  phantom_counts_norm = colors.Normalize(vmin=min, vmax=max)
  pc_normalized = phantom_counts_norm(dose)
  pc_colored = np.empty((*pc_normalized.shape, 4))
  with np.nditer(pc_normalized, flags=['multi_index']) as it:
    for el in it:
      pc_colored[it.multi_index] = phantom_counts_cmap(el, alpha=(0.9))
  ax.voxels(xx, yy, zz, np.ones((1,1,1), dtype=bool), facecolors=pc_colored, edgecolor=None)

def plot_df(df: pd.DataFrame, ticks_x: np.ndarray, ticks_y: np.ndarray, ticks_z: np.ndarray, target_resolution: int, output_filename: str = None) -> None:
  observable = "FieldScalingFactor"

  fig = plt.figure(figsize=(16, 12))
  ax = fig.add_subplot(111, projection='3d')

  # filter out "air"
  df_nonzero_observable = df[df[observable] > 0]

  dose_max = df_nonzero_observable[observable].max()
  dose_min = df_nonzero_observable[observable].min()

  counter=1
  max_voxel=len(df_nonzero_observable)

  for _, row in df_nonzero_observable.iterrows():
    if row[observable] == 0:
      # skip air
      continue
    x_center = row['X [mm]']
    y_center = row['Y [mm]']
    z_center = row['Z [mm]']

    dose = row[observable]
    if(counter % 100 == 0):
      print(f"adding {counter}/{max_voxel} voxel...", end="\r", flush=True)
    counter+=1

    add_voxel(ax, x_center, y_center, z_center, target_resolution, dose, dose_min, dose_max)
  print(f"added {counter-1}/{max_voxel} voxels. ")
  print("rendering image...")
  # Ustawienie limitów osi
  x_min, x_max = df_nonzero_observable['X [mm]'].min() - target_resolution, df_nonzero_observable['X [mm]'].max() + target_resolution
  y_min, y_max = df_nonzero_observable['Y [mm]'].min() - target_resolution, df_nonzero_observable['Y [mm]'].max() + target_resolution
  z_min, z_max = df_nonzero_observable['Z [mm]'].min() - target_resolution, df_nonzero_observable['Z [mm]'].max() + target_resolution

  ax.set_xlim([x_min, x_max])
  ax.set_ylim([y_min, y_max])
  ax.set_zlim([z_min, z_max])

  x_scale = x_max - x_min
  y_scale = y_max - y_min
  z_scale = z_max - z_min
  ax.set_box_aspect([1.2*x_scale, 1.2*y_scale, 1.2*z_scale])
  ax.set_xlabel("x [mm]", fontsize=16,labelpad=10)
  ax.set_ylabel("y [mm]", fontsize=16,labelpad=10)
  ax.set_zlabel("z [mm]", fontsize=16,labelpad=10)
  ax.set_xticks(ticks_x)
  ax.set_yticks(ticks_y)
  ax.set_zticks(ticks_z)
  ax.tick_params(labelsize=12)
  plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
  scalar_mappable = cm.ScalarMappable(norm=colors.Normalize(vmin=dose_min, vmax=dose_max), cmap=phantom_counts_cmap)
  colorbar_axes = fig.add_axes([0.9, 0.1, 0.03, 0.8])  # Adjust the position as needed
  cbar = fig.colorbar(scalar_mappable, cax=colorbar_axes, shrink=1.0, fraction=0.1, pad=0)
  cbar.ax.tick_params(labelsize=20)
  if(output_filename):
    plt.savefig(output_filename, dpi=300)
  # plt.show()