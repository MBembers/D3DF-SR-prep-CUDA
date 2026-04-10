import pandas as pd
import polars as pl
import pyarrow as pa
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

phantom_counts_cmap = cm.magma

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def add_voxel(ax, x_center, y_center, z_center, voxel_side_len, dose, min, max):
        x = [x_center - voxel_side_len / 2, x_center + voxel_side_len / 2]
        y = [y_center - voxel_side_len / 2, y_center + voxel_side_len / 2]
        z = [z_center - voxel_side_len / 2, z_center + voxel_side_len / 2]
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # phantom_counts_norm = colors.LogNorm(vmin=min, vmax=max)
        phantom_counts_norm = colors.Normalize(vmin=min, vmax=max)
        pc_normalized = phantom_counts_norm(dose)
        pc_colored = np.empty((*pc_normalized.shape, 4))
        with np.nditer(pc_normalized, flags=['multi_index']) as it:
            for el in it:
                pc_colored[it.multi_index] = phantom_counts_cmap(el, alpha=(0.9))
        ax.voxels(xx, yy, zz, np.ones((1,1,1), dtype=bool), facecolors=pc_colored, edgecolor=None)

if __name__ == "__main__":
    
    # path = '/home/geant4/workspace/github/g4rt/output/srunet3d_4x4x2_10x10x10_22/sim/ct_dose_0/voxel'
    dtypes_polars = {
    'X [mm]': pl.Float64,
    'Y [mm]': pl.Float64,
    'Z [mm]': pl.Float64,
    'Material': pl.Utf8,
    'Dose [Gy]': pl.Float64,
    'FieldScalingFactor': pl.Float64,
    # Add more columns and their dtypes as needed
    }
    # Get a list of all CSV files in the directory
    all_files = glob.glob("*.csv")

    # Create an empty list to hold DataFrames
    df_list = []

    # Loop through all files and read them into a DataFrame, then append to the list
    for filename in all_files:
        df = pl.read_csv(filename, schema_overrides=dtypes_polars)
        df_list.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_polars_df = pl.concat(df_list)

    # Convert Polars DataFrame to Pandas DataFrame
    combined_pandas_df = combined_polars_df.to_pandas()
    cell_df = combined_pandas_df.sort_values(by=['X [mm]', 'Y [mm]', 'Z [mm]'])
    """
    cell_df_cp8 = pd.read_csv('cell_ct_cp1_job23.csv')
    cell_df_cp9 = pd.read_csv('cell_ct_cp10_job22.csv')
    
    # Subtract FieldScalingFactor of df1 from df2
    cell_df = cell_df_cp8.copy()
    cell_df['FieldScalingFactor'] = abs(cell_df_cp8['FieldScalingFactor'] - cell_df_cp9['FieldScalingFactor'])
    """
    # voxel_side_len = 10.4
    voxel_side_len = 0.985

    observable = "FieldScalingFactor"

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    print (cell_df.size)
    
    # cell_df = cell_df[cell_df['Z [mm]'] > -4]
    # cell_df = cell_df[cell_df['Z [mm]'] < 2]
    dose_max = cell_df[observable].max()
    dose_min = cell_df[cell_df[observable]>0][observable].min()
    
    # cell_df.to_csv('cell_ct_cp1_job23.csv', index=False)
    
    print (cell_df.size)
    
    for _, row in cell_df.iterrows():
        if row[observable] == 0:
            continue
        x_center = row['X [mm]']
        y_center = row['Y [mm]']
        z_center = row['Z [mm]']

        dose = row[observable]
        add_voxel(ax, x_center, y_center, z_center, voxel_side_len, dose, dose_min, dose_max)

    # Ustawienie limitów osi
    x_min, x_max = cell_df['X [mm]'].min() - voxel_side_len, cell_df['X [mm]'].max() + voxel_side_len
    y_min, y_max = cell_df['Y [mm]'].min() - voxel_side_len, cell_df['Y [mm]'].max() + voxel_side_len
    z_min, z_max = cell_df['Z [mm]'].min() - voxel_side_len, cell_df['Z [mm]'].max() + voxel_side_len
    
    print(x_min, x_max, y_min, y_max, z_min, z_max)

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
    ax.tick_params(labelsize=12)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # scalar_mappable = cm.ScalarMappable(norm=colors.LogNorm(vmin=dose_min/100, vmax=dose_max), cmap=phantom_counts_cmap)
    scalar_mappable = cm.ScalarMappable(norm=colors.Normalize(vmin=dose_min, vmax=dose_max), cmap=phantom_counts_cmap)
    colorbar_axes = fig.add_axes([0.9, 0.1, 0.03, 0.8])  # Adjust the position as needed
    cbar = fig.colorbar(scalar_mappable, cax=colorbar_axes, shrink=1.0, fraction=0.1, pad=0)
    cbar.ax.tick_params(labelsize=20)
    plt.show()



