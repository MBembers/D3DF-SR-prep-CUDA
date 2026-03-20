import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors


phantom_counts_cmap = cm.magma


def _build_edges(centers: np.ndarray, voxel_side_len: float) -> np.ndarray:
    """Build monotonic voxel edge coordinates from voxel center coordinates."""
    if centers.size == 0:
        raise ValueError("No voxel centers provided.")

    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)

    if centers.size == 1:
        half = voxel_side_len / 2.0
        edges[0] = centers[0] - half
        edges[1] = centers[0] + half
        return edges

    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - voxel_side_len / 2.0
    edges[-1] = centers[-1] + voxel_side_len / 2.0
    return edges


def _to_index(centers: np.ndarray, values: np.ndarray, axis_name: str) -> np.ndarray:
    """Map center coordinates to integer grid indices."""
    idx = np.searchsorted(centers, values)
    in_range = (idx >= 0) & (idx < centers.size)
    if not np.all(in_range):
        raise ValueError(f"Found {axis_name} values outside available grid range.")

    if not np.all(np.isclose(centers[idx], values)):
        raise ValueError(
            f"Some {axis_name} coordinates do not match known grid centers. "
            "Check coordinate precision or source dataframe consistency."
        )
    return idx


def plot_df(
    df: pd.DataFrame,
    ticks_x: np.ndarray,
    ticks_y: np.ndarray,
    ticks_z: np.ndarray,
    target_resolution: int,
    output_filename: str = None,
) -> None:
    observable = "FieldScalingFactor"
    x_col = "X [mm]"
    y_col = "Y [mm]"
    z_col = "Z [mm]"

    required_cols = [x_col, y_col, z_col, observable]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_nonzero = df.loc[df[observable] > 0, required_cols]
    if df_nonzero.empty:
        raise ValueError(f"No non-zero rows found for '{observable}'.")

    print(f"preparing voxel grid for {len(df_nonzero)} non-zero voxels...")

    # Use full axis centers from the dataframe so sparse voxels keep correct positions.
    x_centers = np.sort(df[x_col].unique())
    y_centers = np.sort(df[y_col].unique())
    z_centers = np.sort(df[z_col].unique())

    x_edges = _build_edges(x_centers, target_resolution)
    y_edges = _build_edges(y_centers, target_resolution)
    z_edges = _build_edges(z_centers, target_resolution)

    x_vals = df_nonzero[x_col].to_numpy(dtype=float)
    y_vals = df_nonzero[y_col].to_numpy(dtype=float)
    z_vals = df_nonzero[z_col].to_numpy(dtype=float)
    doses = df_nonzero[observable].to_numpy(dtype=float)

    x_idx = _to_index(x_centers, x_vals, "X")
    y_idx = _to_index(y_centers, y_vals, "Y")
    z_idx = _to_index(z_centers, z_vals, "Z")

    nx, ny, nz = x_centers.size, y_centers.size, z_centers.size
    filled = np.zeros((nx, ny, nz), dtype=bool)
    facecolors = np.zeros((nx, ny, nz, 4), dtype=float)

    dose_min = float(doses.min())
    dose_max = float(doses.max())
    phantom_counts_norm = colors.Normalize(vmin=dose_min, vmax=dose_max)
    voxel_colors = phantom_counts_cmap(phantom_counts_norm(doses), alpha=0.9)

    filled[x_idx, y_idx, z_idx] = True
    facecolors[x_idx, y_idx, z_idx] = voxel_colors

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")

    print("rendering voxels in a single batched draw call...")
    xx, yy, zz = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")
    ax.voxels(xx, yy, zz, filled, facecolors=facecolors, edgecolor=None)

    x_min = float(df_nonzero[x_col].min() - target_resolution)
    x_max = float(df_nonzero[x_col].max() + target_resolution)
    y_min = float(df_nonzero[y_col].min() - target_resolution)
    y_max = float(df_nonzero[y_col].max() + target_resolution)
    z_min = float(df_nonzero[z_col].min() - target_resolution)
    z_max = float(df_nonzero[z_col].max() + target_resolution)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    ax.set_box_aspect([1.2 * x_scale, 1.2 * y_scale, 1.2 * z_scale])

    ax.set_xlabel("x [mm]", fontsize=16, labelpad=10)
    ax.set_ylabel("y [mm]", fontsize=16, labelpad=10)
    ax.set_zlabel("z [mm]", fontsize=16, labelpad=10)

    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)
    ax.set_zticks(ticks_z)
    ax.tick_params(labelsize=12)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    scalar_mappable = cm.ScalarMappable(
        norm=colors.Normalize(vmin=dose_min, vmax=dose_max),
        cmap=phantom_counts_cmap,
    )
    colorbar_axes = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    cbar = fig.colorbar(scalar_mappable, cax=colorbar_axes, shrink=1.0, fraction=0.1, pad=0)
    cbar.ax.tick_params(labelsize=20)

    if output_filename:
        plt.savefig(output_filename, dpi=300)

    print("done.")
