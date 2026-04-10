"""Verify generated voxel output against simulation voxel CSV.

Compares only:
- X [mm], Y [mm], Z [mm]
- Material
- FieldScalingFactor

Dose columns are intentionally ignored.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VerificationResult:
  ok: bool
  missing_coords: int
  extra_coords: int
  material_mismatches: int
  fsf_failures: int


def _add_coord_key(df: pd.DataFrame, precision: int) -> pd.DataFrame:
  scale = 10 ** precision
  out = df.copy()
  out['_xk'] = np.rint(out['X [mm]'].to_numpy(dtype=np.float64) * scale).astype(np.int64)
  out['_yk'] = np.rint(out['Y [mm]'].to_numpy(dtype=np.float64) * scale).astype(np.int64)
  out['_zk'] = np.rint(out['Z [mm]'].to_numpy(dtype=np.float64) * scale).astype(np.int64)
  return out


def _print_examples(df: pd.DataFrame, cols: list[str], n: int, label: str) -> None:
  if df.empty:
    return
  print(f"\n{label} (showing up to {n}):")
  print(df[cols].head(n).to_string(index=False))


def verify(
  output_csv: str,
  simulation_csv: str,
  fsf_abs_tol: float,
  material_abs_tol: float,
  coord_precision: int,
  shift_x: float,
  shift_y: float,
  shift_z: float,
  show_examples: int,
) -> VerificationResult:
  out_cols = ['X [mm]', 'Y [mm]', 'Z [mm]', 'Material', 'FieldScalingFactor']
  sim_cols = ['X [mm]', 'Y [mm]', 'Z [mm]', 'Material', 'FieldScalingFactor']

  output_df = pd.read_csv(output_csv, usecols=out_cols)
  sim_df = pd.read_csv(simulation_csv, usecols=sim_cols)

  if shift_x != 0.0 or shift_y != 0.0 or shift_z != 0.0:
    output_df['X [mm]'] = output_df['X [mm]'] + shift_x
    output_df['Y [mm]'] = output_df['Y [mm]'] + shift_y
    output_df['Z [mm]'] = output_df['Z [mm]'] + shift_z

  output_df = _add_coord_key(output_df, coord_precision)
  sim_df = _add_coord_key(sim_df, coord_precision)

  output_coords = output_df[['_xk', '_yk', '_zk']].drop_duplicates()
  sim_coords = sim_df[['_xk', '_yk', '_zk']].drop_duplicates()

  missing_coords_df = sim_coords.merge(
    output_coords,
    how='left',
    on=['_xk', '_yk', '_zk'],
    indicator=True,
  )
  missing_coords_df = missing_coords_df[missing_coords_df['_merge'] == 'left_only']

  extra_coords_df = output_coords.merge(
    sim_coords,
    how='left',
    on=['_xk', '_yk', '_zk'],
    indicator=True,
  )
  extra_coords_df = extra_coords_df[extra_coords_df['_merge'] == 'left_only']

  merged = output_df.merge(
    sim_df,
    how='inner',
    on=['_xk', '_yk', '_zk'],
    suffixes=('_out', '_sim'),
  )

  material_diff = np.abs(
    merged['Material_out'].to_numpy(dtype=np.float64) -
    merged['Material_sim'].to_numpy(dtype=np.float64)
  )
  material_bad_mask = material_diff > material_abs_tol

  fsf_diff = np.abs(
    merged['FieldScalingFactor_out'].to_numpy(dtype=np.float64) -
    merged['FieldScalingFactor_sim'].to_numpy(dtype=np.float64)
  )
  fsf_bad_mask = fsf_diff > fsf_abs_tol

  material_mismatches_df = merged.loc[material_bad_mask, [
    'X [mm]_out', 'Y [mm]_out', 'Z [mm]_out', 'Material_out', 'Material_sim'
  ]]

  fsf_mismatches_df = merged.loc[fsf_bad_mask, [
    'X [mm]_out', 'Y [mm]_out', 'Z [mm]_out', 'FieldScalingFactor_out', 'FieldScalingFactor_sim'
  ]]

  print('=== Verification Summary ===')
  print(f'Output rows:        {len(output_df)}')
  print(f'Simulation rows:    {len(sim_df)}')
  print(f'Matched rows:       {len(merged)}')
  print(f'Missing coordinates in output: {len(missing_coords_df)}')
  print(f'Extra coordinates in output:   {len(extra_coords_df)}')
  print(f'Material mismatches (> {material_abs_tol:g}): {material_bad_mask.sum()}')
  print(f'FSF mismatches (> {fsf_abs_tol:g}):          {fsf_bad_mask.sum()}')

  if len(merged) > 0:
    print('\n=== FSF Difference Stats (matched rows) ===')
    print(f'mean abs diff: {float(fsf_diff.mean()):.8g}')
    print(f'max abs diff:  {float(fsf_diff.max()):.8g}')

  out_min = output_df[['X [mm]', 'Y [mm]', 'Z [mm]']].min()
  out_max = output_df[['X [mm]', 'Y [mm]', 'Z [mm]']].max()
  sim_min = sim_df[['X [mm]', 'Y [mm]', 'Z [mm]']].min()
  sim_max = sim_df[['X [mm]', 'Y [mm]', 'Z [mm]']].max()

  print('\n=== Coordinate Ranges ===')
  print(
    'Output min/max: '
    f"X({out_min['X [mm]']}, {out_max['X [mm]']}), "
    f"Y({out_min['Y [mm]']}, {out_max['Y [mm]']}), "
    f"Z({out_min['Z [mm]']}, {out_max['Z [mm]']})"
  )
  print(
    'Simulation min/max: '
    f"X({sim_min['X [mm]']}, {sim_max['X [mm]']}), "
    f"Y({sim_min['Y [mm]']}, {sim_max['Y [mm]']}), "
    f"Z({sim_min['Z [mm]']}, {sim_max['Z [mm]']})"
  )
  print(
    'Range deltas (sim_min - out_min): '
    f"dx={sim_min['X [mm]'] - out_min['X [mm]']}, "
    f"dy={sim_min['Y [mm]'] - out_min['Y [mm]']}, "
    f"dz={sim_min['Z [mm]'] - out_min['Z [mm]']}"
  )

  _print_examples(
    missing_coords_df,
    ['_xk', '_yk', '_zk'],
    show_examples,
    'Missing coordinates in output (key units)'
  )
  _print_examples(
    extra_coords_df,
    ['_xk', '_yk', '_zk'],
    show_examples,
    'Extra coordinates in output (key units)'
  )
  _print_examples(
    material_mismatches_df,
    ['X [mm]_out', 'Y [mm]_out', 'Z [mm]_out', 'Material_out', 'Material_sim'],
    show_examples,
    'Material mismatches'
  )
  _print_examples(
    fsf_mismatches_df,
    ['X [mm]_out', 'Y [mm]_out', 'Z [mm]_out', 'FieldScalingFactor_out', 'FieldScalingFactor_sim'],
    show_examples,
    'FSF mismatches'
  )

  ok = (
    len(missing_coords_df) == 0 and
    len(extra_coords_df) == 0 and
    int(material_bad_mask.sum()) == 0 and
    int(fsf_bad_mask.sum()) == 0
  )

  return VerificationResult(
    ok=ok,
    missing_coords=int(len(missing_coords_df)),
    extra_coords=int(len(extra_coords_df)),
    material_mismatches=int(material_bad_mask.sum()),
    fsf_failures=int(fsf_bad_mask.sum()),
  )


def main() -> int:
  parser = argparse.ArgumentParser(description='Verify voxel output CSV against simulation CSV.')
  parser.add_argument(
    '--output-csv',
    default='outputs/output_numba.csv',
    help='Path to generated output CSV file.'
  )
  parser.add_argument(
    '--simulation-csv',
    default='data/new/prostate_imrt_beam0_cp74_ct_dose_voxel.csv',
    help='Path to simulation CSV file.'
  )
  parser.add_argument(
    '--fsf-abs-tol',
    type=float,
    default=1e-6,
    help='Absolute tolerance for FSF difference.'
  )
  parser.add_argument(
    '--material-abs-tol',
    type=float,
    default=1e-9,
    help='Absolute tolerance for Material difference.'
  )
  parser.add_argument(
    '--coord-precision',
    type=int,
    default=6,
    help='Decimal precision used to key coordinates.'
  )
  parser.add_argument('--shift-x', type=float, default=0.0, help='Global X shift applied to output coordinates before matching.')
  parser.add_argument('--shift-y', type=float, default=0.0, help='Global Y shift applied to output coordinates before matching.')
  parser.add_argument('--shift-z', type=float, default=0.0, help='Global Z shift applied to output coordinates before matching.')
  parser.add_argument('--show-examples', type=int, default=10, help='Number of mismatch examples to print per category.')

  args = parser.parse_args()

  result = verify(
    output_csv=args.output_csv,
    simulation_csv=args.simulation_csv,
    fsf_abs_tol=args.fsf_abs_tol,
    material_abs_tol=args.material_abs_tol,
    coord_precision=args.coord_precision,
    shift_x=args.shift_x,
    shift_y=args.shift_y,
    shift_z=args.shift_z,
    show_examples=args.show_examples,
  )

  if result.ok:
    print('\nPASS: output matches simulation for selected columns within tolerances.')
    return 0

  print('\nFAIL: mismatches found.')
  return 1


if __name__ == '__main__':
  sys.exit(main())
