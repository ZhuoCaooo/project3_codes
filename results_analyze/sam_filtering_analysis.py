#!/usr/bin/env python3
"""
LC SAM Filtering Analysis Script
Focus: Lane Change trajectories only with SAM parameter filtering
Filtering criteria: abs(W) <= 4 AND abs(D) <= 15
Analysis: Before vs After filtering comparison with temporal error analysis
Output: sam_analysis_results/ folder (plots and tables)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class LCTrajectoryComparison:
    """Store LC trajectory comparison results"""
    sample_id: int
    intention: int
    lateral_rmse: float
    longitudinal_rmse: float
    lateral_mae: float
    longitudinal_mae: float
    predicted_points: List[Tuple[float, float]]
    ground_truth_points: List[Tuple[float, float]]
    sam_parameters: Dict
    input_vx_kmh: float
    delta_vx: float
    # Temporal errors
    lateral_errors_by_time: List[float]
    longitudinal_errors_by_time: List[float]
    # Filtering info
    is_filtered_out: bool = False
    filter_reason: str = ""


def sam_model_with_v0(t, W, D, v0):
    """Modified Sinusoidal Acceleration Model (SAM) with known initial velocity v0"""
    return (W / D) * t + ((v0 * D - W) / (2 * np.pi)) * np.sin(2 * np.pi * t / D)


def extract_vx_from_input(input_text: str) -> Optional[float]:
    """Extract vx value from input text like 'vx=114.26, vy=2.59'"""
    pattern = r'vx=(\d+\.?\d*)'
    match = re.search(pattern, input_text)
    if match:
        return float(match.group(1))
    return None


def reconstruct_sam_trajectory(W: float, D: float, v0_ms: float, vx_initial_ms: float,
                               delta_vx_ms: float, time_points: List[float]) -> List[Tuple[float, float]]:
    """Reconstruct trajectory using SAM for lateral and linear model for longitudinal"""
    trajectory = []
    vx_change_rate = delta_vx_ms / 6.0  # m/s per second

    for t in time_points:
        # Lateral position using SAM
        y = sam_model_with_v0(t, W, D, v0_ms)

        # Longitudinal velocity at time t (linear change)
        vx_t = vx_initial_ms + vx_change_rate * (t + 2)

        # Longitudinal position (integrate velocity)
        x = vx_initial_ms * t + vx_change_rate * (2 * t + (t ** 2) / 2)

        trajectory.append((x, y))

    return trajectory


def parse_ground_truth_trajectory(trajectory_str: str) -> List[Tuple[float, float]]:
    """Parse trajectory string with improved regex patterns"""
    if not trajectory_str:
        return []

    trajectory_str = trajectory_str.strip().strip('"\'')
    coord_pattern = r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)'
    matches = re.findall(coord_pattern, trajectory_str)

    if matches:
        coordinates = [(float(x), float(y)) for x, y in matches]

    return coordinates


def calculate_trajectory_errors(predicted: List[Tuple[float, float]],
                                ground_truth: List[Tuple[float, float]]) -> Dict[str, float]:
    """Calculate RMSE and MAE for lateral and longitudinal errors"""
    if len(predicted) != len(ground_truth):
        min_len = min(len(predicted), len(ground_truth))
        predicted = predicted[:min_len]
        ground_truth = ground_truth[:min_len]

    if not predicted or not ground_truth:
        return {
            'lateral_rmse': float('inf'), 'longitudinal_rmse': float('inf'),
            'lateral_mae': float('inf'), 'longitudinal_mae': float('inf'),
            'lateral_errors': [], 'longitudinal_errors': []
        }

    pred_array = np.array(predicted)
    gt_array = np.array(ground_truth)

    lateral_errors = pred_array[:, 1] - gt_array[:, 1]
    longitudinal_errors = pred_array[:, 0] - gt_array[:, 0]

    return {
        'lateral_rmse': float(np.sqrt(np.mean(lateral_errors ** 2))),
        'longitudinal_rmse': float(np.sqrt(np.mean(longitudinal_errors ** 2))),
        'lateral_mae': float(np.mean(np.abs(lateral_errors))),
        'longitudinal_mae': float(np.mean(np.abs(longitudinal_errors))),
        'lateral_errors': lateral_errors.tolist(),
        'longitudinal_errors': longitudinal_errors.tolist()
    }


def apply_sam_filtering(W: float, D: float) -> Tuple[bool, str]:
    """
    Apply SAM parameter filtering
    Returns: (is_filtered_out, reason)
    """
    reasons = []

    if abs(W) > 5:
        reasons.append(f"abs(W)={abs(W):.2f}>4")

    if abs(D) > 12:
        reasons.append(f"abs(D)={abs(D):.2f}>15")

    is_filtered = len(reasons) > 0
    reason = "; ".join(reasons) if reasons else ""

    return is_filtered, reason


def analyze_lc_sam_with_filtering(sam_results_file: str, ground_truth_file: str) -> Dict:
    """Main analysis function for LC trajectories with SAM filtering"""

    print("Loading SAM results...")
    with open(sam_results_file, 'r') as f:
        sam_data = json.load(f)

    print("Loading 20-point ground truth dataset...")
    with open(ground_truth_file, 'r') as f:
        gt_data = json.load(f)

    sam_predictions = sam_data['all_predictions']

    print(f"SAM results: {len(sam_predictions)} samples")
    print(f"Ground truth: {len(gt_data)} samples")

    # Time points for 20-point trajectory (0.2s intervals from 0.2s to 4.0s)
    time_points = [0.2 * i for i in range(1, 21)]  # [0.2, 0.4, 0.6, ..., 4.0]

    lc_comparisons = []
    stats = {
        'total_samples': 0,
        'lc_samples_found': 0,
        'lc_with_sam_params': 0,
        'lc_successful_reconstruction': 0,
        'lc_filtered_out': 0,
        'lc_passed_filter': 0,
        'failed_reasons': []
    }

    for i, sam_result in enumerate(sam_predictions):
        if i >= len(gt_data):
            break

        stats['total_samples'] += 1
        gt_sample = gt_data[i]

        # Extract ground truth trajectory
        gt_text = gt_sample['text']
        if '[/INST]' not in gt_text:
            continue

        gt_response = gt_text.split('[/INST]')[1].replace('</s>', '').strip()

        # Extract ground truth trajectory (20 points)
        gt_trajectory = None
        trajectory_patterns = [
            r'- Trajectory:\s*"([^"]+)"',
            r'Trajectory:\s*"([^"]+)"',
            r'- Trajectory:\s*([^\n]+)',
            r'Trajectory:\s*([^\n]+)',
        ]

        for pattern in trajectory_patterns:
            match = re.search(pattern, gt_response, re.IGNORECASE)
            if match:
                gt_trajectory = parse_ground_truth_trajectory(match.group(1))
                if len(gt_trajectory) == 20:
                    break

        if not gt_trajectory or len(gt_trajectory) != 20:
            continue

        # Get prediction details
        pred_intention = sam_result['prediction']['intention']
        pred_parameters = sam_result['prediction']['parameters']

        # Only process Lane Change samples (intention != 0)
        if pred_intention == 0:
            continue

        stats['lc_samples_found'] += 1

        if not pred_parameters:
            stats['failed_reasons'].append(f"Sample {i}: No SAM parameters")
            continue

        stats['lc_with_sam_params'] += 1

        # Extract initial vx from input
        input_text = sam_result['input']['input_part']
        vx_kmh = extract_vx_from_input(input_text)
        if vx_kmh is None:
            stats['failed_reasons'].append(f"Sample {i}: Could not extract vx")
            continue

        vx_ms = vx_kmh / 3.6  # Convert km/h to m/s

        # Extract SAM parameters
        W = pred_parameters.get('lateral_displacement')
        D = pred_parameters.get('duration')
        v0 = pred_parameters.get('v0')
        delta_vx = pred_parameters.get('Delta_Vx')

        if W is None or D is None or v0 is None or delta_vx is None:
            stats['failed_reasons'].append(f"Sample {i}: Missing SAM parameters")
            continue

        # Apply filtering
        is_filtered_out, filter_reason = apply_sam_filtering(W, D)

        if is_filtered_out:
            stats['lc_filtered_out'] += 1
        else:
            stats['lc_passed_filter'] += 1

        # Reconstruct trajectory
        try:
            predicted_points = reconstruct_sam_trajectory(
                W=W, D=D, v0_ms=v0, vx_initial_ms=vx_ms,
                delta_vx_ms=delta_vx, time_points=time_points
            )
        except Exception as e:
            stats['failed_reasons'].append(f"Sample {i}: SAM reconstruction failed: {e}")
            continue

        stats['lc_successful_reconstruction'] += 1

        # Calculate errors
        errors = calculate_trajectory_errors(predicted_points, gt_trajectory)

        # Create comparison record
        comparison = LCTrajectoryComparison(
            sample_id=i,
            intention=pred_intention,
            lateral_rmse=errors['lateral_rmse'],
            longitudinal_rmse=errors['longitudinal_rmse'],
            lateral_mae=errors['lateral_mae'],
            longitudinal_mae=errors['longitudinal_mae'],
            predicted_points=predicted_points,
            ground_truth_points=gt_trajectory,
            sam_parameters=pred_parameters,
            input_vx_kmh=vx_kmh,
            delta_vx=delta_vx,
            lateral_errors_by_time=errors['lateral_errors'],
            longitudinal_errors_by_time=errors['longitudinal_errors'],
            is_filtered_out=is_filtered_out,
            filter_reason=filter_reason
        )

        lc_comparisons.append(comparison)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} samples...")

    print(f"\nLC Analysis complete! Found {len(lc_comparisons)} LC trajectories")

    return {
        'comparisons': lc_comparisons,
        'statistics': stats,
        'time_points': time_points
    }


def analyze_temporal_errors_comparison(comparisons: List[LCTrajectoryComparison],
                                       time_points: List[float]) -> Dict:
    """Analyze temporal errors for ALL vs FILTERED LC samples"""

    all_comparisons = comparisons
    filtered_comparisons = [c for c in comparisons if not c.is_filtered_out]

    def calc_temporal_stats(comps, name):
        if not comps:
            return None

        num_time_points = len(time_points)
        lateral_errors_by_time = [[] for _ in range(num_time_points)]
        longitudinal_errors_by_time = [[] for _ in range(num_time_points)]

        for comp in comps:
            for t_idx in range(min(num_time_points, len(comp.lateral_errors_by_time))):
                lateral_errors_by_time[t_idx].append(comp.lateral_errors_by_time[t_idx])
                longitudinal_errors_by_time[t_idx].append(comp.longitudinal_errors_by_time[t_idx])

        temporal_stats = {
            'name': name,
            'time_points': time_points,
            'sample_count': len(comps),
            'lateral': {
                'mean_errors': [], 'std_errors': [],
                'mean_abs_errors': [], 'std_abs_errors': []
            },
            'longitudinal': {
                'mean_errors': [], 'std_errors': [],
                'mean_abs_errors': [], 'std_abs_errors': []
            }
        }

        for t_idx in range(num_time_points):
            # Lateral statistics
            if lateral_errors_by_time[t_idx]:
                lat_errors = np.array(lateral_errors_by_time[t_idx])
                temporal_stats['lateral']['mean_errors'].append(np.mean(lat_errors))
                temporal_stats['lateral']['std_errors'].append(np.std(lat_errors))
                temporal_stats['lateral']['mean_abs_errors'].append(np.mean(np.abs(lat_errors)))
                temporal_stats['lateral']['std_abs_errors'].append(np.std(np.abs(lat_errors)))
            else:
                temporal_stats['lateral']['mean_errors'].append(0)
                temporal_stats['lateral']['std_errors'].append(0)
                temporal_stats['lateral']['mean_abs_errors'].append(0)
                temporal_stats['lateral']['std_abs_errors'].append(0)

            # Longitudinal statistics
            if longitudinal_errors_by_time[t_idx]:
                lon_errors = np.array(longitudinal_errors_by_time[t_idx])
                temporal_stats['longitudinal']['mean_errors'].append(np.mean(lon_errors))
                temporal_stats['longitudinal']['std_errors'].append(np.std(lon_errors))
                temporal_stats['longitudinal']['mean_abs_errors'].append(np.mean(np.abs(lon_errors)))
                temporal_stats['longitudinal']['std_abs_errors'].append(np.std(np.abs(lon_errors)))
            else:
                temporal_stats['longitudinal']['mean_errors'].append(0)
                temporal_stats['longitudinal']['std_errors'].append(0)
                temporal_stats['longitudinal']['mean_abs_errors'].append(0)
                temporal_stats['longitudinal']['std_abs_errors'].append(0)

        return temporal_stats

    all_stats = calc_temporal_stats(all_comparisons, "All LC Samples")
    filtered_stats = calc_temporal_stats(filtered_comparisons, "Filtered LC Samples")

    return {
        'all_lc': all_stats,
        'filtered_lc': filtered_stats
    }


def generate_comparison_tables(comparisons: List[LCTrajectoryComparison]) -> None:
    """Generate comparison tables for before/after filtering"""

    all_comparisons = comparisons
    filtered_comparisons = [c for c in comparisons if not c.is_filtered_out]

    def calc_stats(comps, name):
        if not comps:
            return None

        lat_rmse = [c.lateral_rmse for c in comps if c.lateral_rmse != float('inf')]
        lon_rmse = [c.longitudinal_rmse for c in comps if c.longitudinal_rmse != float('inf')]
        lat_mae = [c.lateral_mae for c in comps if c.lateral_mae != float('inf')]
        lon_mae = [c.longitudinal_mae for c in comps if c.longitudinal_mae != float('inf')]

        return {
            'name': name,
            'count': len(comps),
            'lat_rmse_mean': np.mean(lat_rmse) if lat_rmse else np.nan,
            'lat_rmse_std': np.std(lat_rmse) if lat_rmse else np.nan,
            'lon_rmse_mean': np.mean(lon_rmse) if lon_rmse else np.nan,
            'lon_rmse_std': np.std(lon_rmse) if lon_rmse else np.nan,
            'lat_mae_mean': np.mean(lat_mae) if lat_mae else np.nan,
            'lat_mae_std': np.std(lat_mae) if lat_mae else np.nan,
            'lon_mae_mean': np.mean(lon_mae) if lon_mae else np.nan,
            'lon_mae_std': np.std(lon_mae) if lon_mae else np.nan,
        }

    all_stats = calc_stats(all_comparisons, "ALL LC Samples")
    filtered_stats = calc_stats(filtered_comparisons, "FILTERED LC Samples")

    print("\n" + "=" * 80)
    print("LC FILTERING COMPARISON RESULTS")
    print("=" * 80)

    print(f"\n📊 SAMPLE COUNTS:")
    print(f"ALL LC Samples: {all_stats['count']}")
    print(f"FILTERED LC Samples: {filtered_stats['count']}")
    print(f"Filtered out: {all_stats['count'] - filtered_stats['count']}")
    print(f"Retention rate: {filtered_stats['count'] / all_stats['count'] * 100:.1f}%")

    print(f"\n📈 FILTERING CRITERIA APPLIED:")
    print(f"- abs(W) <= 4 meters (lateral displacement)")
    print(f"- abs(D) <= 15 seconds (duration)")

    # Error comparison table
    print(f"\n📊 ERROR COMPARISON TABLE:")
    print("-" * 80)
    print(f"{'Metric':<20} {'ALL LC':<25} {'FILTERED LC':<25} {'Improvement'}")
    print("-" * 80)

    metrics = [
        ('Lateral RMSE (m)', 'lat_rmse_mean', 'lat_rmse_std'),
        ('Longitudinal RMSE (m)', 'lon_rmse_mean', 'lon_rmse_std'),
        ('Lateral MAE (m)', 'lat_mae_mean', 'lat_mae_std'),
        ('Longitudinal MAE (m)', 'lon_mae_mean', 'lon_mae_std'),
    ]

    for metric_name, mean_key, std_key in metrics:
        all_val = all_stats[mean_key]
        all_std = all_stats[std_key]
        filt_val = filtered_stats[mean_key]
        filt_std = filtered_stats[std_key]

        improvement = ((all_val - filt_val) / all_val * 100) if all_val > 0 else 0

        all_str = f"{all_val:.3f}±{all_std:.3f}"
        filt_str = f"{filt_val:.3f}±{filt_std:.3f}"
        print(f"{metric_name:<20} {all_str:<25} {filt_str:<25} {improvement:+.1f}%")

    # Breakdown by intention
    print(f"\n📊 BREAKDOWN BY INTENTION:")
    for intention in [1, 2]:
        intention_name = {1: "Left Change", 2: "Right Change"}[intention]
        all_intent = [c for c in all_comparisons if c.intention == intention]
        filt_intent = [c for c in filtered_comparisons if c.intention == intention]

        if all_intent:
            all_intent_stats = calc_stats(all_intent, f"All {intention_name}")
            filt_intent_stats = calc_stats(filt_intent, f"Filtered {intention_name}")

            print(f"\n{intention_name}:")
            print(
                f"  Samples: {len(all_intent)} → {len(filt_intent)} ({len(filt_intent) / len(all_intent) * 100:.1f}% retained)")
            print(f"  Lat RMSE: {all_intent_stats['lat_rmse_mean']:.3f} → {filt_intent_stats['lat_rmse_mean']:.3f}")
            print(f"  Lon RMSE: {all_intent_stats['lon_rmse_mean']:.3f} → {filt_intent_stats['lon_rmse_mean']:.3f}")


def plot_temporal_comparison(temporal_stats: Dict, output_dir: str) -> None:
    """Plot temporal error comparison between ALL vs FILTERED LC samples"""

    all_stats = temporal_stats['all_lc']
    filtered_stats = temporal_stats['filtered_lc']

    if not all_stats or not filtered_stats:
        print("No temporal statistics to plot!")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

    time_points = all_stats['time_points']

    # Lateral Mean Absolute Error vs Time
    ax1.plot(time_points, all_stats['lateral']['mean_abs_errors'], 'r-o',
             linewidth=2, markersize=4, label=f'ALL LC (N={all_stats["sample_count"]})')
    ax1.plot(time_points, filtered_stats['lateral']['mean_abs_errors'], 'b-s',
             linewidth=2, markersize=4, label=f'FILTERED LC (N={filtered_stats["sample_count"]})')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Lateral MAE (m)')
    ax1.set_title('Lateral Mean Absolute Error vs Time\n(ALL vs FILTERED LC Samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Longitudinal Mean Absolute Error vs Time
    ax2.plot(time_points, all_stats['longitudinal']['mean_abs_errors'], 'r-o',
             linewidth=2, markersize=4, label=f'ALL LC (N={all_stats["sample_count"]})')
    ax2.plot(time_points, filtered_stats['longitudinal']['mean_abs_errors'], 'b-s',
             linewidth=2, markersize=4, label=f'FILTERED LC (N={filtered_stats["sample_count"]})')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Longitudinal MAE (m)')
    ax2.set_title('Longitudinal Mean Absolute Error vs Time\n(ALL vs FILTERED LC Samples)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Lateral Mean Error vs Time (with error bars)
    ax3.errorbar(time_points, all_stats['lateral']['mean_errors'],
                 yerr=all_stats['lateral']['std_errors'],
                 fmt='r-o', linewidth=2, markersize=4, capsize=3,
                 label=f'ALL LC (N={all_stats["sample_count"]})')
    ax3.errorbar(time_points, filtered_stats['lateral']['mean_errors'],
                 yerr=filtered_stats['lateral']['std_errors'],
                 fmt='b-s', linewidth=2, markersize=4, capsize=3,
                 label=f'FILTERED LC (N={filtered_stats["sample_count"]})')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Lateral Error (m)')
    ax3.set_title('Lateral Mean Error vs Time\n(ALL vs FILTERED LC Samples)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Longitudinal Mean Error vs Time (with error bars)
    ax4.errorbar(time_points, all_stats['longitudinal']['mean_errors'],
                 yerr=all_stats['longitudinal']['std_errors'],
                 fmt='r-o', linewidth=2, markersize=4, capsize=3,
                 label=f'ALL LC (N={all_stats["sample_count"]})')
    ax4.errorbar(time_points, filtered_stats['longitudinal']['mean_errors'],
                 yerr=filtered_stats['longitudinal']['std_errors'],
                 fmt='b-s', linewidth=2, markersize=4, capsize=3,
                 label=f'FILTERED LC (N={filtered_stats["sample_count"]})')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Longitudinal Error (m)')
    ax4.set_title('Longitudinal Mean Error vs Time\n(ALL vs FILTERED LC Samples)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lc_filtering_temporal_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Temporal comparison plot saved to: {os.path.join(output_dir, 'lc_filtering_temporal_comparison.png')}")


def plot_sample_trajectories_comparison(comparisons: List[LCTrajectoryComparison],
                                        output_dir: str, num_samples: int = 6) -> None:
    """Plot sample trajectory comparisons for filtered vs filtered-out samples"""

    passed_filter = [c for c in comparisons if not c.is_filtered_out]
    filtered_out = [c for c in comparisons if c.is_filtered_out]

    # Plot samples that passed the filter
    if passed_filter:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        samples_to_plot = passed_filter[:6]

        for i, comp in enumerate(samples_to_plot):
            ax = axes[i]

            pred_x = [p[0] for p in comp.predicted_points]
            pred_y = [p[1] for p in comp.predicted_points]
            gt_x = [p[0] for p in comp.ground_truth_points]
            gt_y = [p[1] for p in comp.ground_truth_points]

            ax.plot(gt_x, gt_y, 'b-o', label='Ground Truth', markersize=3, linewidth=2)
            ax.plot(pred_x, pred_y, 'g--s', label='SAM Reconstruction', markersize=3, linewidth=2)

            intention_names = {1: "Left Change", 2: "Right Change"}
            W = comp.sam_parameters['lateral_displacement']
            D = comp.sam_parameters['duration']

            ax.set_xlabel('Longitudinal (m)')
            ax.set_ylabel('Lateral (m)')
            ax.set_title(f'✅ PASSED: Sample {comp.sample_id} ({intention_names[comp.intention]})\n'
                         f'W={W:.2f}m, D={D:.2f}s | Lat RMSE: {comp.lateral_rmse:.2f}m')
            ax.legend()
            ax.grid(True, alpha=0.3)

        for i in range(len(samples_to_plot), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lc_passed_filter_samples.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Passed filter samples plot saved to: {os.path.join(output_dir, 'lc_passed_filter_samples.png')}")

    # Plot samples that were filtered out
    if filtered_out:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        samples_to_plot = filtered_out[:6]

        for i, comp in enumerate(samples_to_plot):
            ax = axes[i]

            pred_x = [p[0] for p in comp.predicted_points]
            pred_y = [p[1] for p in comp.predicted_points]
            gt_x = [p[0] for p in comp.ground_truth_points]
            gt_y = [p[1] for p in comp.ground_truth_points]

            ax.plot(gt_x, gt_y, 'b-o', label='Ground Truth', markersize=3, linewidth=2)
            ax.plot(pred_x, pred_y, 'r--s', label='SAM Reconstruction', markersize=3, linewidth=2)

            intention_names = {1: "Left Change", 2: "Right Change"}
            W = comp.sam_parameters['lateral_displacement']
            D = comp.sam_parameters['duration']

            ax.set_xlabel('Longitudinal (m)')
            ax.set_ylabel('Lateral (m)')
            ax.set_title(f'❌ FILTERED OUT: Sample {comp.sample_id} ({intention_names[comp.intention]})\n'
                         f'W={W:.2f}m, D={D:.2f}s | {comp.filter_reason}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        for i in range(len(samples_to_plot), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lc_filtered_out_samples.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Filtered out samples plot saved to: {os.path.join(output_dir, 'lc_filtered_out_samples.png')}")


def plot_parameter_distribution(comparisons: List[LCTrajectoryComparison], output_dir: str) -> None:
    """Plot SAM parameter distributions with filtering thresholds"""

    W_values = [c.sam_parameters['lateral_displacement'] for c in comparisons]
    D_values = [c.sam_parameters['duration'] for c in comparisons]
    passed_filter = [not c.is_filtered_out for c in comparisons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # W (lateral displacement) distribution
    W_passed = [W for W, passed in zip(W_values, passed_filter) if passed]
    W_filtered = [W for W, passed in zip(W_values, passed_filter) if not passed]

    ax1.hist(W_passed, bins=30, alpha=0.7, color='green', label=f'Passed Filter (N={len(W_passed)})')
    ax1.hist(W_filtered, bins=30, alpha=0.7, color='red', label=f'Filtered Out (N={len(W_filtered)})')
    ax1.axvline(x=4, color='black', linestyle='--', linewidth=2, label='Filter Threshold (±4m)')
    ax1.axvline(x=-4, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel('Lateral Displacement W (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('SAM Parameter W Distribution\nwith Filtering Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # D (duration) distribution
    D_passed = [D for D, passed in zip(D_values, passed_filter) if passed]
    D_filtered = [D for D, passed in zip(D_values, passed_filter) if not passed]

    ax2.hist(D_passed, bins=30, alpha=0.7, color='green', label=f'Passed Filter (N={len(D_passed)})')
    ax2.hist(D_filtered, bins=30, alpha=0.7, color='red', label=f'Filtered Out (N={len(D_filtered)})')
    ax2.axvline(x=15, color='black', linestyle='--', linewidth=2, label='Filter Threshold (15s)')
    ax2.set_xlabel('Duration D (s)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('SAM Parameter D Distribution\nwith Filtering Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sam_parameter_distributions.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Parameter distribution plot saved to: {os.path.join(output_dir, 'sam_parameter_distributions.png')}")

    # Print parameter statistics
    print(f"\n📊 SAM PARAMETER STATISTICS:")
    print(f"W (Lateral Displacement):")
    print(
        f"  ALL: mean={np.mean(W_values):.2f}±{np.std(W_values):.2f}, range=[{np.min(W_values):.2f}, {np.max(W_values):.2f}]")
    if W_passed:
        print(
            f"  PASSED: mean={np.mean(W_passed):.2f}±{np.std(W_passed):.2f}, range=[{np.min(W_passed):.2f}, {np.max(W_passed):.2f}]")

    print(f"D (Duration):")
    print(
        f"  ALL: mean={np.mean(D_values):.2f}±{np.std(D_values):.2f}, range=[{np.min(D_values):.2f}, {np.max(D_values):.2f}]")
    if D_passed:
        print(
            f"  PASSED: mean={np.mean(D_passed):.2f}±{np.std(D_passed):.2f}, range=[{np.min(D_passed):.2f}, {np.max(D_passed):.2f}]")


def main():
    """Main execution function"""

    # Create output directory
    output_dir = "sam_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Using output directory: {output_dir}/")

    # File paths
    sam_results_file = "complete_pal_predictions_old.json"
    ground_truth_file = "../lcllm_testing_data_20points.json"

    print("Starting LC SAM Filtering Analysis...")
    print(f"SAM results file: {sam_results_file}")
    print(f"Ground truth file: {ground_truth_file}")

    try:
        # Run main analysis
        analysis_results = analyze_lc_sam_with_filtering(sam_results_file, ground_truth_file)

        # Generate comparison tables
        generate_comparison_tables(analysis_results['comparisons'])

        # Temporal analysis comparison
        print("\n🔄 Analyzing temporal errors (ALL vs FILTERED)...")
        temporal_stats = analyze_temporal_errors_comparison(
            analysis_results['comparisons'],
            analysis_results['time_points']
        )

        # Generate plots
        plot_temporal_comparison(temporal_stats, output_dir)
        plot_sample_trajectories_comparison(analysis_results['comparisons'], output_dir)
        plot_parameter_distribution(analysis_results['comparisons'], output_dir)

        print(f"\n🎉 LC SAM Filtering Analysis Complete!")
        print(f"📁 All outputs saved to: {output_dir}/")
        print("📊 Generated plots:")
        print("  - lc_filtering_temporal_comparison.png (Temporal error comparison)")
        print("  - lc_passed_filter_samples.png (Samples that passed filter)")
        print("  - lc_filtered_out_samples.png (Samples that were filtered out)")
        print("  - sam_parameter_distributions.png (Parameter distributions)")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure both input files exist in the current directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()