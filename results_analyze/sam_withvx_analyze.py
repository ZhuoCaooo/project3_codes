#!/usr/bin/env python3
"""
REVISED SAM Results Analysis Script with Temporal Error Analysis
- LK samples: Direct comparison of 4 predicted points vs 20 GT points
- LC samples: Temporal analysis of SAM reconstruction errors at each time point
- Separate analysis for LK vs LC
- Output folder: sam_analysis_results/
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
class TrajectoryComparison:
    """Store trajectory comparison results"""
    sample_id: int
    intention: int
    prediction_type: str  # "trajectory" or "sam_reconstruction"
    lateral_rmse: float
    longitudinal_rmse: float
    lateral_mae: float
    longitudinal_mae: float
    predicted_points: List[Tuple[float, float]]
    ground_truth_points: List[Tuple[float, float]]
    sam_parameters: Optional[Dict] = None
    input_vx_kmh: Optional[float] = None
    delta_vx: Optional[float] = None
    # New: Store temporal errors
    lateral_errors_by_time: Optional[List[float]] = None
    longitudinal_errors_by_time: Optional[List[float]] = None


def sam_model_with_v0(t, W, D, v0):
    """
    Modified Sinusoidal Acceleration Model (SAM) with known initial velocity v0
    """
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
    """
    Reconstruct trajectory using SAM for lateral and linear model for longitudinal
    """
    trajectory = []

    # Delta_Vx is over 6 seconds [-2s, +4s], so velocity change rate per second
    vx_change_rate = delta_vx_ms / 6.0  # m/s per second

    for t in time_points:
        # Lateral position using SAM
        y = sam_model_with_v0(t, W, D, v0_ms)

        # Longitudinal velocity at time t (linear change)
        # Since Delta_Vx is from -2s to +4s, and we're predicting 0s to 4s
        # At t=0, we're already 2s into the 6s period
        vx_t = vx_initial_ms + vx_change_rate * (t + 2)

        # Longitudinal position (integrate velocity)
        x = vx_initial_ms * t + vx_change_rate * (2 * t + (t ** 2) / 2)

        trajectory.append((x, y))

    return trajectory


def parse_ground_truth_trajectory(trajectory_str: str) -> List[Tuple[float, float]]:
    """
    Parse trajectory string with improved regex patterns
    """
    if not trajectory_str:
        return []

    # Clean the string
    trajectory_str = trajectory_str.strip().strip('"\'')

    # Pattern to extract individual coordinate pairs
    coord_pattern = r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)'
    matches = re.findall(coord_pattern, trajectory_str)

    if matches:
        coordinates = [(float(x), float(y)) for x, y in matches]

    return coordinates


def parse_predicted_trajectory(prediction_dict: Dict) -> Optional[List[Tuple[float, float]]]:
    """
    Parse predicted trajectory from the prediction dictionary
    """
    if 'trajectory' in prediction_dict and prediction_dict['trajectory']:
        traj = prediction_dict['trajectory']
        if isinstance(traj, str):
            return parse_ground_truth_trajectory(traj)
        elif isinstance(traj, list):
            return traj
    return None


def calculate_trajectory_errors(predicted: List[Tuple[float, float]],
                                ground_truth: List[Tuple[float, float]]) -> Dict[str, float]:
    """Calculate RMSE and MAE for lateral and longitudinal errors"""
    if len(predicted) != len(ground_truth):
        min_len = min(len(predicted), len(ground_truth))
        predicted = predicted[:min_len]
        ground_truth = ground_truth[:min_len]

    if not predicted or not ground_truth:
        return {
            'lateral_rmse': float('inf'),
            'longitudinal_rmse': float('inf'),
            'lateral_mae': float('inf'),
            'longitudinal_mae': float('inf')
        }

    pred_array = np.array(predicted)
    gt_array = np.array(ground_truth)

    # Separate lateral (y) and longitudinal (x) errors
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


def interpolate_4_to_20_points(four_points: List[Tuple[float, float]],
                               time_points_20: List[float]) -> List[Tuple[float, float]]:
    """
    Interpolate 4 predicted points to 20 time points for LK comparison
    Assumes 4 points are at t=[1.0, 2.0, 3.0, 4.0] seconds
    """
    if len(four_points) != 4:
        return []

    # Time points for the 4 predictions
    time_4 = [1.0, 2.0, 3.0, 4.0]

    # Extract x and y coordinates
    x_coords = [p[0] for p in four_points]
    y_coords = [p[1] for p in four_points]

    # Interpolate to 20 time points
    x_interp = np.interp(time_points_20, time_4, x_coords)
    y_interp = np.interp(time_points_20, time_4, y_coords)

    return list(zip(x_interp, y_interp))


def analyze_sam_results(sam_results_file: str, ground_truth_file: str) -> Dict:
    """Main analysis function with temporal error tracking"""

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

    comparisons = []
    stats = {
        'total_samples': 0,
        'successful_comparisons': 0,
        'lane_keeping_count': 0,
        'lane_change_count': 0,
        'sam_reconstruction_count': 0,
        'trajectory_direct_count': 0,
        'lane_keeping_with_4_points': 0,
        'failed_comparisons': []
    }

    for i, sam_result in enumerate(sam_predictions):
        if i >= len(gt_data):
            print(f"Warning: SAM result {i} has no corresponding ground truth")
            break

        stats['total_samples'] += 1
        gt_sample = gt_data[i]

        # Extract ground truth trajectory
        gt_text = gt_sample['text']
        if '[/INST]' not in gt_text:
            stats['failed_comparisons'].append(f"Sample {i}: No [/INST] in ground truth")
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
            stats['failed_comparisons'].append(
                f"Sample {i}: Could not extract 20-point ground truth trajectory")
            continue

        # Get prediction details
        pred_intention = sam_result['prediction']['intention']
        pred_parameters = sam_result['prediction']['parameters']
        pred_trajectory = sam_result['prediction']['trajectory']

        if pred_intention is None:
            stats['failed_comparisons'].append(f"Sample {i}: No predicted intention")
            continue

        # Extract initial vx from input
        input_text = sam_result['input']['input_part']
        vx_kmh = extract_vx_from_input(input_text)
        if vx_kmh is None:
            stats['failed_comparisons'].append(f"Sample {i}: Could not extract vx from input")
            continue

        vx_ms = vx_kmh / 3.6  # Convert km/h to m/s

        predicted_points = None
        prediction_type = None
        sam_params = None
        delta_vx = None

        if pred_intention == 0:  # Lane keeping
            stats['lane_keeping_count'] += 1

            # For LK, use the 4 predicted trajectory points directly
            pred_4_points = parse_predicted_trajectory(sam_result['prediction'])
            if pred_4_points and len(pred_4_points) == 4:
                # Interpolate 4 points to 20 for comparison
                predicted_points = interpolate_4_to_20_points(pred_4_points, time_points)
                prediction_type = "lk_4_points_interpolated"
                stats['lane_keeping_with_4_points'] += 1
            else:
                stats['failed_comparisons'].append(f"Sample {i}: LK without 4 trajectory points")
                continue

        else:  # Lane change (intention 1 or 2)
            stats['lane_change_count'] += 1

            if pred_parameters:
                # Reconstruct using SAM parameters
                stats['sam_reconstruction_count'] += 1
                prediction_type = "sam_reconstruction"

                W = pred_parameters.get('lateral_displacement')
                D = pred_parameters.get('duration')
                v0 = pred_parameters.get('v0')
                delta_vx = pred_parameters.get('Delta_Vx')

                if W is None or D is None or v0 is None or delta_vx is None:
                    stats['failed_comparisons'].append(
                        f"Sample {i}: Missing SAM parameters")
                    continue

                sam_params = pred_parameters

                try:
                    predicted_points = reconstruct_sam_trajectory(
                        W=W, D=D, v0_ms=v0, vx_initial_ms=vx_ms,
                        delta_vx_ms=delta_vx, time_points=time_points
                    )
                except Exception as e:
                    stats['failed_comparisons'].append(f"Sample {i}: SAM reconstruction failed: {e}")
                    continue
            else:
                stats['failed_comparisons'].append(f"Sample {i}: Lane change without SAM parameters")
                continue

        if predicted_points is None:
            continue

        # Calculate errors with temporal information
        errors = calculate_trajectory_errors(predicted_points, gt_trajectory)

        # Create comparison record
        comparison = TrajectoryComparison(
            sample_id=i,
            intention=pred_intention,
            prediction_type=prediction_type,
            lateral_rmse=errors['lateral_rmse'],
            longitudinal_rmse=errors['longitudinal_rmse'],
            lateral_mae=errors['lateral_mae'],
            longitudinal_mae=errors['longitudinal_mae'],
            predicted_points=predicted_points,
            ground_truth_points=gt_trajectory,
            sam_parameters=sam_params,
            input_vx_kmh=vx_kmh,
            delta_vx=delta_vx,
            lateral_errors_by_time=errors['lateral_errors'],
            longitudinal_errors_by_time=errors['longitudinal_errors']
        )

        comparisons.append(comparison)
        stats['successful_comparisons'] += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} samples...")

    print(f"\nAnalysis complete! Processed {stats['successful_comparisons']}/{stats['total_samples']} samples")

    return {
        'comparisons': comparisons,
        'statistics': stats,
        'time_points': time_points
    }


def analyze_temporal_errors_lc(comparisons: List[TrajectoryComparison],
                               time_points: List[float]) -> Dict:
    """
    Analyze temporal errors for Lane Change samples only
    """
    lc_comparisons = [c for c in comparisons if c.intention != 0 and c.prediction_type == "sam_reconstruction"]

    if not lc_comparisons:
        return {}

    num_time_points = len(time_points)

    # Collect errors at each time point
    lateral_errors_by_time = [[] for _ in range(num_time_points)]
    longitudinal_errors_by_time = [[] for _ in range(num_time_points)]

    for comp in lc_comparisons:
        if comp.lateral_errors_by_time and comp.longitudinal_errors_by_time:
            for t_idx in range(min(num_time_points, len(comp.lateral_errors_by_time))):
                lateral_errors_by_time[t_idx].append(comp.lateral_errors_by_time[t_idx])
                longitudinal_errors_by_time[t_idx].append(comp.longitudinal_errors_by_time[t_idx])

    # Calculate statistics at each time point
    temporal_stats = {
        'time_points': time_points,
        'lateral': {
            'mean_errors': [],
            'std_errors': [],
            'mean_abs_errors': [],
            'std_abs_errors': []
        },
        'longitudinal': {
            'mean_errors': [],
            'std_errors': [],
            'mean_abs_errors': [],
            'std_abs_errors': []
        },
        'sample_count': len(lc_comparisons)
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


def analyze_lk_4_point_errors(comparisons: List[TrajectoryComparison]) -> Dict:
    """
    Analyze errors for Lane Keeping samples at the 4 prediction time points (1s, 2s, 3s, 4s)
    """
    lk_comparisons = [c for c in comparisons if c.intention == 0]

    if not lk_comparisons:
        return {}

    # Time indices for 1s, 2s, 3s, 4s (corresponding to indices 4, 9, 14, 19 in 20-point array)
    time_indices = [4, 9, 14, 19]  # 0.2*5=1.0, 0.2*10=2.0, 0.2*15=3.0, 0.2*20=4.0
    time_points = [1.0, 2.0, 3.0, 4.0]

    lk_stats = {
        'time_points': time_points,
        'lateral': {
            'mean_errors': [],
            'std_errors': [],
            'mean_abs_errors': [],
            'std_abs_errors': []
        },
        'longitudinal': {
            'mean_errors': [],
            'std_errors': [],
            'mean_abs_errors': [],
            'std_abs_errors': []
        },
        'sample_count': len(lk_comparisons)
    }

    for t_idx in time_indices:
        lateral_errors = []
        longitudinal_errors = []

        for comp in lk_comparisons:
            if (comp.lateral_errors_by_time and comp.longitudinal_errors_by_time and
                    len(comp.lateral_errors_by_time) > t_idx):
                lateral_errors.append(comp.lateral_errors_by_time[t_idx])
                longitudinal_errors.append(comp.longitudinal_errors_by_time[t_idx])

        if lateral_errors:
            lat_errors = np.array(lateral_errors)
            lk_stats['lateral']['mean_errors'].append(np.mean(lat_errors))
            lk_stats['lateral']['std_errors'].append(np.std(lat_errors))
            lk_stats['lateral']['mean_abs_errors'].append(np.mean(np.abs(lat_errors)))
            lk_stats['lateral']['std_abs_errors'].append(np.std(np.abs(lat_errors)))
        else:
            lk_stats['lateral']['mean_errors'].append(0)
            lk_stats['lateral']['std_errors'].append(0)
            lk_stats['lateral']['mean_abs_errors'].append(0)
            lk_stats['lateral']['std_abs_errors'].append(0)

        if longitudinal_errors:
            lon_errors = np.array(longitudinal_errors)
            lk_stats['longitudinal']['mean_errors'].append(np.mean(lon_errors))
            lk_stats['longitudinal']['std_errors'].append(np.std(lon_errors))
            lk_stats['longitudinal']['mean_abs_errors'].append(np.mean(np.abs(lon_errors)))
            lk_stats['longitudinal']['std_abs_errors'].append(np.std(np.abs(lon_errors)))
        else:
            lk_stats['longitudinal']['mean_errors'].append(0)
            lk_stats['longitudinal']['std_errors'].append(0)
            lk_stats['longitudinal']['mean_abs_errors'].append(0)
            lk_stats['longitudinal']['std_abs_errors'].append(0)

    return lk_stats


def plot_temporal_errors(temporal_stats: Dict, output_dir: str) -> None:
    """
    Plot temporal error analysis for Lane Change samples
    """
    if not temporal_stats:
        print("No temporal statistics to plot!")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    time_points = temporal_stats['time_points']

    # Lateral Mean Error vs Time
    ax1.plot(time_points, temporal_stats['lateral']['mean_errors'], 'b-o', linewidth=2, markersize=4)
    ax1.fill_between(time_points,
                     np.array(temporal_stats['lateral']['mean_errors']) - np.array(
                         temporal_stats['lateral']['std_errors']),
                     np.array(temporal_stats['lateral']['mean_errors']) + np.array(
                         temporal_stats['lateral']['std_errors']),
                     alpha=0.3, color='blue')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Lateral Error (m)')
    ax1.set_title(f'Lateral Mean Error vs Time\n(Lane Changes, N={temporal_stats["sample_count"]})')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Lateral Absolute Error vs Time
    ax2.plot(time_points, temporal_stats['lateral']['mean_abs_errors'], 'g-o', linewidth=2, markersize=4)
    ax2.fill_between(time_points,
                     np.array(temporal_stats['lateral']['mean_abs_errors']) - np.array(
                         temporal_stats['lateral']['std_abs_errors']),
                     np.array(temporal_stats['lateral']['mean_abs_errors']) + np.array(
                         temporal_stats['lateral']['std_abs_errors']),
                     alpha=0.3, color='green')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Lateral Absolute Error (m)')
    ax2.set_title(f'Lateral Mean Absolute Error vs Time\n(Lane Changes, N={temporal_stats["sample_count"]})')
    ax2.grid(True, alpha=0.3)

    # Longitudinal Mean Error vs Time
    ax3.plot(time_points, temporal_stats['longitudinal']['mean_errors'], 'r-o', linewidth=2, markersize=4)
    ax3.fill_between(time_points,
                     np.array(temporal_stats['longitudinal']['mean_errors']) - np.array(
                         temporal_stats['longitudinal']['std_errors']),
                     np.array(temporal_stats['longitudinal']['mean_errors']) + np.array(
                         temporal_stats['longitudinal']['std_errors']),
                     alpha=0.3, color='red')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Longitudinal Error (m)')
    ax3.set_title(f'Longitudinal Mean Error vs Time\n(Lane Changes, N={temporal_stats["sample_count"]})')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Longitudinal Absolute Error vs Time
    ax4.plot(time_points, temporal_stats['longitudinal']['mean_abs_errors'], 'm-o', linewidth=2, markersize=4)
    ax4.fill_between(time_points,
                     np.array(temporal_stats['longitudinal']['mean_abs_errors']) - np.array(
                         temporal_stats['longitudinal']['std_abs_errors']),
                     np.array(temporal_stats['longitudinal']['mean_abs_errors']) + np.array(
                         temporal_stats['longitudinal']['std_abs_errors']),
                     alpha=0.3, color='magenta')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Longitudinal Absolute Error (m)')
    ax4.set_title(f'Longitudinal Mean Absolute Error vs Time\n(Lane Changes, N={temporal_stats["sample_count"]})')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_error_analysis_lc.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Temporal error analysis plot saved to: {os.path.join(output_dir, 'temporal_error_analysis_lc.png')}")


def plot_lk_4_point_errors(lk_stats: Dict, output_dir: str) -> None:
    """
    Plot 4-point error analysis for Lane Keeping samples
    """
    if not lk_stats:
        print("No LK statistics to plot!")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    time_points = lk_stats['time_points']

    # Lateral Mean Error
    ax1.bar(time_points, lk_stats['lateral']['mean_errors'],
            yerr=lk_stats['lateral']['std_errors'], capsize=5, color='blue', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Lateral Error (m)')
    ax1.set_title(f'Lateral Mean Error at 4 Time Points\n(Lane Keeping, N={lk_stats["sample_count"]})')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Lateral Absolute Error
    ax2.bar(time_points, lk_stats['lateral']['mean_abs_errors'],
            yerr=lk_stats['lateral']['std_abs_errors'], capsize=5, color='green', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Lateral Absolute Error (m)')
    ax2.set_title(f'Lateral Mean Absolute Error at 4 Time Points\n(Lane Keeping, N={lk_stats["sample_count"]})')
    ax2.grid(True, alpha=0.3)

    # Longitudinal Mean Error
    ax3.bar(time_points, lk_stats['longitudinal']['mean_errors'],
            yerr=lk_stats['longitudinal']['std_errors'], capsize=5, color='red', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Longitudinal Error (m)')
    ax3.set_title(f'Longitudinal Mean Error at 4 Time Points\n(Lane Keeping, N={lk_stats["sample_count"]})')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Longitudinal Absolute Error
    ax4.bar(time_points, lk_stats['longitudinal']['mean_abs_errors'],
            yerr=lk_stats['longitudinal']['std_abs_errors'], capsize=5, color='magenta', alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Longitudinal Absolute Error (m)')
    ax4.set_title(f'Longitudinal Mean Absolute Error at 4 Time Points\n(Lane Keeping, N={lk_stats["sample_count"]})')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lk_4_point_error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 LK 4-point error analysis plot saved to: {os.path.join(output_dir, 'lk_4_point_error_analysis.png')}")


def plot_sample_trajectories(analysis_results: Dict, output_dir: str, num_samples: int = 6) -> None:
    """
    Plot sample trajectory comparisons - separate LK and LC samples
    """
    comparisons = analysis_results['comparisons']
    if not comparisons:
        print("No comparisons to plot!")
        return

    # Separate LC and LK samples
    lc_comparisons = [c for c in comparisons if c.intention != 0]
    lk_comparisons = [c for c in comparisons if c.intention == 0]

    # Plot LC samples (SAM reconstruction)
    if lc_comparisons:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        samples_to_plot = lc_comparisons[:6]

        for i, comp in enumerate(samples_to_plot):
            ax = axes[i]

            # Extract coordinates
            pred_x = [p[0] for p in comp.predicted_points]
            pred_y = [p[1] for p in comp.predicted_points]
            gt_x = [p[0] for p in comp.ground_truth_points]
            gt_y = [p[1] for p in comp.ground_truth_points]

            # Plot trajectories
            ax.plot(gt_x, gt_y, 'b-o', label='Ground Truth (20 pts)', markersize=3, linewidth=2)
            ax.plot(pred_x, pred_y, 'r--s', label='SAM Reconstruction', markersize=3, linewidth=2)

            intention_names = {1: "Left Change", 2: "Right Change"}
            ax.set_xlabel('Longitudinal (m)')
            ax.set_ylabel('Lateral (m)')
            ax.set_title(f'Sample {comp.sample_id}: {intention_names[comp.intention]}\n'
                         f'Lat RMSE: {comp.lateral_rmse:.2f}m, Lon RMSE: {comp.longitudinal_rmse:.2f}m')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(samples_to_plot), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lc_sam_trajectory_samples.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 LC SAM trajectory plots saved to: {os.path.join(output_dir, 'lc_sam_trajectory_samples.png')}")

    # Plot LK samples (4 predicted points vs 20 GT points)
    if lk_comparisons:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        samples_to_plot = lk_comparisons[:6]

        for i, comp in enumerate(samples_to_plot):
            ax = axes[i]

            # GT: 20 points
            gt_x = [p[0] for p in comp.ground_truth_points]
            gt_y = [p[1] for p in comp.ground_truth_points]

            # Predicted: interpolated 20 points from 4 original
            pred_x = [p[0] for p in comp.predicted_points]
            pred_y = [p[1] for p in comp.predicted_points]

            # Original 4 predicted points (at t=1,2,3,4s, indices 4,9,14,19)
            orig_indices = [4, 9, 14, 19]
            orig_pred_x = [pred_x[idx] for idx in orig_indices if idx < len(pred_x)]
            orig_pred_y = [pred_y[idx] for idx in orig_indices if idx < len(pred_y)]

            # Plot trajectories
            ax.plot(gt_x, gt_y, 'b-o', label='Ground Truth (20 pts)', markersize=3, linewidth=2)
            ax.plot(pred_x, pred_y, 'g--', label='Interpolated (4→20 pts)', linewidth=1, alpha=0.7)
            ax.plot(orig_pred_x, orig_pred_y, 'r^', label='Original 4 Predictions', markersize=8)

            ax.set_xlabel('Longitudinal (m)')
            ax.set_ylabel('Lateral (m)')
            ax.set_title(f'Sample {comp.sample_id}: Lane Keeping\n'
                         f'Lat RMSE: {comp.lateral_rmse:.2f}m, Lon RMSE: {comp.longitudinal_rmse:.2f}m')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(samples_to_plot), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lk_4point_trajectory_samples.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(
            f"📊 LK 4-point trajectory plots saved to: {os.path.join(output_dir, 'lk_4point_trajectory_samples.png')}")


def generate_analysis_report(analysis_results: Dict) -> None:
    """Generate comprehensive analysis report"""

    comparisons = analysis_results['comparisons']
    stats = analysis_results['statistics']

    print("\n" + "=" * 60)
    print("REVISED SAM RESULTS ANALYSIS REPORT")
    print("=" * 60)

    print(f"\n📊 OVERVIEW:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Successful comparisons: {stats['successful_comparisons']}")
    print(f"Failed comparisons: {len(stats['failed_comparisons'])}")
    print(f"Success rate: {stats['successful_comparisons'] / stats['total_samples'] * 100:.1f}%")

    print(f"\n🎯 PREDICTION TYPES:")
    print(f"Lane keeping: {stats['lane_keeping_count']} (4-point direct: {stats['lane_keeping_with_4_points']})")
    print(f"Lane change: {stats['lane_change_count']} (SAM reconstruction: {stats['sam_reconstruction_count']})")

    if not comparisons:
        print("\n❌ No successful comparisons to analyze!")
        return

    # Separate analysis for LK and LC
    lk_comparisons = [c for c in comparisons if c.intention == 0]
    lc_comparisons = [c for c in comparisons if c.intention != 0]

    print(f"\n🔄 LANE KEEPING ANALYSIS ({len(lk_comparisons)} samples):")
    if lk_comparisons:
        lk_lat_rmse = [c.lateral_rmse for c in lk_comparisons if c.lateral_rmse != float('inf')]
        lk_lon_rmse = [c.longitudinal_rmse for c in lk_comparisons if c.longitudinal_rmse != float('inf')]
        print(f"Lateral RMSE: {np.mean(lk_lat_rmse):.3f} ± {np.std(lk_lat_rmse):.3f} m")
        print(f"Longitudinal RMSE: {np.mean(lk_lon_rmse):.3f} ± {np.std(lk_lon_rmse):.3f} m")

    print(f"\n🔄 LANE CHANGE ANALYSIS ({len(lc_comparisons)} samples):")
    if lc_comparisons:
        lc_lat_rmse = [c.lateral_rmse for c in lc_comparisons if c.lateral_rmse != float('inf')]
        lc_lon_rmse = [c.longitudinal_rmse for c in lc_comparisons if c.longitudinal_rmse != float('inf')]
        print(f"Lateral RMSE: {np.mean(lc_lat_rmse):.3f} ± {np.std(lc_lat_rmse):.3f} m")
        print(f"Longitudinal RMSE: {np.mean(lc_lon_rmse):.3f} ± {np.std(lc_lon_rmse):.3f} m")

        # Breakdown by intention
        for intention in [1, 2]:
            intention_comps = [c for c in lc_comparisons if c.intention == intention]
            if intention_comps:
                intention_name = {1: "Left Change", 2: "Right Change"}[intention]
                lat_rmse = [c.lateral_rmse for c in intention_comps if c.lateral_rmse != float('inf')]
                lon_rmse = [c.longitudinal_rmse for c in intention_comps if c.longitudinal_rmse != float('inf')]
                print(f"  {intention_name}: {len(intention_comps)} samples")
                if lat_rmse:
                    print(f"    Lateral RMSE: {np.mean(lat_rmse):.3f} ± {np.std(lat_rmse):.3f} m")
                if lon_rmse:
                    print(f"    Longitudinal RMSE: {np.mean(lon_rmse):.3f} ± {np.std(lon_rmse):.3f} m")


def save_detailed_results(analysis_results: Dict, temporal_stats: Dict, lk_stats: Dict, output_dir: str) -> None:
    """Save all detailed results to JSON files"""

    # Main results
    serializable_results = {
        'metadata': {
            'analysis_type': 'REVISED_SAM_vs_GroundTruth_Temporal',
            'total_samples': analysis_results['statistics']['total_samples'],
            'successful_comparisons': analysis_results['statistics']['successful_comparisons'],
            'time_points': analysis_results['time_points']
        },
        'statistics': analysis_results['statistics'],
        'comparisons': []
    }

    for comp in analysis_results['comparisons']:
        comp_dict = {
            'sample_id': comp.sample_id,
            'intention': comp.intention,
            'prediction_type': comp.prediction_type,
            'errors': {
                'lateral_rmse': comp.lateral_rmse,
                'longitudinal_rmse': comp.longitudinal_rmse,
                'lateral_mae': comp.lateral_mae,
                'longitudinal_mae': comp.longitudinal_mae
            },
            'temporal_errors': {
                'lateral_errors_by_time': comp.lateral_errors_by_time,
                'longitudinal_errors_by_time': comp.longitudinal_errors_by_time
            },
            'predicted_points': comp.predicted_points,
            'ground_truth_points': comp.ground_truth_points,
            'sam_parameters': comp.sam_parameters,
            'input_vx_kmh': comp.input_vx_kmh,
            'delta_vx': comp.delta_vx
        }
        serializable_results['comparisons'].append(comp_dict)

    # Save main results
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)

    # Save temporal statistics
    if temporal_stats:
        with open(os.path.join(output_dir, 'temporal_stats_lc.json'), 'w') as f:
            json.dump(temporal_stats, f, indent=2)

    # Save LK statistics
    if lk_stats:
        with open(os.path.join(output_dir, 'lk_4point_stats.json'), 'w') as f:
            json.dump(lk_stats, f, indent=2)

    print(f"\n💾 All results saved to: {output_dir}/")


def main():
    """Main analysis execution"""

    # Create output directory
    output_dir = "sam_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}/")

    # File paths
    sam_results_file = "complete_pal_predictions.json"
    ground_truth_file = "../lcllm_testing_data_20points.json"

    print("Starting REVISED SAM Results Analysis...")
    print(f"SAM results file: {sam_results_file}")
    print(f"Ground truth file: {ground_truth_file}")

    try:
        # Run main analysis
        analysis_results = analyze_sam_results(sam_results_file, ground_truth_file)

        # Generate report
        generate_analysis_report(analysis_results)

        # Temporal analysis for LC samples
        print("\n🔄 Analyzing temporal errors for Lane Changes...")
        temporal_stats = analyze_temporal_errors_lc(analysis_results['comparisons'],
                                                    analysis_results['time_points'])

        # 4-point analysis for LK samples
        print("\n🔄 Analyzing 4-point errors for Lane Keeping...")
        lk_stats = analyze_lk_4_point_errors(analysis_results['comparisons'])

        # Save all results
        save_detailed_results(analysis_results, temporal_stats, lk_stats, output_dir)

        # Generate plots
        plot_sample_trajectories(analysis_results, output_dir, num_samples=6)

        if temporal_stats:
            plot_temporal_errors(temporal_stats, output_dir)

        if lk_stats:
            plot_lk_4_point_errors(lk_stats, output_dir)

        print(f"\n🎉 REVISED Analysis complete!")
        print(f"📁 All outputs saved to: {output_dir}/")
        print("📊 Generated plots:")
        print("  - lc_sam_trajectory_samples.png (Lane Change SAM reconstructions)")
        print("  - lk_4point_trajectory_samples.png (Lane Keeping 4-point comparisons)")
        print("  - temporal_error_analysis_lc.png (Lane Change temporal errors)")
        print("  - lk_4_point_error_analysis.png (Lane Keeping 4-point errors)")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure both input files exist in the current directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()