#!/usr/bin/env python3
"""
FIXED SAM Results Analysis Script
Compares SAM model predictions with ground truth trajectories from 20-point dataset
FIXES: Lane keeping handling, ground truth extraction, trajectory reconstruction
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import re
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


def reconstruct_lane_keeping_trajectory(vx_initial_ms: float, time_points: List[float]) -> List[Tuple[float, float]]:
    """
    Reconstruct lane keeping trajectory (straight line motion)
    """
    trajectory = []
    for t in time_points:
        x = vx_initial_ms * t  # Constant velocity motion
        y = 0.0  # No lateral movement for lane keeping
        trajectory.append((x, y))
    return trajectory


def parse_ground_truth_trajectory(trajectory_str: str) -> List[Tuple[float, float]]:
    """
    FIXED: Parse trajectory string with improved regex patterns
    """
    if not trajectory_str:
        return []

    # Clean the string
    trajectory_str = trajectory_str.strip().strip('"\'')

    # Try multiple patterns to match different formats
    patterns = [
        r'\[\s*\(([^)]+)\)\s*(?:,\s*\(([^)]+)\))*\s*\]',  # Full bracket notation
        r'\(([^)]+)\)',  # Individual parentheses
    ]

    coordinates = []

    # Pattern to extract individual coordinate pairs
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
        'longitudinal_mae': float(np.mean(np.abs(longitudinal_errors)))
    }


def analyze_sam_results(sam_results_file: str, ground_truth_file: str) -> Dict:
    """FIXED: Main analysis function with proper handling of all scenarios"""

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
        'lane_keeping_reconstructed': 0,
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

        # FIXED: Better trajectory extraction with multiple patterns
        gt_trajectory = None
        trajectory_patterns = [
            r'- Trajectory:\s*"([^"]+)"',  # With dash and quotes
            r'Trajectory:\s*"([^"]+)"',  # Without dash, with quotes
            r'- Trajectory:\s*([^\n]+)',  # With dash, no quotes
            r'Trajectory:\s*([^\n]+)',  # Without dash, no quotes
        ]

        for pattern in trajectory_patterns:
            match = re.search(pattern, gt_response, re.IGNORECASE)
            if match:
                gt_trajectory = parse_ground_truth_trajectory(match.group(1))
                if len(gt_trajectory) == 20:  # Ensure 20 points
                    break

        if not gt_trajectory or len(gt_trajectory) != 20:
            stats['failed_comparisons'].append(
                f"Sample {i}: Could not extract 20-point ground truth trajectory (found {len(gt_trajectory) if gt_trajectory else 0} points)")
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

        # FIXED: Handle both lane keeping and lane changes
        if pred_intention == 0:  # Lane keeping
            stats['lane_keeping_count'] += 1

            # For lane keeping, reconstruct using constant velocity
            predicted_points = reconstruct_lane_keeping_trajectory(vx_ms, time_points)
            prediction_type = "lane_keeping_reconstruction"
            stats['lane_keeping_reconstructed'] += 1

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
                        f"Sample {i}: Missing SAM parameters - W:{W}, D:{D}, v0:{v0}, Delta_Vx:{delta_vx}")
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

            elif pred_trajectory:
                # Use direct trajectory prediction
                stats['trajectory_direct_count'] += 1
                prediction_type = "trajectory_direct"

                # Assuming pred_trajectory is already a list of tuples
                if len(pred_trajectory) >= 20:
                    predicted_points = pred_trajectory[:20]
                else:
                    stats['failed_comparisons'].append(
                        f"Sample {i}: Direct trajectory has insufficient points: {len(pred_trajectory)}")
                    continue
            else:
                stats['failed_comparisons'].append(f"Sample {i}: Lane change without SAM parameters or trajectory")
                continue

        if predicted_points is None:
            continue

        # Calculate errors
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
            delta_vx=delta_vx
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


def generate_analysis_report(analysis_results: Dict) -> None:
    """FIXED: Generate comprehensive analysis report"""

    comparisons = analysis_results['comparisons']
    stats = analysis_results['statistics']

    print("\n" + "=" * 60)
    print("FIXED SAM RESULTS ANALYSIS REPORT")
    print("=" * 60)

    print(f"\n📊 OVERVIEW:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Successful comparisons: {stats['successful_comparisons']}")
    print(f"Failed comparisons: {len(stats['failed_comparisons'])}")
    print(f"Success rate: {stats['successful_comparisons'] / stats['total_samples'] * 100:.1f}%")

    print(f"\n🎯 PREDICTION TYPES:")
    print(f"Lane keeping: {stats['lane_keeping_count']} (reconstructed: {stats['lane_keeping_reconstructed']})")
    print(f"Lane change: {stats['lane_change_count']}")
    print(f"SAM reconstructions: {stats['sam_reconstruction_count']}")
    print(f"Direct trajectories: {stats['trajectory_direct_count']}")

    if not comparisons:
        print("\n❌ No successful comparisons to analyze!")
        return

    # Calculate aggregate statistics
    lateral_rmse_values = [c.lateral_rmse for c in comparisons if c.lateral_rmse != float('inf')]
    longitudinal_rmse_values = [c.longitudinal_rmse for c in comparisons if c.longitudinal_rmse != float('inf')]
    lateral_mae_values = [c.lateral_mae for c in comparisons if c.lateral_mae != float('inf')]
    longitudinal_mae_values = [c.longitudinal_mae for c in comparisons if c.longitudinal_mae != float('inf')]

    print(f"\n📈 OVERALL ERROR STATISTICS:")
    if lateral_rmse_values:
        print(f"Lateral RMSE: {np.mean(lateral_rmse_values):.3f} ± {np.std(lateral_rmse_values):.3f} m")
        print(f"Lateral MAE:  {np.mean(lateral_mae_values):.3f} ± {np.std(lateral_mae_values):.3f} m")

    if longitudinal_rmse_values:
        print(f"Longitudinal RMSE: {np.mean(longitudinal_rmse_values):.3f} ± {np.std(longitudinal_rmse_values):.3f} m")
        print(f"Longitudinal MAE:  {np.mean(longitudinal_mae_values):.3f} ± {np.std(longitudinal_mae_values):.3f} m")

    # Intention-based breakdown
    print(f"\n🔄 BY INTENTION:")
    for intention in [0, 1, 2]:
        intention_comparisons = [c for c in comparisons if c.intention == intention]
        if intention_comparisons:
            intention_name = {0: "Keep Lane", 1: "Left Change", 2: "Right Change"}[intention]
            lat_rmse = [c.lateral_rmse for c in intention_comparisons if c.lateral_rmse != float('inf')]
            lon_rmse = [c.longitudinal_rmse for c in intention_comparisons if c.longitudinal_rmse != float('inf')]
            lat_mae = [c.lateral_mae for c in intention_comparisons if c.lateral_mae != float('inf')]
            lon_mae = [c.longitudinal_mae for c in intention_comparisons if c.longitudinal_mae != float('inf')]

            print(f"{intention_name}: {len(intention_comparisons)} samples")
            if lat_rmse:
                print(f"  Lateral RMSE: {np.mean(lat_rmse):.3f} ± {np.std(lat_rmse):.3f} m")
                print(f"  Lateral MAE:  {np.mean(lat_mae):.3f} ± {np.std(lat_mae):.3f} m")
            if lon_rmse:
                print(f"  Longitudinal RMSE: {np.mean(lon_rmse):.3f} ± {np.std(lon_rmse):.3f} m")
                print(f"  Longitudinal MAE:  {np.mean(lon_mae):.3f} ± {np.std(lon_mae):.3f} m")

    # Prediction type breakdown
    print(f"\n🔧 BY PREDICTION TYPE:")
    for pred_type in ["lane_keeping_reconstruction", "sam_reconstruction", "trajectory_direct"]:
        type_comparisons = [c for c in comparisons if c.prediction_type == pred_type]
        if type_comparisons:
            lat_rmse = [c.lateral_rmse for c in type_comparisons if c.lateral_rmse != float('inf')]
            lon_rmse = [c.longitudinal_rmse for c in type_comparisons if c.longitudinal_rmse != float('inf')]

            print(f"{pred_type}: {len(type_comparisons)} samples")
            if lat_rmse:
                print(f"  Lateral RMSE: {np.mean(lat_rmse):.3f} ± {np.std(lat_rmse):.3f} m")
            if lon_rmse:
                print(f"  Longitudinal RMSE: {np.mean(lon_rmse):.3f} ± {np.std(lon_rmse):.3f} m")

    # Failed comparisons summary
    if stats['failed_comparisons']:
        print(f"\n❌ FAILED COMPARISONS ({len(stats['failed_comparisons'])}):")
        failure_types = {}
        for failure in stats['failed_comparisons']:
            # Extract failure type
            if ':' in failure:
                failure_type = failure.split(':', 2)[1].strip() if len(failure.split(':', 2)) > 1 else failure
            else:
                failure_type = failure
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1

        for failure_type, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {failure_type}: {count}")


def save_detailed_results(analysis_results: Dict, output_file: str) -> None:
    """Save detailed results to JSON file"""

    # Convert comparisons to serializable format
    serializable_results = {
        'metadata': {
            'analysis_type': 'FIXED_SAM_vs_GroundTruth_20Points',
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
            'predicted_points': comp.predicted_points,
            'ground_truth_points': comp.ground_truth_points,
            'sam_parameters': comp.sam_parameters,
            'input_vx_kmh': comp.input_vx_kmh,
            'delta_vx': comp.delta_vx
        }
        serializable_results['comparisons'].append(comp_dict)

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {output_file}")


def plot_sample_trajectories(analysis_results: Dict, num_samples: int = 6) -> None:
    """Plot sample trajectory comparisons for different categories"""

    comparisons = analysis_results['comparisons']
    if not comparisons:
        print("No comparisons to plot!")
        return

    # Select samples from different categories
    lane_keeping = [c for c in comparisons if c.intention == 0]
    left_change = [c for c in comparisons if c.intention == 1]
    right_change = [c for c in comparisons if c.intention == 2]

    samples_to_plot = []

    # Get 2 from each category if available
    if lane_keeping:
        samples_to_plot.extend(lane_keeping[:2])
    if left_change:
        samples_to_plot.extend(left_change[:2])
    if right_change:
        samples_to_plot.extend(right_change[:2])

    # Fill up to num_samples if we don't have enough
    if len(samples_to_plot) < num_samples:
        remaining = [c for c in comparisons if c not in samples_to_plot]
        samples_to_plot.extend(remaining[:num_samples - len(samples_to_plot)])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, comp in enumerate(samples_to_plot[:6]):
        ax = axes[i]

        # Extract coordinates
        pred_x = [p[0] for p in comp.predicted_points]
        pred_y = [p[1] for p in comp.predicted_points]
        gt_x = [p[0] for p in comp.ground_truth_points]
        gt_y = [p[1] for p in comp.ground_truth_points]

        # Plot trajectories
        ax.plot(gt_x, gt_y, 'b-o', label='Ground Truth', markersize=3, linewidth=2)
        ax.plot(pred_x, pred_y, 'r--s', label=f'{comp.prediction_type}', markersize=3, linewidth=2)

        intention_names = {0: "Keep Lane", 1: "Left Change", 2: "Right Change"}
        ax.set_xlabel('Longitudinal (m)')
        ax.set_ylabel('Lateral (m)')
        ax.set_title(f'Sample {comp.sample_id}: {intention_names[comp.intention]}\n'
                     f'Lat RMSE: {comp.lateral_rmse:.2f}m, Lon RMSE: {comp.longitudinal_rmse:.2f}m\n'
                     f'Type: {comp.prediction_type}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(samples_to_plot), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('fixed_sam_trajectory_comparison_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Sample trajectory plots saved to: fixed_sam_trajectory_comparison_samples.png")


def main():
    """Main analysis execution"""

    # File paths
    sam_results_file = "complete_pal_predictions.json"
    ground_truth_file = "../lcllm_testing_data_20points.json"

    print("Starting FIXED SAM Results Analysis...")
    print(f"SAM results file: {sam_results_file}")
    print(f"Ground truth file: {ground_truth_file}")

    try:
        # Run analysis
        analysis_results = analyze_sam_results(sam_results_file, ground_truth_file)

        # Generate report
        generate_analysis_report(analysis_results)

        # Save detailed results
        save_detailed_results(analysis_results, "fixed_sam_analysis_detailed_results.json")

        # Plot sample comparisons
        plot_sample_trajectories(analysis_results, num_samples=6)

        print(f"\n🎉 FIXED Analysis complete!")
        print(f"📁 Detailed results: fixed_sam_analysis_detailed_results.json")
        print(f"📊 Sample plots: fixed_sam_trajectory_comparison_samples.png")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure both input files exist in the current directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()