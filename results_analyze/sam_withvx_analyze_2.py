#!/usr/bin/env python3
"""
SAM Results Confusion Matrix and Detailed Class Analysis Script
- Generates confusion matrix for intention prediction accuracy
- Provides detailed statistics breakdown by intention class (LK, LLC, RLC)
- Point-by-point error analysis for first 5 and last 5 time points
- Output folder: sam_class_analysis_results/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


@dataclass
class ClassAnalysisResult:
    """Store detailed analysis results for each intention class"""
    class_id: int
    class_name: str
    total_samples: int
    correct_predictions: int
    accuracy: float
    lateral_rmse_mean: float
    lateral_rmse_std: float
    longitudinal_rmse_mean: float
    longitudinal_rmse_std: float
    point_errors: List[Dict]  # Error stats for each time point


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


def parse_predicted_trajectory(prediction_dict: Dict) -> Optional[List[Tuple[float, float]]]:
    """Parse predicted trajectory from the prediction dictionary"""
    if 'trajectory' in prediction_dict and prediction_dict['trajectory']:
        traj = prediction_dict['trajectory']
        if isinstance(traj, str):
            return parse_ground_truth_trajectory(traj)
        elif isinstance(traj, list):
            return traj
    return None


def extract_ground_truth_intention(gt_text: str) -> Optional[int]:
    """Extract ground truth intention from the response text"""
    if '[/INST]' not in gt_text:
        return None

    gt_response = gt_text.split('[/INST]')[1].replace('</s>', '').strip()

    # Look for intention patterns
    intention_patterns = [
        r'- Intention:\s*"?(\d+)',
        r'Intention:\s*"?(\d+)',
        r'- Intention:\s*"([^"]+)"',
        r'Intention:\s*"([^"]+)"',
    ]

    for pattern in intention_patterns:
        match = re.search(pattern, gt_response, re.IGNORECASE)
        if match:
            intention_str = match.group(1)
            if intention_str.isdigit():
                return int(intention_str)
            elif 'left' in intention_str.lower():
                return 1
            elif 'right' in intention_str.lower():
                return 2
            elif 'keep' in intention_str.lower() or 'lane' in intention_str.lower():
                return 0

    return None


def interpolate_4_to_20_points(four_points: List[Tuple[float, float]],
                               time_points_20: List[float]) -> List[Tuple[float, float]]:
    """Interpolate 4 predicted points to 20 time points for LK comparison"""
    if len(four_points) != 4:
        return []

    time_4 = [1.0, 2.0, 3.0, 4.0]
    x_coords = [p[0] for p in four_points]
    y_coords = [p[1] for p in four_points]

    x_interp = np.interp(time_points_20, time_4, x_coords)
    y_interp = np.interp(time_points_20, time_4, y_coords)

    return list(zip(x_interp, y_interp))


def calculate_point_by_point_errors(predicted: List[Tuple[float, float]],
                                    ground_truth: List[Tuple[float, float]]) -> List[Dict]:
    """Calculate detailed error statistics for each time point"""
    if len(predicted) != len(ground_truth):
        min_len = min(len(predicted), len(ground_truth))
        predicted = predicted[:min_len]
        ground_truth = ground_truth[:min_len]

    point_errors = []

    for i in range(len(predicted)):
        pred_x, pred_y = predicted[i]
        gt_x, gt_y = ground_truth[i]

        lat_error = abs(pred_y - gt_y)
        lon_error = abs(pred_x - gt_x)
        euc_error = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

        point_errors.append({
            'point_index': i + 1,
            'lateral_error': lat_error,
            'longitudinal_error': lon_error,
            'euclidean_error': euc_error
        })

    return point_errors


def load_and_process_data(sam_results_file: str, ground_truth_file: str) -> Tuple[List, List, List]:
    """Load data and extract predicted vs actual intentions with trajectory comparisons"""

    print("Loading SAM results...")
    with open(sam_results_file, 'r') as f:
        sam_data = json.load(f)

    print("Loading ground truth dataset...")
    with open(ground_truth_file, 'r') as f:
        gt_data = json.load(f)

    sam_predictions = sam_data['all_predictions']
    time_points = [0.2 * i for i in range(1, 21)]  # [0.2, 0.4, 0.6, ..., 4.0]

    y_true = []  # Ground truth intentions
    y_pred = []  # Predicted intentions
    detailed_comparisons = []  # Detailed trajectory comparisons

    print(f"Processing {len(sam_predictions)} samples...")

    for i, sam_result in enumerate(sam_predictions):
        if i >= len(gt_data):
            break

        gt_sample = gt_data[i]

        # Extract ground truth intention
        gt_intention = extract_ground_truth_intention(gt_sample['text'])
        if gt_intention is None:
            continue

        # Extract predicted intention
        pred_intention = sam_result['prediction']['intention']
        if pred_intention is None:
            continue

        y_true.append(gt_intention)
        y_pred.append(pred_intention)

        # Extract ground truth trajectory
        gt_text = gt_sample['text']
        if '[/INST]' not in gt_text:
            continue

        gt_response = gt_text.split('[/INST]')[1].replace('</s>', '').strip()
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

        # Process predicted trajectory
        input_text = sam_result['input']['input_part']
        vx_kmh = extract_vx_from_input(input_text)
        if vx_kmh is None:
            continue

        vx_ms = vx_kmh / 3.6

        predicted_points = None
        point_errors = None

        if pred_intention == 0:  # Lane keeping
            pred_4_points = parse_predicted_trajectory(sam_result['prediction'])
            if pred_4_points and len(pred_4_points) == 4:
                predicted_points = interpolate_4_to_20_points(pred_4_points, time_points)
        else:  # Lane change
            pred_parameters = sam_result['prediction']['parameters']
            if pred_parameters:
                W = pred_parameters.get('lateral_displacement')
                D = pred_parameters.get('duration')
                v0 = pred_parameters.get('v0')
                delta_vx = pred_parameters.get('Delta_Vx')

                if all(x is not None for x in [W, D, v0, delta_vx]):
                    try:
                        predicted_points = reconstruct_sam_trajectory(
                            W=W, D=D, v0_ms=v0, vx_initial_ms=vx_ms,
                            delta_vx_ms=delta_vx, time_points=time_points
                        )
                    except Exception:
                        continue

        if predicted_points:
            point_errors = calculate_point_by_point_errors(predicted_points, gt_trajectory)

            # Calculate overall RMSE
            pred_array = np.array(predicted_points)
            gt_array = np.array(gt_trajectory)
            lateral_rmse = np.sqrt(np.mean((pred_array[:, 1] - gt_array[:, 1]) ** 2))
            longitudinal_rmse = np.sqrt(np.mean((pred_array[:, 0] - gt_array[:, 0]) ** 2))

            detailed_comparisons.append({
                'sample_id': i,
                'gt_intention': gt_intention,
                'pred_intention': pred_intention,
                'lateral_rmse': lateral_rmse,
                'longitudinal_rmse': longitudinal_rmse,
                'point_errors': point_errors
            })

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} samples...")

    return y_true, y_pred, detailed_comparisons


def generate_confusion_matrix(y_true: List[int], y_pred: List[int], output_dir: str):
    """Generate and save confusion matrix plot"""

    class_names = ['Keep Lane (LK)', 'Left Change (LLC)', 'Right Change (RLC)']
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Create annotations that show both count and percentage
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})

    plt.title('Confusion Matrix: SAM Intention Prediction\n(Count and Percentage)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Intention', fontsize=12)
    plt.ylabel('Actual Intention', fontsize=12)

    # Add accuracy information
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)',
                fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Confusion matrix saved to: {os.path.join(output_dir, 'confusion_matrix.png')}")

    # Print classification report
    print("\n" + "=" * 60)
    print("SKLEARN CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def analyze_by_intention_class(detailed_comparisons: List[Dict]) -> List[ClassAnalysisResult]:
    """Analyze detailed statistics by intention class"""

    class_names = {0: 'Keep Lane (LK)', 1: 'Left Lane Change (LLC)', 2: 'Right Lane Change (RLC)'}
    results = []

    for class_id in [0, 1, 2]:
        class_samples = [comp for comp in detailed_comparisons if comp['gt_intention'] == class_id]

        if not class_samples:
            continue

        # Basic statistics
        total_samples = len(class_samples)
        correct_predictions = sum(1 for comp in class_samples if comp['gt_intention'] == comp['pred_intention'])
        accuracy = correct_predictions / total_samples

        # RMSE statistics
        lateral_rmses = [comp['lateral_rmse'] for comp in class_samples]
        longitudinal_rmses = [comp['longitudinal_rmse'] for comp in class_samples]

        # Point-by-point error analysis
        num_points = len(class_samples[0]['point_errors']) if class_samples else 20
        point_stats = []

        for point_idx in range(num_points):
            lateral_errors = []
            longitudinal_errors = []
            euclidean_errors = []

            for comp in class_samples:
                if point_idx < len(comp['point_errors']):
                    lateral_errors.append(comp['point_errors'][point_idx]['lateral_error'])
                    longitudinal_errors.append(comp['point_errors'][point_idx]['longitudinal_error'])
                    euclidean_errors.append(comp['point_errors'][point_idx]['euclidean_error'])

            if lateral_errors:
                point_stats.append({
                    'point_index': point_idx + 1,
                    'lateral_mean': np.mean(lateral_errors),
                    'lateral_std': np.std(lateral_errors),
                    'longitudinal_mean': np.mean(longitudinal_errors),
                    'longitudinal_std': np.std(longitudinal_errors),
                    'euclidean_mean': np.mean(euclidean_errors),
                    'euclidean_std': np.std(euclidean_errors)
                })

        result = ClassAnalysisResult(
            class_id=class_id,
            class_name=class_names[class_id],
            total_samples=total_samples,
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            lateral_rmse_mean=np.mean(lateral_rmses),
            lateral_rmse_std=np.std(lateral_rmses),
            longitudinal_rmse_mean=np.mean(longitudinal_rmses),
            longitudinal_rmse_std=np.std(longitudinal_rmses),
            point_errors=point_stats
        )

        results.append(result)

    return results


def print_detailed_class_analysis(class_results: List[ClassAnalysisResult]):
    """Print detailed analysis tables by intention class"""

    print("\n" + "=" * 50)
    print("ANALYZING BY INTENTION CLASS")
    print("=" * 50)

    for result in class_results:
        print(f"\n{result.class_name} (Class {result.class_id}):")
        print(f"  Samples: {result.total_samples}")
        print(f"  Accuracy: {result.accuracy:.4f} ({result.correct_predictions}/{result.total_samples})")
        print(f"  Lateral RMSE: {result.lateral_rmse_mean:.4f} ± {result.lateral_rmse_std:.4f}")
        print(f"  Longitudinal RMSE: {result.longitudinal_rmse_mean:.4f} ± {result.longitudinal_rmse_std:.4f}")

        # All 20 points
        print(f"  Point-by-point errors (Mean ± Std) - All points:")
        for i in range(len(result.point_errors)):
            point = result.point_errors[i]
            print(f"    point_{point['point_index']}: "
                  f"Lat={point['lateral_mean']:.3f}±{point['lateral_std']:.3f}, "
                  f"Lon={point['longitudinal_mean']:.3f}±{point['longitudinal_std']:.3f}, "
                  f"Euc={point['euclidean_mean']:.3f}±{point['euclidean_std']:.3f}")


def plot_point_by_point_errors(class_results: List[ClassAnalysisResult], output_dir: str):
    """Plot point-by-point error analysis"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    class_colors = ['blue', 'red', 'green']
    class_names = ['Keep Lane (LK)', 'Left Change (LLC)', 'Right Change (RLC)']

    for i, result in enumerate(class_results):
        if not result.point_errors:
            continue

        time_points = [p['point_index'] * 0.2 for p in result.point_errors]

        # Lateral errors
        lat_means = [p['lateral_mean'] for p in result.point_errors]
        lat_stds = [p['lateral_std'] for p in result.point_errors]

        axes[0, i].plot(time_points, lat_means, color=class_colors[i], linewidth=2, marker='o', markersize=3)
        axes[0, i].fill_between(time_points,
                                np.array(lat_means) - np.array(lat_stds),
                                np.array(lat_means) + np.array(lat_stds),
                                alpha=0.3, color=class_colors[i])
        axes[0, i].set_title(f'{class_names[i]}\nLateral Errors', fontsize=12)
        axes[0, i].set_xlabel('Time (s)')
        axes[0, i].set_ylabel('Lateral Error (m)')
        axes[0, i].grid(True, alpha=0.3)

        # Longitudinal errors
        lon_means = [p['longitudinal_mean'] for p in result.point_errors]
        lon_stds = [p['longitudinal_std'] for p in result.point_errors]

        axes[1, i].plot(time_points, lon_means, color=class_colors[i], linewidth=2, marker='s', markersize=3)
        axes[1, i].fill_between(time_points,
                                np.array(lon_means) - np.array(lon_stds),
                                np.array(lon_means) + np.array(lon_stds),
                                alpha=0.3, color=class_colors[i])
        axes[1, i].set_title(f'{class_names[i]}\nLongitudinal Errors', fontsize=12)
        axes[1, i].set_xlabel('Time (s)')
        axes[1, i].set_ylabel('Longitudinal Error (m)')
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle('Point-by-Point Error Analysis by Intention Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'point_by_point_errors_by_class.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Point-by-point error plots saved to: {os.path.join(output_dir, 'point_by_point_errors_by_class.png')}")


def save_detailed_class_results(class_results: List[ClassAnalysisResult],
                                y_true: List[int], y_pred: List[int],
                                detailed_comparisons: List[Dict], output_dir: str):
    """Save all detailed results to JSON files"""

    # Convert class results to serializable format
    serializable_results = []
    for result in class_results:
        serializable_results.append({
            'class_id': result.class_id,
            'class_name': result.class_name,
            'total_samples': result.total_samples,
            'correct_predictions': result.correct_predictions,
            'accuracy': result.accuracy,
            'lateral_rmse_mean': result.lateral_rmse_mean,
            'lateral_rmse_std': result.lateral_rmse_std,
            'longitudinal_rmse_mean': result.longitudinal_rmse_mean,
            'longitudinal_rmse_std': result.longitudinal_rmse_std,
            'point_errors': result.point_errors
        })

    # Save class analysis results
    with open(os.path.join(output_dir, 'class_analysis_results.json'), 'w') as f:
        json.dump({
            'class_results': serializable_results,
            'confusion_matrix_data': {
                'y_true': y_true,
                'y_pred': y_pred
            },
            'detailed_comparisons': detailed_comparisons
        }, f, indent=2)

    print(f"\n💾 Class analysis results saved to: {output_dir}/class_analysis_results.json")


def main():
    """Main execution function"""

    # Create output directory
    output_dir = "sam_class_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}/")

    # File paths
    sam_results_file = "complete_pal_predictions.json"
    ground_truth_file = "../lcllm_testing_data_20points.json"

    print("Starting SAM Class Analysis...")
    print(f"SAM results file: {sam_results_file}")
    print(f"Ground truth file: {ground_truth_file}")

    try:
        # Load and process data
        y_true, y_pred, detailed_comparisons = load_and_process_data(sam_results_file, ground_truth_file)

        print(f"\n📊 Data Summary:")
        print(f"Total processed samples: {len(y_true)}")
        print(f"Samples with trajectory analysis: {len(detailed_comparisons)}")

        # Generate confusion matrix
        generate_confusion_matrix(y_true, y_pred, output_dir)

        # Analyze by intention class
        class_results = analyze_by_intention_class(detailed_comparisons)

        # Print detailed analysis
        print_detailed_class_analysis(class_results)

        # Generate plots
        plot_point_by_point_errors(class_results, output_dir)

        # Save results
        save_detailed_class_results(class_results, y_true, y_pred, detailed_comparisons, output_dir)

        print(f"\n🎉 Class analysis complete!")
        print(f"📁 All outputs saved to: {output_dir}/")
        print("📊 Generated files:")
        print("  - confusion_matrix.png")
        print("  - point_by_point_errors_by_class.png")
        print("  - class_analysis_results.json")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure both input files exist in the current directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()