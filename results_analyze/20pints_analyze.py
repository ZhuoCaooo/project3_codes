#!/usr/bin/env python
# coding=utf-8
"""
LC-LLM 20-Point Detailed Results Analysis Script
Analyzes validation results JSON file with 20-point trajectory predictions
Provides comprehensive performance analysis by prediction class and point-by-point errors
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, Counter
import os
from datetime import datetime

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class LCLLMAnalyzer20Point:
    def __init__(self, results_path):
        """Initialize analyzer with result path."""
        self.results_path = results_path
        self.results_data = None
        self.analysis_results = {}

        # Lane change type mapping
        self.intention_map = {0: 'Keep Lane (LK)', 1: 'Left Lane Change (LLC)', 2: 'Right Lane Change (RLC)'}

    def load_data(self):
        """Load validation results."""
        print("Loading validation results...")

        # Load validation results
        with open(self.results_path, 'r') as f:
            self.results_data = json.load(f)

        # Check if it's the new format or old format
        if 'detailed_sample_results' in self.results_data:
            # Old format with experiment info
            samples = self.results_data['detailed_sample_results']
            print(f"✓ Loaded validation results: {len(samples)} samples")
            print(f"\nExperiment Info:")
            for key, value in self.results_data['experiment_info'].items():
                print(f"  {key}: {value}")
        else:
            # New format - list of samples directly
            samples = self.results_data
            print(f"✓ Loaded validation results: {len(samples)} samples")

        # Store samples for easier access
        self.samples = samples

        # Detect trajectory points count from first valid sample
        for sample in samples:
            if (sample.get('processing_successful', True) and
                    sample.get('ground_truth', {}).get('trajectory')):
                points_count = len(sample['ground_truth']['trajectory'])
                print(f"✓ Detected trajectory format: {points_count} points per trajectory")
                break

    def analyze_by_intention_class(self):
        """Analyze performance by lane change intention class."""
        print("\n" + "=" * 50)
        print("ANALYZING BY INTENTION CLASS")
        print("=" * 50)

        # Group samples by ground truth intention
        intention_groups = defaultdict(list)

        for sample in self.samples:
            if (sample.get('processing_successful', True) and
                    sample.get('ground_truth', {}).get('intention') is not None):
                gt_intention = sample['ground_truth']['intention']
                intention_groups[gt_intention].append(sample)

        class_analysis = {}

        for intention, samples in intention_groups.items():
            intention_name = self.intention_map[intention]

            # Calculate accuracy for this class
            correct_predictions = sum(1 for s in samples
                                      if s.get('predictions', {}).get('intention') == intention)
            accuracy = correct_predictions / len(samples) if samples else 0

            # Calculate trajectory errors
            lateral_errors = []
            longitudinal_errors = []
            point_errors = {}

            # Initialize point errors for all 20 points
            for i in range(20):
                point_errors[f'point_{i + 1}'] = []

            for sample in samples:
                traj_metrics = sample.get('evaluation', {}).get('trajectory_metrics')
                if traj_metrics:
                    lateral_errors.append(traj_metrics['lateral_rmse'])
                    longitudinal_errors.append(traj_metrics['longitudinal_rmse'])

                    # Point-by-point errors
                    lat_errors = traj_metrics.get('point_by_point_lateral_errors', [])
                    lon_errors = traj_metrics.get('point_by_point_longitudinal_errors', [])

                    for i, (lat_err, lon_err) in enumerate(zip(lat_errors, lon_errors)):
                        if i < 20:  # 20 points
                            point_key = f'point_{i + 1}'
                            point_errors[point_key].append({
                                'lateral': abs(lat_err),
                                'longitudinal': abs(lon_err),
                                'euclidean': np.sqrt(lat_err ** 2 + lon_err ** 2)
                            })

            # Calculate statistics
            class_stats = {
                'intention': intention,
                'intention_name': intention_name,
                'total_samples': len(samples),
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'lateral_rmse_mean': np.mean(lateral_errors) if lateral_errors else None,
                'lateral_rmse_std': np.std(lateral_errors) if lateral_errors else None,
                'longitudinal_rmse_mean': np.mean(longitudinal_errors) if longitudinal_errors else None,
                'longitudinal_rmse_std': np.std(longitudinal_errors) if longitudinal_errors else None,
                'point_errors': {}
            }

            # Point-by-point statistics
            for point_key, errors in point_errors.items():
                if errors:
                    lat_errs = [e['lateral'] for e in errors]
                    lon_errs = [e['longitudinal'] for e in errors]
                    euc_errs = [e['euclidean'] for e in errors]

                    class_stats['point_errors'][point_key] = {
                        'lateral_mean': np.mean(lat_errs),
                        'lateral_std': np.std(lat_errs),
                        'longitudinal_mean': np.mean(lon_errs),
                        'longitudinal_std': np.std(lon_errs),
                        'euclidean_mean': np.mean(euc_errs),
                        'euclidean_std': np.std(euc_errs),
                        'count': len(errors)
                    }

            class_analysis[intention] = class_stats

            # Print results for this class
            print(f"\n{intention_name} (Class {intention}):")
            print(f"  Samples: {len(samples)}")
            print(f"  Accuracy: {accuracy:.4f} ({correct_predictions}/{len(samples)})")
            if lateral_errors:
                print(f"  Lateral RMSE: {np.mean(lateral_errors):.4f} ± {np.std(lateral_errors):.4f}")
                print(f"  Longitudinal RMSE: {np.mean(longitudinal_errors):.4f} ± {np.std(longitudinal_errors):.4f}")

            # Print first 5 and last 5 point errors for brevity
            print(f"  Point-by-point errors (Mean ± Std) - First 5 points:")
            for i in range(5):
                point_key = f'point_{i + 1}'
                if point_key in class_stats['point_errors'] and class_stats['point_errors'][point_key]:
                    stats = class_stats['point_errors'][point_key]
                    print(f"    {point_key}: Lat={stats['lateral_mean']:.3f}±{stats['lateral_std']:.3f}, "
                          f"Lon={stats['longitudinal_mean']:.3f}±{stats['longitudinal_std']:.3f}, "
                          f"Euc={stats['euclidean_mean']:.3f}±{stats['euclidean_std']:.3f}")

            print(f"  Point-by-point errors (Mean ± Std) - Last 5 points:")
            for i in range(15, 20):
                point_key = f'point_{i + 1}'
                if point_key in class_stats['point_errors'] and class_stats['point_errors'][point_key]:
                    stats = class_stats['point_errors'][point_key]
                    print(f"    {point_key}: Lat={stats['lateral_mean']:.3f}±{stats['lateral_std']:.3f}, "
                          f"Lon={stats['longitudinal_mean']:.3f}±{stats['longitudinal_std']:.3f}, "
                          f"Euc={stats['euclidean_mean']:.3f}±{stats['euclidean_std']:.3f}")

        self.analysis_results['class_analysis'] = class_analysis
        return class_analysis

    def analyze_confusion_matrix(self):
        """Create confusion matrix for intention predictions."""
        print("\n" + "=" * 50)
        print("CONFUSION MATRIX ANALYSIS")
        print("=" * 50)

        gt_intentions = []
        pred_intentions = []

        for sample in self.samples:
            if (sample.get('processing_successful', True) and
                    sample.get('ground_truth', {}).get('intention') is not None and
                    sample.get('predictions', {}).get('intention') is not None):
                gt_intentions.append(sample['ground_truth']['intention'])
                pred_intentions.append(sample['predictions']['intention'])

        # Create confusion matrix
        unique_intentions = sorted(set(gt_intentions + pred_intentions))
        confusion_matrix = np.zeros((len(unique_intentions), len(unique_intentions)), dtype=int)

        for gt, pred in zip(gt_intentions, pred_intentions):
            gt_idx = unique_intentions.index(gt)
            pred_idx = unique_intentions.index(pred)
            confusion_matrix[gt_idx, pred_idx] += 1

        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("Rows: Ground Truth, Columns: Predictions")

        header = "GT\\Pred".ljust(10)
        for intention in unique_intentions:
            header += f"{self.intention_map[intention][:6]}".rjust(8)
        print(header)

        for i, gt_intention in enumerate(unique_intentions):
            row = f"{self.intention_map[gt_intention][:8]}".ljust(10)
            for j, pred_intention in enumerate(unique_intentions):
                row += f"{confusion_matrix[i, j]}".rjust(8)
            print(row)

        # Calculate per-class precision, recall, F1
        print("\nPer-class Metrics:")
        print("Class".ljust(20) + "Precision".rjust(10) + "Recall".rjust(10) + "F1-Score".rjust(10))

        class_metrics = {}
        for i, intention in enumerate(unique_intentions):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            class_metrics[intention] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }

            intention_name = self.intention_map[intention]
            print(f"{intention_name}".ljust(20) + f"{precision:.4f}".rjust(10) +
                  f"{recall:.4f}".rjust(10) + f"{f1:.4f}".rjust(10))

        self.analysis_results['confusion_matrix'] = confusion_matrix
        self.analysis_results['class_metrics'] = class_metrics
        return confusion_matrix, class_metrics

    def analyze_point_by_point_errors(self):
        """Analyze trajectory errors for each of the 20 prediction points."""
        print("\n" + "=" * 50)
        print("POINT-BY-POINT TRAJECTORY ANALYSIS (20 POINTS)")
        print("=" * 50)

        point_errors = {f'point_{i + 1}': {'lateral': [], 'longitudinal': [], 'euclidean': []}
                        for i in range(20)}

        for sample in self.samples:
            if (sample.get('processing_successful', True) and
                    sample.get('evaluation', {}).get('trajectory_metrics')):

                traj_metrics = sample['evaluation']['trajectory_metrics']
                lat_errors = traj_metrics.get('point_by_point_lateral_errors', [])
                lon_errors = traj_metrics.get('point_by_point_longitudinal_errors', [])

                for i, (lat_err, lon_err) in enumerate(zip(lat_errors, lon_errors)):
                    if i < 20:  # Only analyze first 20 points
                        point_key = f'point_{i + 1}'
                        point_errors[point_key]['lateral'].append(abs(lat_err))
                        point_errors[point_key]['longitudinal'].append(abs(lon_err))
                        point_errors[point_key]['euclidean'].append(np.sqrt(lat_err ** 2 + lon_err ** 2))

        # Calculate statistics for each point
        point_stats = {}
        print("\nPoint-by-Point Error Statistics:")
        print("Point".ljust(8) + "Lateral RMSE".rjust(15) + "Longitudinal RMSE".rjust(18) + "Euclidean RMSE".rjust(16))

        for point_key, errors in point_errors.items():
            if errors['lateral']:
                stats = {
                    'lateral_mean': np.mean(errors['lateral']),
                    'lateral_std': np.std(errors['lateral']),
                    'longitudinal_mean': np.mean(errors['longitudinal']),
                    'longitudinal_std': np.std(errors['longitudinal']),
                    'euclidean_mean': np.mean(errors['euclidean']),
                    'euclidean_std': np.std(errors['euclidean']),
                    'count': len(errors['lateral'])
                }
                point_stats[point_key] = stats

        # Print first 10 points
        print("\nFirst 10 points:")
        for i in range(10):
            point_key = f'point_{i + 1}'
            if point_key in point_stats:
                stats = point_stats[point_key]
                print(f"{point_key}".ljust(8) +
                      f"{stats['lateral_mean']:.3f}±{stats['lateral_std']:.3f}".rjust(15) +
                      f"{stats['longitudinal_mean']:.3f}±{stats['longitudinal_std']:.3f}".rjust(18) +
                      f"{stats['euclidean_mean']:.3f}±{stats['euclidean_std']:.3f}".rjust(16))

        # Print last 10 points
        print("\nLast 10 points:")
        for i in range(10, 20):
            point_key = f'point_{i + 1}'
            if point_key in point_stats:
                stats = point_stats[point_key]
                print(f"{point_key}".ljust(8) +
                      f"{stats['lateral_mean']:.3f}±{stats['lateral_std']:.3f}".rjust(15) +
                      f"{stats['longitudinal_mean']:.3f}±{stats['longitudinal_std']:.3f}".rjust(18) +
                      f"{stats['euclidean_mean']:.3f}±{stats['euclidean_std']:.3f}".rjust(16))

        self.analysis_results['point_errors'] = point_stats
        return point_stats

    def analyze_temporal_progression(self):
        """Analyze how errors progress over time (0.2s intervals)."""
        print("\n" + "=" * 50)
        print("TEMPORAL ERROR PROGRESSION ANALYSIS")
        print("=" * 50)

        # Group points by time intervals
        time_intervals = []
        for i in range(20):
            time_intervals.append(f"{(i + 1) * 0.2:.1f}s")

        if 'point_errors' in self.analysis_results:
            point_data = self.analysis_results['point_errors']

            print("Time Progression of Errors:")
            print("Time".ljust(8) + "Lateral".rjust(12) + "Longitudinal".rjust(15) + "Euclidean".rjust(12))

            # Print every 4th point for readability (1s intervals)
            for i in range(0, 20, 4):
                point_key = f'point_{i + 1}'
                if point_key in point_data:
                    stats = point_data[point_key]
                    time_label = time_intervals[i]
                    print(f"{time_label}".ljust(8) +
                          f"{stats['lateral_mean']:.3f}".rjust(12) +
                          f"{stats['longitudinal_mean']:.3f}".rjust(15) +
                          f"{stats['euclidean_mean']:.3f}".rjust(12))

    def create_visualizations(self, save_dir='analysis_plots_20points'):
        """Create comprehensive visualizations of the results."""
        print(f"\n" + "=" * 50)
        print("CREATING VISUALIZATIONS")
        print("=" * 50)

        os.makedirs(save_dir, exist_ok=True)

        # 1. Class-wise accuracy bar plot
        if 'class_analysis' in self.analysis_results:
            class_data = self.analysis_results['class_analysis']

            intentions = list(class_data.keys())
            accuracies = [class_data[i]['accuracy'] for i in intentions]
            intention_names = [class_data[i]['intention_name'] for i in intentions]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(intention_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
            plt.title('Intention Prediction Accuracy by Class (20-Point Trajectories)', fontsize=14, fontweight='bold')
            plt.ylabel('Accuracy', fontsize=12)
            plt.ylim(0, 1)

            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.savefig(f'{save_dir}/accuracy_by_class.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved accuracy plot: {save_dir}/accuracy_by_class.png")

        # 2. RMSE comparison by class
        if 'class_analysis' in self.analysis_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            intentions = list(class_data.keys())
            lateral_means = [class_data[i]['lateral_rmse_mean'] for i in intentions if
                             class_data[i]['lateral_rmse_mean']]
            lateral_stds = [class_data[i]['lateral_rmse_std'] for i in intentions if class_data[i]['lateral_rmse_std']]
            lon_means = [class_data[i]['longitudinal_rmse_mean'] for i in intentions if
                         class_data[i]['longitudinal_rmse_mean']]
            lon_stds = [class_data[i]['longitudinal_rmse_std'] for i in intentions if
                        class_data[i]['longitudinal_rmse_std']]

            if lateral_means and lon_means:
                # Lateral RMSE
                bars1 = ax1.bar(intention_names, lateral_means, yerr=lateral_stds,
                                color=['skyblue', 'lightcoral', 'lightgreen'],
                                capsize=5, alpha=0.8)
                ax1.set_title('Lateral RMSE by Class', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Lateral RMSE (m)', fontsize=10)

                # Longitudinal RMSE
                bars2 = ax2.bar(intention_names, lon_means, yerr=lon_stds,
                                color=['skyblue', 'lightcoral', 'lightgreen'],
                                capsize=5, alpha=0.8)
                ax2.set_title('Longitudinal RMSE by Class', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Longitudinal RMSE (m)', fontsize=10)

                plt.tight_layout()
                plt.savefig(f'{save_dir}/rmse_by_class.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved RMSE plot: {save_dir}/rmse_by_class.png")

        # 3. Point-by-point error progression (all 20 points)
        if 'point_errors' in self.analysis_results:
            point_data = self.analysis_results['point_errors']

            points = [f'point_{i + 1}' for i in range(20)]
            lateral_means = [point_data[p]['lateral_mean'] for p in points if p in point_data]
            lateral_stds = [point_data[p]['lateral_std'] for p in points if p in point_data]
            lon_means = [point_data[p]['longitudinal_mean'] for p in points if p in point_data]
            lon_stds = [point_data[p]['longitudinal_std'] for p in points if p in point_data]

            plt.figure(figsize=(16, 8))
            x_pos = np.arange(len(lateral_means))
            time_labels = [f'{(i + 1) * 0.2:.1f}s' for i in range(len(lateral_means))]

            plt.errorbar(x_pos, lateral_means, yerr=lateral_stds,
                         label='Lateral Error', marker='o', capsize=3, linewidth=2, markersize=4)
            plt.errorbar(x_pos, lon_means, yerr=lon_stds,
                         label='Longitudinal Error', marker='s', capsize=3, linewidth=2, markersize=4)

            plt.xlabel('Time (seconds)', fontsize=12)
            plt.ylabel('Mean Absolute Error (m)', fontsize=12)
            plt.title('Point-by-Point Trajectory Error Progression (20 Points, 0.2s intervals)',
                      fontsize=14, fontweight='bold')

            # Show every 4th label for readability
            plt.xticks(x_pos[::4], time_labels[::4], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/point_by_point_errors.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved point-by-point plot: {save_dir}/point_by_point_errors.png")

        # 4. Temporal error heatmap
        if 'class_analysis' in self.analysis_results:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

            # Prepare data for heatmap
            intentions = sorted(class_data.keys())
            intention_names_short = [self.intention_map[i].replace(' (', '\n(') for i in intentions]

            # Lateral errors heatmap
            lateral_matrix = []
            lon_matrix = []

            for intention in intentions:
                if intention in class_data:
                    lat_row = []
                    lon_row = []
                    for i in range(20):
                        point_key = f'point_{i + 1}'
                        if (point_key in class_data[intention]['point_errors'] and
                                class_data[intention]['point_errors'][point_key]):
                            lat_row.append(class_data[intention]['point_errors'][point_key]['lateral_mean'])
                            lon_row.append(class_data[intention]['point_errors'][point_key]['longitudinal_mean'])
                        else:
                            lat_row.append(0)
                            lon_row.append(0)
                    lateral_matrix.append(lat_row)
                    lon_matrix.append(lon_row)

            if lateral_matrix:
                # Lateral heatmap
                sns.heatmap(lateral_matrix, annot=False, cmap='Reds', ax=ax1,
                            xticklabels=[f'{(i + 1) * 0.2:.1f}s' for i in range(20)],
                            yticklabels=intention_names_short,
                            cbar_kws={'label': 'Lateral Error (m)'})
                ax1.set_title('Lateral Error by Class and Time', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Time (seconds)')

                # Show every 4th x-tick
                ax1.set_xticks(np.arange(0, 20, 4) + 0.5)
                ax1.set_xticklabels([f'{(i + 1) * 0.2:.1f}s' for i in range(0, 20, 4)])

                # Longitudinal heatmap
                sns.heatmap(lon_matrix, annot=False, cmap='Blues', ax=ax2,
                            xticklabels=[f'{(i + 1) * 0.2:.1f}s' for i in range(20)],
                            yticklabels=intention_names_short,
                            cbar_kws={'label': 'Longitudinal Error (m)'})
                ax2.set_title('Longitudinal Error by Class and Time', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Time (seconds)')

                # Show every 4th x-tick
                ax2.set_xticks(np.arange(0, 20, 4) + 0.5)
                ax2.set_xticklabels([f'{(i + 1) * 0.2:.1f}s' for i in range(0, 20, 4)])

                plt.tight_layout()
                plt.savefig(f'{save_dir}/temporal_error_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved temporal heatmap: {save_dir}/temporal_error_heatmap.png")

        # 5. Confusion matrix heatmap
        if 'confusion_matrix' in self.analysis_results:
            cm = self.analysis_results['confusion_matrix']

            plt.figure(figsize=(8, 6))
            labels = [self.intention_map[i] for i in sorted(self.intention_map.keys())]

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels,
                        cbar_kws={'label': 'Number of Samples'})

            plt.title('Intention Prediction Confusion Matrix (20-Point Trajectories)',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Intention', fontsize=12)
            plt.ylabel('Ground Truth Intention', fontsize=12)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved confusion matrix: {save_dir}/confusion_matrix.png")

        print(f"\n✓ All visualizations saved to '{save_dir}/' directory")

    def convert_numpy_to_python(self, obj):
        """Recursively convert NumPy arrays to Python lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_to_python(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    def save_detailed_analysis(self, output_path='analysis_results_20points.json'):
        """Save all analysis results to a JSON file."""
        analysis_output = {
            'analysis_timestamp': datetime.now().isoformat(),
            'trajectory_format': '20_points_0.2s_intervals',
            'total_samples_analyzed': len(self.samples),
            'detailed_analysis': self.analysis_results
        }

        # Add original experiment info if available
        if hasattr(self, 'results_data') and isinstance(self.results_data, dict):
            if 'experiment_info' in self.results_data:
                analysis_output['original_experiment_info'] = self.results_data['experiment_info']
            if 'summary_statistics' in self.results_data:
                analysis_output['original_summary_stats'] = self.results_data['summary_statistics']

        # Convert NumPy arrays to Python lists for JSON serialization
        analysis_output = self.convert_numpy_to_python(analysis_output)

        with open(output_path, 'w') as f:
            json.dump(analysis_output, f, indent=2)

        print(f"\n✓ Detailed analysis saved to: {output_path}")

    def print_summary_report(self):
        """Print a comprehensive summary report."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE ANALYSIS SUMMARY REPORT (20-POINT TRAJECTORIES)")
        print("=" * 60)

        # Overall performance
        total_samples = len(self.samples)
        processed_samples = sum(1 for s in self.samples if s.get('processing_successful', True))

        # Calculate overall metrics
        all_intentions_correct = 0
        all_lateral_errors = []
        all_lon_errors = []

        for sample in self.samples:
            if (sample.get('processing_successful', True) and
                    sample.get('ground_truth', {}).get('intention') is not None and
                    sample.get('predictions', {}).get('intention') is not None):

                if sample['ground_truth']['intention'] == sample['predictions']['intention']:
                    all_intentions_correct += 1

                traj_metrics = sample.get('evaluation', {}).get('trajectory_metrics')
                if traj_metrics:
                    all_lateral_errors.append(traj_metrics['lateral_rmse'])
                    all_lon_errors.append(traj_metrics['longitudinal_rmse'])

        overall_accuracy = all_intentions_correct / processed_samples if processed_samples > 0 else 0

        print(f"\nOverall Performance:")
        print(f"  Total samples tested: {total_samples}")
        print(f"  Successfully processed: {processed_samples}")
        print(f"  Overall intention accuracy: {overall_accuracy:.4f}")
        if all_lateral_errors:
            print(f"  Overall lateral RMSE: {np.mean(all_lateral_errors):.4f} ± {np.std(all_lateral_errors):.4f}")
            print(f"  Overall longitudinal RMSE: {np.mean(all_lon_errors):.4f} ± {np.std(all_lon_errors):.4f}")

        # Best and worst performing classes
        if 'class_analysis' in self.analysis_results:
            class_data = self.analysis_results['class_analysis']

            # Sort by accuracy
            sorted_classes = sorted(class_data.items(), key=lambda x: x[1]['accuracy'], reverse=True)

            print(f"\nBest performing class:")
            best_class = sorted_classes[0][1]
            print(f"  {best_class['intention_name']}: {best_class['accuracy']:.4f} accuracy")

            print(f"\nWorst performing class:")
            worst_class = sorted_classes[-1][1]
            print(f"  {worst_class['intention_name']}: {worst_class['accuracy']:.4f} accuracy")

        # Point-wise error trend
        if 'point_errors' in self.analysis_results:
            point_data = self.analysis_results['point_errors']
            print(f"\nTrajectory error progression (every 1s):")
            for i in range(0, 20, 5):  # Show every 5th point (1s intervals)
                point_key = f'point_{i + 1}'
                if point_key in point_data:
                    stats = point_data[point_key]
                    time_label = f"{(i + 1) * 0.2:.1f}s"
                    print(f"  {time_label}: Euclidean error = {stats['euclidean_mean']:.3f}m")

        # Error trends
        if 'point_errors' in self.analysis_results:
            point_data = self.analysis_results['point_errors']
            early_errors = []  # First 5 points (1s)
            late_errors = []  # Last 5 points (3.6-4s)

            for i in range(5):
                point_key = f'point_{i + 1}'
                if point_key in point_data:
                    early_errors.append(point_data[point_key]['euclidean_mean'])

            for i in range(15, 20):
                point_key = f'point_{i + 1}'
                if point_key in point_data:
                    late_errors.append(point_data[point_key]['euclidean_mean'])

            if early_errors and late_errors:
                early_avg = np.mean(early_errors)
                late_avg = np.mean(late_errors)
                print(f"\nTemporal Error Analysis:")
                print(f"  Early prediction error (0.2-1.0s): {early_avg:.3f}m")
                print(f"  Late prediction error (3.2-4.0s): {late_avg:.3f}m")
                print(f"  Error progression: {((late_avg - early_avg) / early_avg * 100):+.1f}%")

        print("\n" + "=" * 60)


def main():
    """Main analysis function."""
    print("LC-LLM 20-Point Trajectory Results Analysis Tool")
    print("=" * 50)

    # File path - modify this to match your results file
    results_path = 'validation_20points_detailed_results.json'

    # Check if file exists
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        print("Please make sure the validation results file is in the current directory.")
        print("Expected file format: JSON with detailed sample results including 20-point trajectories")
        return

    # Initialize analyzer
    analyzer = LCLLMAnalyzer20Point(results_path)

    try:
        # Load data
        analyzer.load_data()

        # Run all analyses
        print("\nRunning comprehensive analysis...")
        analyzer.analyze_by_intention_class()
        analyzer.analyze_confusion_matrix()
        analyzer.analyze_point_by_point_errors()
        analyzer.analyze_temporal_progression()

        # Create visualizations
        analyzer.create_visualizations()

        # Save detailed results
        analyzer.save_detailed_analysis()

        # Print summary
        analyzer.print_summary_report()

        print("\n🎉 Analysis completed successfully!")
        print("\nGenerated files:")
        print("  - analysis_results_20points.json (detailed analysis)")
        print("  - analysis_plots_20points/ (visualizations)")
        print("    * accuracy_by_class.png")
        print("    * rmse_by_class.png")
        print("    * point_by_point_errors.png")
        print("    * temporal_error_heatmap.png")
        print("    * confusion_matrix.png")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()