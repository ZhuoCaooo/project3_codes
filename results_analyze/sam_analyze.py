#!/usr/bin/env python
# coding=utf-8
"""
SAM Trajectory Reconstruction and Analysis Script
Analyzes PAL-LLM predictions by:
1. Reconstructing trajectories from predicted hyperparameters
2. Reconstructing trajectories from ground truth hyperparameters
3. Comparing trajectories at different temporal stages
4. Computing comprehensive error metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import seaborn as sns


class SAMTrajectoryReconstructor:
    """Reconstructs trajectories using Sinusoidal Acceleration Model (SAM)"""

    def __init__(self):
        pass

    def sam_model_with_v0(self, t: float, W: float, D: float, v0: float) -> float:
        """
        Modified Sinusoidal Acceleration Model (SAM) with known initial velocity v0
        Based on your exact implementation - matches your curve fitting approach

        Args:
            t: Time since start (seconds)
            W: Total lateral displacement (corresponds to fitted_W)
            D: Duration (corresponds to fitted_D)
            v0: Initial lateral velocity at boundary (absolute value)

        Returns:
            Lateral position relative to starting point
        """
        return (W / D) * t + ((v0 * D - W) / (2 * np.pi)) * np.sin(2 * np.pi * t / D)

    def reconstruct_trajectory(self, v0: float, duration: float, lateral_displacement: float,
                               time_points: Optional[List[float]] = None) -> List[Tuple[float, float]]:
        """
        Reconstruct trajectory using SAM parameters - UPDATED to match your exact formula.

        Args:
            v0: Initial lateral velocity (m/s) - uses absolute value as in your implementation
            duration: Lane change duration (s) - corresponds to 'fitted_D' in your data
            lateral_displacement: Total lateral displacement (m) - corresponds to 'fitted_W' in your data
            time_points: Custom time points, defaults to [1, 2, 3, 4] seconds

        Returns:
            List of (x, y) coordinate tuples representing trajectory points
        """
        if time_points is None:
            time_points = [1.0, 2.0, 3.0, 4.0]  # Default 4-second prediction

        trajectory = []

        # Use absolute value of v0 as in your implementation
        v0_abs = abs(v0)
        W = lateral_displacement  # fitted_W
        D = duration  # fitted_D

        for t in time_points:
            # Longitudinal position (assuming constant longitudinal velocity)
            # Estimate from typical highway speeds (~25 m/s ≈ 90 km/h)
            x = 25.0 * t  # You can adjust this based on your data

            # Lateral position using YOUR exact SAM formula
            if D > 0:
                y = self.sam_model_with_v0(t, W, D, v0_abs)
            else:
                y = v0_abs * t  # Fallback if duration is invalid

            trajectory.append((x, y))

        return trajectory

    def validate_parameters(self, params: Dict) -> bool:
        """Validate if parameters are reasonable for trajectory reconstruction - UPDATED for your parameter names"""
        # Updated to match your exact parameter names from curve fitting
        required_keys = ['v0', 'fitted_D', 'fitted_W']  # OR check for both naming conventions

        # Check for either naming convention (your training data vs fitted results)
        has_fitted_params = all(key in params for key in ['v0', 'fitted_D', 'fitted_W'])
        has_standard_params = all(key in params for key in ['v0', 'duration', 'lateral_displacement'])

        if not (has_fitted_params or has_standard_params):
            return False

        # Extract parameters with fallback naming
        v0 = params.get('v0', 0)
        duration = params.get('fitted_D', params.get('duration', 0))
        displacement = params.get('fitted_W', params.get('lateral_displacement', 0))

        # Check for reasonable parameter ranges based on your curve fitting bounds
        if not (-5.0 <= v0 <= 5.0):  # Lateral velocity shouldn't be extreme
            return False
        if not (2.0 <= duration <= 8.0):  # Duration bounds from your curve_fit: [2.0, 8.0]
            return False
        if not (0.5 <= abs(displacement) <= 8.0):  # Displacement bounds from your curve_fit: [0.5, 8.0]
            return False

        return True


class TrajectoryAnalyzer:
    """Analyzes and compares trajectories at different temporal stages"""

    def __init__(self):
        self.reconstructor = SAMTrajectoryReconstructor()

    def calculate_trajectory_errors(self, pred_traj: List[Tuple[float, float]],
                                    gt_traj: List[Tuple[float, float]]) -> Dict:
        """Calculate comprehensive trajectory errors"""
        if len(pred_traj) != len(gt_traj):
            return {'error': 'Trajectory length mismatch'}

        pred_array = np.array(pred_traj)
        gt_array = np.array(gt_traj)

        # Overall errors
        lateral_errors = pred_array[:, 1] - gt_array[:, 1]
        longitudinal_errors = pred_array[:, 0] - gt_array[:, 0]

        # Euclidean distances at each time point
        euclidean_errors = np.sqrt(lateral_errors ** 2 + longitudinal_errors ** 2)

        results = {
            'overall_metrics': {
                'lateral_rmse': float(np.sqrt(np.mean(lateral_errors ** 2))),
                'longitudinal_rmse': float(np.sqrt(np.mean(longitudinal_errors ** 2))),
                'lateral_mae': float(np.mean(np.abs(lateral_errors))),
                'longitudinal_mae': float(np.mean(np.abs(longitudinal_errors))),
                'euclidean_rmse': float(np.sqrt(np.mean(euclidean_errors ** 2))),
                'euclidean_mae': float(np.mean(euclidean_errors))
            },
            'temporal_analysis': {
                'time_points': [1.0, 2.0, 3.0, 4.0],
                'lateral_errors': lateral_errors.tolist(),
                'longitudinal_errors': longitudinal_errors.tolist(),
                'euclidean_errors': euclidean_errors.tolist()
            }
        }

        # Stage-wise analysis
        stages = {
            'first_1s': [0],  # t=1s
            'second_1s': [1],  # t=2s
            'third_1s': [2],  # t=3s
            'fourth_1s': [3],  # t=4s
            'first_2s': [0, 1],  # t=1-2s
            'last_2s': [2, 3],  # t=3-4s
            'first_half': [0, 1],  # First half
            'second_half': [2, 3]  # Second half
        }

        results['stage_analysis'] = {}
        for stage_name, indices in stages.items():
            stage_lateral = lateral_errors[indices]
            stage_longitudinal = longitudinal_errors[indices]
            stage_euclidean = euclidean_errors[indices]

            results['stage_analysis'][stage_name] = {
                'lateral_rmse': float(np.sqrt(np.mean(stage_lateral ** 2))),
                'longitudinal_rmse': float(np.sqrt(np.mean(stage_longitudinal ** 2))),
                'euclidean_rmse': float(np.sqrt(np.mean(stage_euclidean ** 2))),
                'lateral_mae': float(np.mean(np.abs(stage_lateral))),
                'longitudinal_mae': float(np.mean(np.abs(stage_longitudinal))),
                'euclidean_mae': float(np.mean(stage_euclidean)),
                'time_points_included': [1.0 + i for i in indices]
            }

        return results

    def analyze_parameter_errors(self, pred_params: Dict, gt_params: Dict) -> Dict:
        """Analyze differences in SAM parameters - UPDATED to handle your parameter naming"""
        errors = {}

        # Map parameter names to handle both conventions
        param_mapping = {
            'v0': ['v0'],
            'duration': ['fitted_D', 'duration', 'D'],
            'lateral_displacement': ['fitted_W', 'lateral_displacement', 'W']
        }

        for param_key, possible_names in param_mapping.items():
            # Find parameter in predicted results
            pred_val = None
            for name in possible_names:
                if name in pred_params:
                    pred_val = pred_params[name]
                    break

            # Find parameter in ground truth results
            gt_val = None
            for name in possible_names:
                if name in gt_params:
                    gt_val = gt_params[name]
                    break

            # Calculate errors if both values found
            if pred_val is not None and gt_val is not None:
                abs_error = abs(pred_val - gt_val)
                rel_error = abs_error / abs(gt_val) if gt_val != 0 else float('inf')

                errors[param_key] = {
                    'predicted': pred_val,
                    'ground_truth': gt_val,
                    'absolute_error': abs_error,
                    'relative_error': rel_error,
                    'percentage_error': rel_error * 100
                }

        return errors

    def process_dataset(self, sam_data_file: str, output_dir: str = 'trajectory_analysis') -> Dict:
        """Process entire dataset and generate comprehensive analysis"""

        # Load data with error handling
        try:
            with open(sam_data_file, 'r') as f:
                sam_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error loading JSON file: {e}")
            return {}
        except FileNotFoundError:
            print(f"❌ File not found: {sam_data_file}")
            return {}

        # Handle different JSON structures
        if isinstance(sam_data, dict):
            # If it's a dict, try to extract the data array
            if 'all_predictions' in sam_data:
                sam_data = sam_data['all_predictions']
            elif 'data' in sam_data:
                sam_data = sam_data['data']
            else:
                print(f"❌ Unexpected JSON structure. Keys found: {list(sam_data.keys())}")
                return {}

        # Ensure we have a list
        if not isinstance(sam_data, list):
            print(f"❌ Expected list of samples, got {type(sam_data)}")
            return {}

        print(f"📊 Processing {len(sam_data)} samples...")

        # Debug: Check first sample structure
        if len(sam_data) > 0:
            print(f"🔍 First sample type: {type(sam_data[0])}")
            if isinstance(sam_data[0], dict):
                print(f"🔍 First sample keys: {list(sam_data[0].keys())}")
            elif isinstance(sam_data[0], str):
                print(f"🔍 First sample content (first 100 chars): {sam_data[0][:100]}")
                # Try to parse individual strings as JSON
                try:
                    parsed_sample = json.loads(sam_data[0])
                    print(f"🔍 Parsed sample keys: {list(parsed_sample.keys())}")
                except:
                    print("❌ Cannot parse individual samples as JSON")
                    return {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize collectors
        all_results = []
        valid_reconstructions = 0
        parameter_comparison_results = []
        trajectory_comparison_results = []

        # Process each sample
        for i, sample in enumerate(sam_data):
            # Handle string samples (parse as JSON)
            if isinstance(sample, str):
                try:
                    sample = json.loads(sample)
                except json.JSONDecodeError:
                    print(f"❌ Failed to parse sample {i} as JSON")
                    continue

            # Ensure sample is now a dictionary
            if not isinstance(sample, dict):
                print(f"❌ Sample {i} is not a dictionary: {type(sample)}")
                continue
            sample_id = sample.get('sample_id', i)
            intention = sample.get('predicted_intention', sample.get('prediction', {}).get('intention'))

            result = {
                'sample_id': sample_id,
                'intention': intention,
                'processing_status': 'failed',
                'error_message': None
            }

            try:
                # Only process lane change predictions (intention 1 or 2)
                if intention not in [1, 2]:
                    result['error_message'] = f"Skipping non-lane-change prediction (intention={intention})"
                    all_results.append(result)
                    continue

                # Extract parameters from the nested structure
                pred_params = sample.get('predicted_parameters')
                gt_params = sample.get('ground_truth_parameters')

                # Handle nested structure from complete_pal_predictions.json
                if pred_params is None:
                    pred_params = sample.get('prediction', {}).get('parameters')
                if gt_params is None:
                    gt_params = sample.get('ground_truth', {}).get('parameters')

                # Debug: Print first few samples to understand structure
                if i < 3:
                    print(f"Sample {i}: intention={intention}")
                    print(f"  pred_params: {pred_params}")
                    print(f"  gt_params: {gt_params}")

                if pred_params is None:
                    result['error_message'] = f"No predicted parameters found"
                    all_results.append(result)
                    continue

                if gt_params is None:
                    result['error_message'] = f"No ground truth parameters found"
                    all_results.append(result)
                    continue

                # Validate parameters - UPDATED to handle your parameter naming
                if not self.reconstructor.validate_parameters(pred_params):
                    result['error_message'] = f"Invalid predicted parameters: {pred_params}"
                    all_results.append(result)
                    continue

                if not self.reconstructor.validate_parameters(gt_params):
                    result['error_message'] = f"Invalid ground truth parameters: {gt_params}"
                    all_results.append(result)
                    continue

                # Extract parameters with flexible naming - handles both your curve fitting output and training data
                def extract_sam_params(params):
                    v0 = params.get('v0', 0)
                    duration = params.get('fitted_D', params.get('duration', params.get('D', 0)))
                    displacement = params.get('fitted_W', params.get('lateral_displacement', params.get('W', 0)))
                    return v0, duration, displacement

                pred_v0, pred_duration, pred_displacement = extract_sam_params(pred_params)
                gt_v0, gt_duration, gt_displacement = extract_sam_params(gt_params)

                # Reconstruct trajectories using YOUR exact SAM formula
                pred_trajectory = self.reconstructor.reconstruct_trajectory(
                    v0=pred_v0,
                    duration=pred_duration,
                    lateral_displacement=pred_displacement
                )

                gt_trajectory = self.reconstructor.reconstruct_trajectory(
                    v0=gt_v0,
                    duration=gt_duration,
                    lateral_displacement=gt_displacement
                )

                # Calculate trajectory errors
                trajectory_errors = self.calculate_trajectory_errors(pred_trajectory, gt_trajectory)

                # Calculate parameter errors
                parameter_errors = self.analyze_parameter_errors(pred_params, gt_params)

                # Store results
                result.update({
                    'processing_status': 'success',
                    'predicted_parameters': pred_params,
                    'ground_truth_parameters': gt_params,
                    'predicted_trajectory': pred_trajectory,
                    'ground_truth_trajectory': gt_trajectory,
                    'trajectory_errors': trajectory_errors,
                    'parameter_errors': parameter_errors
                })

                valid_reconstructions += 1
                parameter_comparison_results.append(parameter_errors)
                trajectory_comparison_results.append(trajectory_errors)

            except Exception as e:
                result['error_message'] = f"Processing error: {str(e)}"

            all_results.append(result)

            # Progress update
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(sam_data)} samples... ({valid_reconstructions} successful)")

        print(f"✓ Processing complete: {valid_reconstructions}/{len(sam_data)} successful reconstructions")

        # Generate summary statistics
        summary_stats = self.generate_summary_statistics(
            trajectory_comparison_results,
            parameter_comparison_results
        )

        # Save detailed results
        detailed_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(sam_data),
                'valid_reconstructions': valid_reconstructions,
                'success_rate': valid_reconstructions / len(sam_data) if sam_data else 0
            },
            'summary_statistics': summary_stats,
            'individual_results': all_results
        }

        # Save results
        with open(os.path.join(output_dir, 'detailed_trajectory_analysis.json'), 'w') as f:
            json.dump(detailed_results, f, indent=2)

        with open(os.path.join(output_dir, 'summary_statistics.json'), 'w') as f:
            json.dump(summary_stats, f, indent=2)

        # Generate visualizations
        if valid_reconstructions > 0:
            self.generate_visualizations(all_results, output_dir)

        return detailed_results

    def generate_summary_statistics(self, trajectory_results: List[Dict],
                                    parameter_results: List[Dict]) -> Dict:
        """Generate comprehensive summary statistics"""

        summary = {
            'trajectory_performance': {},
            'parameter_performance': {},
            'temporal_analysis': {}
        }

        if trajectory_results:
            # Overall trajectory performance
            overall_metrics = [r['overall_metrics'] for r in trajectory_results]

            for metric in ['lateral_rmse', 'longitudinal_rmse', 'euclidean_rmse',
                           'lateral_mae', 'longitudinal_mae', 'euclidean_mae']:
                values = [m[metric] for m in overall_metrics]
                summary['trajectory_performance'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

            # Stage-wise performance
            stages = ['first_1s', 'second_1s', 'third_1s', 'fourth_1s',
                      'first_2s', 'last_2s', 'first_half', 'second_half']

            for stage in stages:
                stage_data = [r['stage_analysis'][stage] for r in trajectory_results if
                              stage in r.get('stage_analysis', {})]

                if stage_data:
                    summary['temporal_analysis'][stage] = {}
                    for metric in ['lateral_rmse', 'longitudinal_rmse', 'euclidean_rmse']:
                        values = [s[metric] for s in stage_data]
                        summary['temporal_analysis'][stage][metric] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'median': float(np.median(values))
                        }

        if parameter_results:
            # Parameter performance
            for param in ['v0', 'duration', 'lateral_displacement']:
                param_data = [r[param] for r in parameter_results if param in r]

                if param_data:
                    abs_errors = [p['absolute_error'] for p in param_data]
                    rel_errors = [p['relative_error'] for p in param_data if p['relative_error'] != float('inf')]

                    summary['parameter_performance'][param] = {
                        'absolute_error': {
                            'mean': float(np.mean(abs_errors)),
                            'std': float(np.std(abs_errors)),
                            'median': float(np.median(abs_errors))
                        },
                        'relative_error': {
                            'mean': float(np.mean(rel_errors)) if rel_errors else None,
                            'std': float(np.std(rel_errors)) if rel_errors else None,
                            'median': float(np.median(rel_errors)) if rel_errors else None
                        }
                    }

        return summary

    def generate_visualizations(self, results: List[Dict], output_dir: str):
        """Generate comprehensive visualizations"""

        # Filter successful results
        valid_results = [r for r in results if r['processing_status'] == 'success']

        if not valid_results:
            print("⚠️ No valid results for visualization")
            return

        print(f"📈 Generating visualizations for {len(valid_results)} samples...")

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Trajectory Error Distribution
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Trajectory Error Distributions', fontsize=16)

        # Extract error data
        lateral_rmse = [r['trajectory_errors']['overall_metrics']['lateral_rmse'] for r in valid_results]
        longitudinal_rmse = [r['trajectory_errors']['overall_metrics']['longitudinal_rmse'] for r in valid_results]
        euclidean_rmse = [r['trajectory_errors']['overall_metrics']['euclidean_rmse'] for r in valid_results]

        axes[0, 0].hist(lateral_rmse, bins=20, alpha=0.7)
        axes[0, 0].set_title('Lateral RMSE Distribution')
        axes[0, 0].set_xlabel('RMSE (m)')

        axes[0, 1].hist(longitudinal_rmse, bins=20, alpha=0.7)
        axes[0, 1].set_title('Longitudinal RMSE Distribution')
        axes[0, 1].set_xlabel('RMSE (m)')

        axes[0, 2].hist(euclidean_rmse, bins=20, alpha=0.7)
        axes[0, 2].set_title('Euclidean RMSE Distribution')
        axes[0, 2].set_xlabel('RMSE (m)')

        # 2. Temporal Error Analysis
        stages = ['first_1s', 'second_1s', 'third_1s', 'fourth_1s']
        stage_labels = ['1st Second', '2nd Second', '3rd Second', '4th Second']

        lateral_stage_errors = []
        for stage in stages:
            stage_errors = [r['trajectory_errors']['stage_analysis'][stage]['lateral_rmse']
                            for r in valid_results if stage in r['trajectory_errors'].get('stage_analysis', {})]
            lateral_stage_errors.append(stage_errors)

        axes[1, 0].boxplot(lateral_stage_errors, labels=stage_labels)
        axes[1, 0].set_title('Lateral RMSE by Time Stage')
        axes[1, 0].set_ylabel('RMSE (m)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 3. Parameter Error Analysis
        param_names = ['v0', 'duration', 'lateral_displacement']
        param_labels = ['Initial Velocity (m/s)', 'Duration (s)', 'Displacement (m)']

        param_abs_errors = []
        for param in param_names:
            errors = [r['parameter_errors'][param]['absolute_error']
                      for r in valid_results if param in r.get('parameter_errors', {})]
            param_abs_errors.append(errors)

        axes[1, 1].boxplot(param_abs_errors, labels=param_labels)
        axes[1, 1].set_title('Parameter Absolute Errors')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # 4. Error Correlation
        if len(lateral_rmse) > 1:
            axes[1, 2].scatter(lateral_rmse, longitudinal_rmse, alpha=0.6)
            axes[1, 2].set_xlabel('Lateral RMSE (m)')
            axes[1, 2].set_ylabel('Longitudinal RMSE (m)')
            axes[1, 2].set_title('Lateral vs Longitudinal Error')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trajectory_error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Detailed temporal analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Detailed Temporal Error Analysis', fontsize=16)

        # Extract temporal data
        time_points = [1, 2, 3, 4]
        all_lateral_errors = []
        all_longitudinal_errors = []

        for t_idx in range(4):
            lat_errors = [r['trajectory_errors']['temporal_analysis']['lateral_errors'][t_idx]
                          for r in valid_results]
            lon_errors = [r['trajectory_errors']['temporal_analysis']['longitudinal_errors'][t_idx]
                          for r in valid_results]
            all_lateral_errors.append(lat_errors)
            all_longitudinal_errors.append(lon_errors)

        # Box plots for each time point
        axes[0, 0].boxplot(all_lateral_errors, labels=[f't={t}s' for t in time_points])
        axes[0, 0].set_title('Lateral Errors Over Time')
        axes[0, 0].set_ylabel('Error (m)')

        axes[0, 1].boxplot(all_longitudinal_errors, labels=[f't={t}s' for t in time_points])
        axes[0, 1].set_title('Longitudinal Errors Over Time')
        axes[0, 1].set_ylabel('Error (m)')

        # Mean error trends
        mean_lateral = [np.mean(errors) for errors in all_lateral_errors]
        mean_longitudinal = [np.mean(errors) for errors in all_longitudinal_errors]

        axes[1, 0].plot(time_points, mean_lateral, 'o-', label='Mean Lateral Error')
        axes[1, 0].fill_between(time_points,
                                [np.mean(e) - np.std(e) for e in all_lateral_errors],
                                [np.mean(e) + np.std(e) for e in all_lateral_errors],
                                alpha=0.3)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Error (m)')
        axes[1, 0].set_title('Mean Lateral Error Trend')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(time_points, mean_longitudinal, 'o-', label='Mean Longitudinal Error', color='orange')
        axes[1, 1].fill_between(time_points,
                                [np.mean(e) - np.std(e) for e in all_longitudinal_errors],
                                [np.mean(e) + np.std(e) for e in all_longitudinal_errors],
                                alpha=0.3, color='orange')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Error (m)')
        axes[1, 1].set_title('Mean Longitudinal Error Trend')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_error_details.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualizations saved to {output_dir}/")


def main():
    """Main analysis function"""

    # Configuration
    SAM_DATA_FILE = 'complete_pal_predictions.json'
    OUTPUT_DIR = 'trajectory_analysis'

    print("=== SAM Trajectory Reconstruction & Analysis ===")
    print(f"Input file: {SAM_DATA_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not os.path.exists(SAM_DATA_FILE):
        print(f"❌ Error: Input file '{SAM_DATA_FILE}' not found!")
        return

    analyzer = TrajectoryAnalyzer()

    try:
        results = analyzer.process_dataset(SAM_DATA_FILE, OUTPUT_DIR)

        if not results:
            print("❌ Processing failed - no results generated")
            return

        metadata = results.get('metadata', {})
        summary = results.get('summary_statistics', {})

        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Total samples: {metadata.get('total_samples', 0)}")
        print(f"Valid reconstructions: {metadata.get('valid_reconstructions', 0)}")
        success_rate = metadata.get('success_rate', 0)
        print(f"Success rate: {success_rate * 100:.1f}%")

        if 'trajectory_performance' in summary and summary['trajectory_performance']:
            print(f"\n📊 TRAJECTORY PERFORMANCE:")
            traj_perf = summary['trajectory_performance']

            lat_mean = traj_perf.get('lateral_rmse', {}).get('mean')
            lat_std = traj_perf.get('lateral_rmse', {}).get('std')
            if lat_mean is not None and lat_std is not None:
                print(f"  Lateral RMSE: {lat_mean:.4f} ± {lat_std:.4f}")

            lon_mean = traj_perf.get('longitudinal_rmse', {}).get('mean')
            lon_std = traj_perf.get('longitudinal_rmse', {}).get('std')
            if lon_mean is not None and lon_std is not None:
                print(f"  Longitudinal RMSE: {lon_mean:.4f} ± {lon_std:.4f}")

        if 'temporal_analysis' in summary and summary['temporal_analysis']:
            print(f"\n⏱️ TEMPORAL ANALYSIS:")
            temporal = summary['temporal_analysis']
            for stage in ['first_1s', 'second_1s', 'third_1s', 'fourth_1s']:
                if stage in temporal:
                    rmse = temporal[stage].get('lateral_rmse', {}).get('mean')
                    if rmse is not None:
                        print(f"  {stage.replace('_', ' ').title()}: {rmse:.4f} lateral RMSE")

        print(f"\n💾 Results saved to: {OUTPUT_DIR}/")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()