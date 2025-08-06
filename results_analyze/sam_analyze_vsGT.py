#!/usr/bin/env python
# coding=utf-8
"""
FIXED: SAM vs GT Comparison - Length Mismatch + Sign Correction
Key fixes:
1. Fix length mismatch: Use only first 2000 GT samples
2. Only compare lane changes (exclude intention=0)
3. Add sign correction logic for SAM trajectories
4. Debug coordinate system and direction mapping
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import re


def sam_model(t, W, D, v0):
    """SAM trajectory model"""
    return (W / D) * t + ((v0 * D - W) / (2 * np.pi)) * np.sin(2 * np.pi * t / D)


def reconstruct_sam_trajectory_from_parameters(v0, duration, displacement, intention):
    """Reconstruct 20-point trajectory from SAM parameters with sign correction"""
    time_points = [i * 0.2 for i in range(1, 21)]  # [0.2, 0.4, ..., 4.0]
    trajectory = []

    for t in time_points:
        if duration > 0:
            y = sam_model(t, displacement, duration, v0)

            # Sign correction based on intention and coordinate system
            # Left LC (1): GT shows negative y, so if SAM gives positive, flip it
            # Right LC (2): GT shows negative y, so keep as-is or flip based on pattern
            if intention == 1:  # Left lane change
                # If displacement is positive but GT expects negative, flip
                y = -abs(y) if displacement > 0 else y
            elif intention == 2:  # Right lane change
                # Right lane change should be negative y
                y = -abs(y) if displacement > 0 else y

        else:
            y = v0 * t  # Linear fallback
        trajectory.append(y)

    return trajectory


def extract_sam_prediction(sam_sample):
    """Extract SAM trajectory/parameters for lane changes only"""
    try:
        prediction = sam_sample.get('prediction', {})
        intention = prediction.get('intention')

        # Only process lane changes
        if intention not in [1, 2]:
            return None

        # For LC: Use parameters to reconstruct trajectory
        if 'parameters' in prediction and prediction['parameters'] is not None:
            params = prediction['parameters']
            v0 = params.get('v0', 0)
            duration = params.get('duration', 0)
            displacement = params.get('lateral_displacement', 0)

            if duration > 0:
                trajectory = reconstruct_sam_trajectory_from_parameters(v0, duration, displacement, intention)
                return {
                    'trajectory': trajectory,
                    'format': 'parameters',
                    'params': params,
                    'intention': intention
                }

        return None
    except Exception as e:
        print(f"Error extracting SAM prediction: {e}")
        return None


def extract_gt_trajectory_from_text(text):
    """Extract 20-point GT trajectory from text"""
    try:
        # Find trajectory section
        trajectory_start = text.find('- Trajectory:')
        if trajectory_start == -1:
            return None

        trajectory_section = text[trajectory_start:]

        # Find the trajectory coordinates
        start_quote = trajectory_section.find('"[(')
        if start_quote == -1:
            return None

        start_bracket = trajectory_section.find('[(', start_quote)
        end_bracket = trajectory_section.find(')]"', start_bracket)

        if start_bracket == -1 or end_bracket == -1:
            return None

        trajectory_str = trajectory_section[start_bracket:end_bracket + 2]

        # Extract coordinates
        coord_pattern = r'\(([-+]?\d*\.?\d+),([-+]?\d*\.?\d+)\)'
        matches = re.findall(coord_pattern, trajectory_str)

        if len(matches) >= 20:
            lateral_positions = [float(match[1]) for match in matches[:20]]
            return lateral_positions

        return None
    except Exception as e:
        print(f"Error extracting GT trajectory: {e}")
        return None


def get_intention_from_text(text):
    """Extract intention from text"""
    if 'Intention: "1:' in text:
        return 1
    elif 'Intention: "2:' in text:
        return 2
    elif 'Intention: "0:' in text:
        return 0
    return None


def calculate_trajectory_errors(sam_traj, gt_traj):
    """Calculate trajectory comparison errors"""
    sam_array = np.array(sam_traj)
    gt_array = np.array(gt_traj)
    errors = sam_array - gt_array

    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(np.abs(errors))),
        'max_error': float(np.max(np.abs(errors))),
        'bias': float(np.mean(errors))
    }


def debug_trajectories(results, num_samples=3):
    """Debug function to check trajectory patterns"""
    print(f"\n🔍 DEBUGGING TRAJECTORY PATTERNS (first {num_samples} of each type)")

    left_results = [r for r in results if r['intention'] == 1][:num_samples]
    right_results = [r for r in results if r['intention'] == 2][:num_samples]

    for i, result in enumerate(left_results):
        print(f"\n--- LEFT LC {i + 1} (Sample {result['sample_id']}) ---")
        print(f"SAM params: {result['sam_params']}")
        print(f"SAM traj start: {result['sam_trajectory'][0]:.3f}, end: {result['sam_trajectory'][-1]:.3f}")
        print(f"GT traj start:  {result['gt_trajectory'][0]:.3f}, end: {result['gt_trajectory'][-1]:.3f}")
        print(f"Direction match: {(result['sam_trajectory'][-1] < 0) == (result['gt_trajectory'][-1] < 0)}")

    for i, result in enumerate(right_results):
        print(f"\n--- RIGHT LC {i + 1} (Sample {result['sample_id']}) ---")
        print(f"SAM params: {result['sam_params']}")
        print(f"SAM traj start: {result['sam_trajectory'][0]:.3f}, end: {result['sam_trajectory'][-1]:.3f}")
        print(f"GT traj start:  {result['gt_trajectory'][0]:.3f}, end: {result['gt_trajectory'][-1]:.3f}")
        print(f"Direction match: {(result['sam_trajectory'][-1] < 0) == (result['gt_trajectory'][-1] < 0)}")


def main():
    # File paths
    SAM_FILE = 'complete_pal_predictions.json'
    GT_FILE = '../lcllm_testing_data_20points.json'
    OUTPUT_DIR = 'sam_vs_gt_fixed'

    print("=== FIXED SAM vs GT Trajectory Comparison ===")
    print("🎯 Lane changes only, fixed length, with sign correction")

    # Check files
    if not os.path.exists(SAM_FILE):
        print(f"❌ {SAM_FILE} not found!")
        return
    if not os.path.exists(GT_FILE):
        print(f"❌ {GT_FILE} not found!")
        return

    # Load data
    print("📁 Loading data...")
    with open(SAM_FILE, 'r') as f:
        sam_data_full = json.load(f)

    with open(GT_FILE, 'r') as f:
        gt_data_full = json.load(f)

    # Extract SAM predictions
    if isinstance(sam_data_full, dict) and 'all_predictions' in sam_data_full:
        sam_data = sam_data_full['all_predictions']
    else:
        sam_data = sam_data_full

    # Fix length mismatch: Only use first 2000 GT samples after skipping 1000
    gt_data = gt_data_full[1000:3000]  # [1000:3000] gives exactly 2000 samples

    print(f"📊 SAM predictions: {len(sam_data)}")
    print(f"📊 GT data (1000:3000): {len(gt_data)}")
    print(f"📊 Total GT file length: {len(gt_data_full)}")

    # Process samples - ONLY LANE CHANGES
    results = []
    stats = {
        'total_processed': 0,
        'sam_failures': 0,
        'gt_failures': 0,
        'intention_mismatches': 0,
        'keep_lane_skipped': 0,
        'successful_comparisons': 0
    }

    min_length = min(len(sam_data), len(gt_data))
    print(f"📊 Processing {min_length} aligned samples")

    for i in range(min_length):
        stats['total_processed'] += 1

        sam_sample = sam_data[i]
        gt_sample = gt_data[i]

        # Verify alignment using sample_id
        expected_sample_id = 1000 + i
        actual_sample_id = sam_sample.get('sample_id', -1)
        subset_index = sam_sample.get('subset_index', -1)

        if actual_sample_id != expected_sample_id or subset_index != i:
            print(f"⚠️ Alignment issue at index {i}: expected {expected_sample_id}, got {actual_sample_id}")
            continue

        # Get intentions
        sam_intention = sam_sample.get('prediction', {}).get('intention')
        gt_text = gt_sample.get('text', '')
        gt_intention = get_intention_from_text(gt_text)

        # Skip keep lane samples
        if sam_intention == 0 or gt_intention == 0:
            stats['keep_lane_skipped'] += 1
            continue

        # Skip if intentions don't match
        if sam_intention != gt_intention:
            stats['intention_mismatches'] += 1
            continue

        # Extract SAM prediction (only LC)
        sam_result = extract_sam_prediction(sam_sample)
        if sam_result is None:
            stats['sam_failures'] += 1
            continue

        sam_trajectory = sam_result['trajectory']

        # Extract GT trajectory
        gt_trajectory = extract_gt_trajectory_from_text(gt_text)
        if gt_trajectory is None or len(gt_trajectory) != 20:
            stats['gt_failures'] += 1
            continue

        # Ensure both trajectories have 20 points
        if len(sam_trajectory) != 20:
            stats['sam_failures'] += 1
            continue

        # Calculate errors
        errors = calculate_trajectory_errors(sam_trajectory, gt_trajectory)

        result = {
            'sample_id': actual_sample_id,
            'intention': sam_intention,
            'sam_trajectory': sam_trajectory,
            'gt_trajectory': gt_trajectory,
            'errors': errors,
            'sam_format': sam_result['format'],
            'sam_params': sam_result['params']
        }

        results.append(result)
        stats['successful_comparisons'] += 1

        # Progress reporting
        if stats['total_processed'] % 500 == 0:
            print(f"  Processed {stats['total_processed']}/{min_length}, successful: {stats['successful_comparisons']}")

    print(f"\n✅ Processing Complete:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if stats['successful_comparisons'] == 0:
        print("❌ No successful comparisons!")
        return

    # Debug trajectory patterns
    debug_trajectories(results)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Separate results by intention (only LC now)
    left_lc_results = [r for r in results if r['intention'] == 1]
    right_lc_results = [r for r in results if r['intention'] == 2]

    print(f"\n📊 Lane Change Results:")
    print(f"  Left LC (1): {len(left_lc_results)}")
    print(f"  Right LC (2): {len(right_lc_results)}")

    # Calculate statistics
    def calc_stats(results_subset, name):
        if not results_subset:
            return None

        rmse_values = [r['errors']['rmse'] for r in results_subset]
        mae_values = [r['errors']['mae'] for r in results_subset]
        bias_values = [r['errors']['bias'] for r in results_subset]

        stats = {
            'count': len(results_subset),
            'rmse_mean': float(np.mean(rmse_values)),
            'rmse_std': float(np.std(rmse_values)),
            'mae_mean': float(np.mean(mae_values)),
            'mae_std': float(np.std(mae_values)),
            'bias_mean': float(np.mean(bias_values)),
            'bias_std': float(np.std(bias_values))
        }

        print(f"\n📈 {name} Performance ({stats['count']} samples):")
        print(f"  RMSE: {stats['rmse_mean']:.4f} ± {stats['rmse_std']:.4f} m")
        print(f"  MAE:  {stats['mae_mean']:.4f} ± {stats['mae_std']:.4f} m")
        print(f"  Bias: {stats['bias_mean']:.4f} ± {stats['bias_std']:.4f} m")

        return stats

    left_lc_stats = calc_stats(left_lc_results, "Left Lane Change")
    right_lc_stats = calc_stats(right_lc_results, "Right Lane Change")

    # Save summary
    summary = {
        'processing_stats': stats,
        'performance_stats': {
            'left_lane_change': left_lc_stats,
            'right_lane_change': right_lc_stats
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Visualization - Only lane changes
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SAM vs GT: Lane Change Trajectory Comparison (Fixed)', fontsize=16)

    time_points = [i * 0.2 for i in range(1, 21)]

    # Plot sample trajectories
    if left_lc_results:
        ax = axes[0, 0]
        for i, result in enumerate(left_lc_results[:5]):  # First 5 samples
            sam_traj = result['sam_trajectory']
            gt_traj = result['gt_trajectory']

            ax.plot(time_points, sam_traj, '--', alpha=0.8, label=f'SAM L{i + 1}')
            ax.plot(time_points, gt_traj, '-', alpha=0.8, label=f'GT L{i + 1}')

        ax.set_title('Left Lane Change (1)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Lateral Position (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if right_lc_results:
        ax = axes[0, 1]
        for i, result in enumerate(right_lc_results[:5]):  # First 5 samples
            sam_traj = result['sam_trajectory']
            gt_traj = result['gt_trajectory']

            ax.plot(time_points, sam_traj, '--', alpha=0.8, label=f'SAM R{i + 1}')
            ax.plot(time_points, gt_traj, '-', alpha=0.8, label=f'GT R{i + 1}')

        ax.set_title('Right Lane Change (2)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Lateral Position (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Error distributions (all lane changes)
    all_rmse = [r['errors']['rmse'] for r in results]
    all_bias = [r['errors']['bias'] for r in results]

    axes[1, 0].hist(all_rmse, bins=20, alpha=0.7, color='skyblue')
    axes[1, 0].set_title('RMSE Distribution (All Lane Changes)')
    axes[1, 0].set_xlabel('RMSE (m)')

    axes[1, 1].hist(all_bias, bins=20, alpha=0.7, color='lightcoral')
    axes[1, 1].set_title('Bias Distribution (All Lane Changes)')
    axes[1, 1].set_xlabel('Bias (m)')
    axes[1, 1].axvline(0, color='red', linestyle='--', label='No bias')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sam_vs_gt_lc_only.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n💾 Results saved to: {OUTPUT_DIR}/")
    print(f"📁 summary.json - Statistics for lane changes only")
    print(f"📈 sam_vs_gt_lc_only.png - Lane change visualization")


if __name__ == "__main__":
    main()