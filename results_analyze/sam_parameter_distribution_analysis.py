#!/usr/bin/env python3
"""
SAM Parameter Distribution Analysis Script
Focus: Analyze distributions of predicted SAM parameters (W, D, v0, Delta_Vx)
For: Lane Change trajectories only
Output: Histograms, box plots, statistics tables
Save to: sam_analysis_results/ folder
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
from scipy import stats


def load_sam_results(sam_results_file: str) -> List[Dict]:
    """Load SAM results and extract LC samples with parameters"""

    print("Loading SAM results...")
    with open(sam_results_file, 'r') as f:
        sam_data = json.load(f)

    sam_predictions = sam_data['all_predictions']
    print(f"Total SAM results: {len(sam_predictions)} samples")

    lc_samples = []
    stats = {
        'total_samples': len(sam_predictions),
        'lc_samples_found': 0,
        'lc_with_sam_params': 0,
        'missing_params': []
    }

    for i, sam_result in enumerate(sam_predictions):
        pred_intention = sam_result['prediction']['intention']
        pred_parameters = sam_result['prediction']['parameters']

        # Only process Lane Change samples (intention != 0)
        if pred_intention == 0:
            continue

        stats['lc_samples_found'] += 1

        if not pred_parameters:
            stats['missing_params'].append(f"Sample {i}: No SAM parameters")
            continue

        # Extract SAM parameters
        W = pred_parameters.get('lateral_displacement')
        D = pred_parameters.get('duration')
        v0 = pred_parameters.get('v0')
        delta_vx = pred_parameters.get('Delta_Vx')

        if W is None or D is None or v0 is None or delta_vx is None:
            missing = [param for param, val in [('W', W), ('D', D), ('v0', v0), ('Delta_Vx', delta_vx)] if val is None]
            stats['missing_params'].append(f"Sample {i}: Missing {missing}")
            continue

        stats['lc_with_sam_params'] += 1

        lc_sample = {
            'sample_id': i,
            'intention': pred_intention,
            'W': W,  # lateral_displacement
            'D': D,  # duration
            'v0': v0,  # initial lateral velocity
            'Delta_Vx': delta_vx  # longitudinal velocity change
        }

        lc_samples.append(lc_sample)

    print(f"LC samples found: {stats['lc_samples_found']}")
    print(f"LC samples with complete SAM parameters: {stats['lc_with_sam_params']}")
    if stats['missing_params']:
        print(f"Samples with missing parameters: {len(stats['missing_params'])}")

    return lc_samples


def calculate_parameter_statistics(lc_samples: List[Dict]) -> pd.DataFrame:
    """Calculate comprehensive statistics for each parameter"""

    if not lc_samples:
        return pd.DataFrame()

    # Extract parameter arrays
    parameters = {
        'W (Lateral Displacement)': [s['W'] for s in lc_samples],
        'D (Duration)': [s['D'] for s in lc_samples],
        'v0 (Initial Lateral Velocity)': [s['v0'] for s in lc_samples],
        'Delta_Vx (Longitudinal Velocity Change)': [s['Delta_Vx'] for s in lc_samples]
    }

    stats_data = []

    for param_name, values in parameters.items():
        values = np.array(values)

        stat_row = {
            'Parameter': param_name,
            'Count': len(values),
            'Mean': np.mean(values),
            'Std': np.std(values),
            'Min': np.min(values),
            'Q25': np.percentile(values, 25),
            'Median': np.median(values),
            'Q75': np.percentile(values, 75),
            'Max': np.max(values),
            'Range': np.max(values) - np.min(values),
            'IQR': np.percentile(values, 75) - np.percentile(values, 25),
            'Skewness': stats.skew(values),
            'Kurtosis': stats.kurtosis(values)
        }

        stats_data.append(stat_row)

    return pd.DataFrame(stats_data)


def calculate_intention_statistics(lc_samples: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate statistics separately for left vs right lane changes"""

    left_samples = [s for s in lc_samples if s['intention'] == 1]
    right_samples = [s for s in lc_samples if s['intention'] == 2]

    def calc_stats_for_group(samples, group_name):
        if not samples:
            return pd.DataFrame()

        parameters = {
            'W': [s['W'] for s in samples],
            'D': [s['D'] for s in samples],
            'v0': [s['v0'] for s in samples],
            'Delta_Vx': [s['Delta_Vx'] for s in samples]
        }

        stats_data = []
        for param_name, values in parameters.items():
            values = np.array(values)
            stat_row = {
                'Group': group_name,
                'Parameter': param_name,
                'Count': len(values),
                'Mean': np.mean(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Median': np.median(values),
                'Max': np.max(values)
            }
            stats_data.append(stat_row)

        return pd.DataFrame(stats_data)

    left_stats = calc_stats_for_group(left_samples, 'Left Change (1)')
    right_stats = calc_stats_for_group(right_samples, 'Right Change (2)')

    return left_stats, right_stats


def plot_parameter_distributions(lc_samples: List[Dict], output_dir: str) -> None:
    """Create comprehensive distribution plots for all parameters"""

    if not lc_samples:
        print("No LC samples to plot!")
        return

    # Extract data
    W_values = [s['W'] for s in lc_samples]
    D_values = [s['D'] for s in lc_samples]
    v0_values = [s['v0'] for s in lc_samples]
    delta_vx_values = [s['Delta_Vx'] for s in lc_samples]
    intentions = [s['intention'] for s in lc_samples]

    # Create main distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # W (Lateral Displacement)
    ax1 = axes[0, 0]
    ax1.hist(W_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(W_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(W_values):.2f}')
    ax1.axvline(np.median(W_values), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(W_values):.2f}')
    ax1.set_xlabel('W - Lateral Displacement (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(
        f'Distribution of W (Lateral Displacement)\nN={len(W_values)}, Range=[{np.min(W_values):.2f}, {np.max(W_values):.2f}]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # D (Duration)
    ax2 = axes[0, 1]
    ax2.hist(D_values, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(D_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(D_values):.2f}')
    ax2.axvline(np.median(D_values), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(D_values):.2f}')
    ax2.set_xlabel('D - Duration (s)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(
        f'Distribution of D (Duration)\nN={len(D_values)}, Range=[{np.min(D_values):.2f}, {np.max(D_values):.2f}]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # v0 (Initial Lateral Velocity)
    ax3 = axes[1, 0]
    ax3.hist(v0_values, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(np.mean(v0_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(v0_values):.3f}')
    ax3.axvline(np.median(v0_values), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(v0_values):.3f}')
    ax3.set_xlabel('v0 - Initial Lateral Velocity (m/s)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(
        f'Distribution of v0 (Initial Lateral Velocity)\nN={len(v0_values)}, Range=[{np.min(v0_values):.3f}, {np.max(v0_values):.3f}]')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Delta_Vx (Longitudinal Velocity Change)
    ax4 = axes[1, 1]
    ax4.hist(delta_vx_values, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.axvline(np.mean(delta_vx_values), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(delta_vx_values):.2f}')
    ax4.axvline(np.median(delta_vx_values), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(delta_vx_values):.2f}')
    ax4.set_xlabel('Delta_Vx - Longitudinal Velocity Change (m/s)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(
        f'Distribution of Delta_Vx (Longitudinal Velocity Change)\nN={len(delta_vx_values)}, Range=[{np.min(delta_vx_values):.2f}, {np.max(delta_vx_values):.2f}]')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sam_parameter_distributions_all.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(
        f"📊 Parameter distributions plot saved to: {os.path.join(output_dir, 'sam_parameter_distributions_all.png')}")


def plot_intention_comparison(lc_samples: List[Dict], output_dir: str) -> None:
    """Create side-by-side comparison plots for left vs right lane changes"""

    if not lc_samples:
        return

    # Separate by intention
    left_samples = [s for s in lc_samples if s['intention'] == 1]
    right_samples = [s for s in lc_samples if s['intention'] == 2]

    if not left_samples or not right_samples:
        print("Not enough samples for intention comparison!")
        return

    # Extract data for each group
    left_data = {
        'W': [s['W'] for s in left_samples],
        'D': [s['D'] for s in left_samples],
        'v0': [s['v0'] for s in left_samples],
        'Delta_Vx': [s['Delta_Vx'] for s in left_samples]
    }

    right_data = {
        'W': [s['W'] for s in right_samples],
        'D': [s['D'] for s in right_samples],
        'v0': [s['v0'] for s in right_samples],
        'Delta_Vx': [s['Delta_Vx'] for s in right_samples]
    }

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    parameters = ['W', 'D', 'v0', 'Delta_Vx']
    titles = ['W - Lateral Displacement (m)', 'D - Duration (s)',
              'v0 - Initial Lateral Velocity (m/s)', 'Delta_Vx - Longitudinal Velocity Change (m/s)']
    colors = ['blue', 'green', 'purple', 'orange']

    for i, (param, title, color) in enumerate(zip(parameters, titles, colors)):
        ax = axes[i // 2, i % 2]

        # Plot histograms for both intentions
        ax.hist(left_data[param], bins=20, alpha=0.6, color='red',
                label=f'Left Change (N={len(left_data[param])})', density=True)
        ax.hist(right_data[param], bins=20, alpha=0.6, color='blue',
                label=f'Right Change (N={len(right_data[param])})', density=True)

        # Add mean lines
        ax.axvline(np.mean(left_data[param]), color='red', linestyle='--', linewidth=2,
                   label=f'Left Mean: {np.mean(left_data[param]):.3f}')
        ax.axvline(np.mean(right_data[param]), color='blue', linestyle='--', linewidth=2,
                   label=f'Right Mean: {np.mean(right_data[param]):.3f}')

        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.set_title(f'{title}\nLeft vs Right Lane Changes')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sam_parameter_left_vs_right_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(
        f"📊 Intention comparison plot saved to: {os.path.join(output_dir, 'sam_parameter_left_vs_right_comparison.png')}")


def plot_box_plots(lc_samples: List[Dict], output_dir: str) -> None:
    """Create box plots for all parameters"""

    if not lc_samples:
        return

    # Prepare data for box plots
    W_values = [s['W'] for s in lc_samples]
    D_values = [s['D'] for s in lc_samples]
    v0_values = [s['v0'] for s in lc_samples]
    delta_vx_values = [s['Delta_Vx'] for s in lc_samples]

    # Create box plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    # W box plot
    axes[0].boxplot(W_values, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[0].set_ylabel('W - Lateral Displacement (m)')
    axes[0].set_title(f'W Distribution\n(N={len(W_values)})')
    axes[0].grid(True, alpha=0.3)

    # D box plot
    axes[1].boxplot(D_values, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    axes[1].set_ylabel('D - Duration (s)')
    axes[1].set_title(f'D Distribution\n(N={len(D_values)})')
    axes[1].grid(True, alpha=0.3)

    # v0 box plot
    axes[2].boxplot(v0_values, patch_artist=True, boxprops=dict(facecolor='plum'))
    axes[2].set_ylabel('v0 - Initial Lateral Velocity (m/s)')
    axes[2].set_title(f'v0 Distribution\n(N={len(v0_values)})')
    axes[2].grid(True, alpha=0.3)

    # Delta_Vx box plot
    axes[3].boxplot(delta_vx_values, patch_artist=True, boxprops=dict(facecolor='lightsalmon'))
    axes[3].set_ylabel('Delta_Vx - Longitudinal Velocity Change (m/s)')
    axes[3].set_title(f'Delta_Vx Distribution\n(N={len(delta_vx_values)})')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sam_parameter_box_plots.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Box plots saved to: {os.path.join(output_dir, 'sam_parameter_box_plots.png')}")


def plot_correlation_matrix(lc_samples: List[Dict], output_dir: str) -> None:
    """Create correlation matrix heatmap for all parameters"""

    if not lc_samples:
        return

    # Create DataFrame
    df = pd.DataFrame({
        'W': [s['W'] for s in lc_samples],
        'D': [s['D'] for s in lc_samples],
        'v0': [s['v0'] for s in lc_samples],
        'Delta_Vx': [s['Delta_Vx'] for s in lc_samples],
        'Intention': [s['intention'] for s in lc_samples]
    })

    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title(f'SAM Parameter Correlation Matrix\n(N={len(lc_samples)} LC samples)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sam_parameter_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Correlation matrix saved to: {os.path.join(output_dir, 'sam_parameter_correlation_matrix.png')}")


def plot_scatter_plots(lc_samples: List[Dict], output_dir: str) -> None:
    """Create scatter plots to show relationships between parameters"""

    if not lc_samples:
        return

    # Extract data
    W_values = [s['W'] for s in lc_samples]
    D_values = [s['D'] for s in lc_samples]
    v0_values = [s['v0'] for s in lc_samples]
    delta_vx_values = [s['Delta_Vx'] for s in lc_samples]
    intentions = [s['intention'] for s in lc_samples]

    # Create scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Color by intention
    colors = ['red' if intent == 1 else 'blue' for intent in intentions]

    # W vs D
    axes[0, 0].scatter(W_values, D_values, c=colors, alpha=0.6)
    axes[0, 0].set_xlabel('W - Lateral Displacement (m)')
    axes[0, 0].set_ylabel('D - Duration (s)')
    axes[0, 0].set_title('W vs D')
    axes[0, 0].grid(True, alpha=0.3)

    # W vs v0
    axes[0, 1].scatter(W_values, v0_values, c=colors, alpha=0.6)
    axes[0, 1].set_xlabel('W - Lateral Displacement (m)')
    axes[0, 1].set_ylabel('v0 - Initial Lateral Velocity (m/s)')
    axes[0, 1].set_title('W vs v0')
    axes[0, 1].grid(True, alpha=0.3)

    # W vs Delta_Vx
    axes[0, 2].scatter(W_values, delta_vx_values, c=colors, alpha=0.6)
    axes[0, 2].set_xlabel('W - Lateral Displacement (m)')
    axes[0, 2].set_ylabel('Delta_Vx - Longitudinal Velocity Change (m/s)')
    axes[0, 2].set_title('W vs Delta_Vx')
    axes[0, 2].grid(True, alpha=0.3)

    # D vs v0
    axes[1, 0].scatter(D_values, v0_values, c=colors, alpha=0.6)
    axes[1, 0].set_xlabel('D - Duration (s)')
    axes[1, 0].set_ylabel('v0 - Initial Lateral Velocity (m/s)')
    axes[1, 0].set_title('D vs v0')
    axes[1, 0].grid(True, alpha=0.3)

    # D vs Delta_Vx
    axes[1, 1].scatter(D_values, delta_vx_values, c=colors, alpha=0.6)
    axes[1, 1].set_xlabel('D - Duration (s)')
    axes[1, 1].set_ylabel('Delta_Vx - Longitudinal Velocity Change (m/s)')
    axes[1, 1].set_title('D vs Delta_Vx')
    axes[1, 1].grid(True, alpha=0.3)

    # v0 vs Delta_Vx
    axes[1, 2].scatter(v0_values, delta_vx_values, c=colors, alpha=0.6)
    axes[1, 2].set_xlabel('v0 - Initial Lateral Velocity (m/s)')
    axes[1, 2].set_ylabel('Delta_Vx - Longitudinal Velocity Change (m/s)')
    axes[1, 2].set_title('v0 vs Delta_Vx')
    axes[1, 2].grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.6, label='Left Change (1)'),
                       Patch(facecolor='blue', alpha=0.6, label='Right Change (2)')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sam_parameter_scatter_plots.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Scatter plots saved to: {os.path.join(output_dir, 'sam_parameter_scatter_plots.png')}")


def print_summary_statistics(overall_stats: pd.DataFrame, left_stats: pd.DataFrame, right_stats: pd.DataFrame) -> None:
    """Print comprehensive summary statistics"""

    print("\n" + "=" * 80)
    print("SAM PARAMETER DISTRIBUTION ANALYSIS")
    print("=" * 80)

    print(f"\n📊 OVERALL STATISTICS:")
    print("-" * 80)
    if not overall_stats.empty:
        for _, row in overall_stats.iterrows():
            print(f"{row['Parameter']:<35}")
            print(f"  Count: {row['Count']:<8.0f} | Mean: {row['Mean']:<8.3f} | Std: {row['Std']:<8.3f}")
            print(f"  Range: [{row['Min']:<8.3f}, {row['Max']:<8.3f}] | IQR: [{row['Q25']:<8.3f}, {row['Q75']:<8.3f}]")
            print(f"  Skewness: {row['Skewness']:<8.3f} | Kurtosis: {row['Kurtosis']:<8.3f}")
            print()

    print(f"\n📊 LEFT vs RIGHT LANE CHANGE COMPARISON:")
    print("-" * 80)

    if not left_stats.empty and not right_stats.empty:
        # Combine for easier comparison
        combined_stats = pd.concat([left_stats, right_stats], ignore_index=True)

        # Define parameter descriptions
        param_descriptions = {
            'W': 'Lateral Displacement (m)',
            'D': 'Duration (s)',
            'v0': 'Initial Lateral Velocity (m/s)',
            'Delta_Vx': 'Longitudinal Velocity Change (m/s)'
        }

        for param in ['W', 'D', 'v0', 'Delta_Vx']:
            param_data = combined_stats[combined_stats['Parameter'] == param]
            if len(param_data) == 2:
                left_row = param_data[param_data['Group'] == 'Left Change (1)'].iloc[0]
                right_row = param_data[param_data['Group'] == 'Right Change (2)'].iloc[0]

                param_desc = param_descriptions[param]
                print(f"{param} - {param_desc}:")
                print(
                    f"  Left Change:  Count={left_row['Count']:<4.0f} | Mean={left_row['Mean']:<8.3f} | Std={left_row['Std']:<8.3f} | Range=[{left_row['Min']:<8.3f}, {left_row['Max']:<8.3f}]")
                print(
                    f"  Right Change: Count={right_row['Count']:<4.0f} | Mean={right_row['Mean']:<8.3f} | Std={right_row['Std']:<8.3f} | Range=[{right_row['Min']:<8.3f}, {right_row['Max']:<8.3f}]")

                # Calculate difference
                mean_diff = abs(left_row['Mean'] - right_row['Mean'])
                print(f"  Difference: |Mean_Left - Mean_Right| = {mean_diff:.3f}")
                print()


def identify_outliers(lc_samples: List[Dict]) -> None:
    """Identify and report potential outliers"""

    if not lc_samples:
        return

    print(f"\n🔍 OUTLIER ANALYSIS:")
    print("-" * 50)

    parameters = {
        'W': [s['W'] for s in lc_samples],
        'D': [s['D'] for s in lc_samples],
        'v0': [s['v0'] for s in lc_samples],
        'Delta_Vx': [s['Delta_Vx'] for s in lc_samples]
    }

    for param_name, values in parameters.items():
        values = np.array(values)

        # Calculate IQR-based outlier bounds
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find outliers
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        outlier_percentage = len(outliers) / len(values) * 100

        print(f"{param_name}:")
        print(f"  IQR bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
        print(f"  Outliers: {len(outliers)}/{len(values)} ({outlier_percentage:.1f}%)")
        if len(outliers) > 0:
            print(f"  Outlier range: [{np.min(outliers):.3f}, {np.max(outliers):.3f}]")
        print()


def main():
    """Main execution function"""

    # Create output directory
    output_dir = "sam_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Using output directory: {output_dir}/")

    # File path
    sam_results_file = "complete_pal_predictions_old.json"

    print("Starting SAM Parameter Distribution Analysis...")
    print(f"SAM results file: {sam_results_file}")

    try:
        # Load data
        lc_samples = load_sam_results(sam_results_file)

        if not lc_samples:
            print("❌ No LC samples with complete SAM parameters found!")
            return

        # Calculate statistics
        overall_stats = calculate_parameter_statistics(lc_samples)
        left_stats, right_stats = calculate_intention_statistics(lc_samples)

        # Print summary
        print_summary_statistics(overall_stats, left_stats, right_stats)

        # Identify outliers
        identify_outliers(lc_samples)

        # Generate plots
        print(f"\n📊 Generating distribution plots...")
        plot_parameter_distributions(lc_samples, output_dir)
        plot_intention_comparison(lc_samples, output_dir)
        plot_box_plots(lc_samples, output_dir)
        plot_correlation_matrix(lc_samples, output_dir)
        plot_scatter_plots(lc_samples, output_dir)

        print(f"\n🎉 SAM Parameter Distribution Analysis Complete!")
        print(f"📁 All outputs saved to: {output_dir}/")
        print("📊 Generated plots:")
        print("  - sam_parameter_distributions_all.png (Overall histograms)")
        print("  - sam_parameter_left_vs_right_comparison.png (Left vs Right comparison)")
        print("  - sam_parameter_box_plots.png (Box plots)")
        print("  - sam_parameter_correlation_matrix.png (Correlation heatmap)")
        print("  - sam_parameter_scatter_plots.png (Scatter plot matrix)")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure the input file exists in the current directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()