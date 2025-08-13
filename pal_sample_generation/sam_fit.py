import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import os
import glob

warnings.filterwarnings('ignore')


# -------------------- MODEL DEFINITION --------------------
def sam_model_with_v0(t, W, D, v0):
    """
    Modified Sinusoidal Acceleration Model (SAM) with known initial velocity v0
    Based on standard SAM but adjusted for non-zero initial velocity

    Derivation:
    - Standard SAM assumes v(0) = 0, but post-boundary trajectories have v(0) = v0
    - Modified to ensure: y(0) = 0, v(0) = v0, y(D) = W

    Mathematical formulation:
    y(t) = (W/D)*t + ((v0*D - W)/(2π))*sin(2πt/D)

    Parameters:
    - W: Total lateral displacement
    - D: Duration of lane change
    - v0: Initial lateral velocity (extracted from trajectory data at boundary)
    - t: Time since lane change start
    """
    return (W / D) * t + ((v0 * D - W) / (2 * np.pi)) * np.sin(2 * np.pi * t / D)


# -------------------- FITTING FUNCTION --------------------
def fit_sam_to_trajectory(trajectory_data, directions, traj_idx=0, filename=""):
    """
    Fits the SAM model with initial velocity to the post-lane-change trajectory data.
    """
    # Basic validation of the trajectory data structure
    if len(trajectory_data) != 200:
        return None

    # Use the provided direction labels (1=LEFT, 2=RIGHT, 0=LANE_KEEPING)
    direction = directions[100] if len(directions) > 100 else directions[0]
    if direction not in [1, 2]:
        return None  # Skip lane keeping trajectories (direction = 0)

    # Extract relevant data columns using correct feature indices
    delta_y = np.array([frame[2] for frame in trajectory_data])  # Index 2: ΔY (difference from lane center)
    y_velocities = np.array([frame[3] for frame in trajectory_data])  # Index 3: Vy (Y velocity)

    # Isolate the post-lane-change segment (the last 4 seconds / 100 frames)
    boundary_frame = 100
    v0 = y_velocities[boundary_frame]  # Extract initial velocity at boundary crossing
    post_boundary_y = delta_y[boundary_frame:]
    delta_y_at_boundary = delta_y[boundary_frame]

    # Normalize the data for fitting: time starts at 0, displacement starts at 0
    y_relative = post_boundary_y - delta_y_at_boundary
    t_post = np.arange(len(y_relative)) / 25.0  # Time in seconds (assuming 25 Hz)
    post_displacement = y_relative[-1]

    # Filter out trajectories with very small lateral movement
    if abs(post_displacement) < 0.3:
        return None

    # --- Fit the SAM Model with v0 ---
    try:
        # Initial guess for [W, D, v0]: Use actual displacement for W, 4.0s for D, extracted v0
        p0_sam = [post_displacement, 4.0, v0]

        # Define bounds to guide the fitter for [W, D, v0]
        v0_tolerance = max(abs(v0) * 1, 1)  # Allow ±100% variation or minimum 1 m/s

        if post_displacement > 0:  # Left lane change (positive y)
            bounds_sam = (
                [post_displacement * 0, 1.0, v0 - v0_tolerance],
                [post_displacement * 3, 12.0, v0 + v0_tolerance]
            )
        else:  # Right lane change (negative y)
            bounds_sam = (
                [post_displacement * 3, 1.0, v0 - v0_tolerance],
                [post_displacement * 0, 12.0, v0 + v0_tolerance]
            )

        # Perform the curve fitting using the correct SAM model
        popt_sam, _ = curve_fit(sam_model_with_v0, t_post, y_relative,
                                p0=p0_sam, bounds=bounds_sam, maxfev=10000)

        W_fitted, D_fitted, v0_fitted = popt_sam
        y_fitted_sam = sam_model_with_v0(t_post, W_fitted, D_fitted, v0_fitted)

        # Calculate performance metrics
        r2 = r2_score(y_relative, y_fitted_sam)
        rmse = np.sqrt(mean_squared_error(y_relative, y_fitted_sam))

        # Only consider fits with a good R² value
        if r2 < 0.85:
            return None

        # Store results in a dictionary - use the provided direction labels directly
        results = {
            'traj_idx': traj_idx, 'filename': filename,
            'direction': "LEFT" if direction == 1 else "RIGHT",  # Direct label mapping
            'direction_label': direction,  # Store the original numeric label too
            'v0_extracted': v0, 'v0_fitted': v0_fitted,
            'post_displacement': post_displacement,
            'y_relative': y_relative, 't_post': t_post,
            'sam_W': W_fitted, 'sam_D': D_fitted, 'sam_v0': v0_fitted,
            'sam_fitted': y_fitted_sam,
            'sam_r2': r2, 'sam_rmse': rmse
        }
        return results

    except Exception as e:
        # If fitting fails for any reason, return None
        return None


# -------------------- VISUALIZATION --------------------
def visualize_best_fits(fits, n_plots=6):
    """
    Visualizes the top N best fits for the SAM model with v0.
    """
    if not fits:
        print("No successful fits to visualize.")
        return

    # Sort fits by R² score in descending order to show the best ones
    fits.sort(key=lambda x: x['sam_r2'], reverse=True)
    n_vis = min(n_plots, len(fits))

    fig, axes = plt.subplots(2, n_vis, figsize=(n_vis * 3.5, 7), squeeze=False)

    for i in range(n_vis):
        fit = fits[i]

        # --- Plot 1: Trajectory Fit ---
        ax1 = axes[0, i]
        ax1.plot(fit['t_post'], fit['y_relative'], 'b.', alpha=0.6, markersize=4, label='Actual Data')
        ax1.plot(fit['t_post'], fit['sam_fitted'], 'r-', linewidth=2.5, label='SAM w/ v₀ Fit')

        title_text = f"{fit['direction']} | R²={fit['sam_r2']:.3f}"
        ax1.set_title(title_text, fontsize=10)
        ax1.set_ylabel('Relative y (m)')
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.legend(fontsize=8)

        # --- Plot 2: Residuals ---
        ax2 = axes[1, i]
        residuals = fit['y_relative'] - fit['sam_fitted']
        ax2.plot(fit['t_post'], residuals, 'r-', alpha=0.8, label=f'RMSE={fit["sam_rmse"]:.3f} m')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Residual (m)')
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.legend(fontsize=8)

        # Add a text box with key parameters including v0 comparison
        param_text = (f'v₀ ext={fit["v0_extracted"]:.2f} m/s\n'
                      f'v₀ fit={fit["v0_fitted"]:.2f} m/s\n'
                      f'Δy={fit["post_displacement"]:.2f} m\n'
                      f'W={fit["sam_W"]:.2f}, D={fit["sam_D"]:.2f}')
        ax2.text(0.05, 0.95, param_text, transform=ax2.transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Top {n_vis} SAM Model with v₀ Fits Across All Files', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


# -------------------- MAIN ANALYSIS SCRIPT --------------------
def run_analysis_on_all_files(data_directory):
    """
    Main function to load all data files, run the SAM fit with v0, and report results.
    """
    # Find all result*.pickle files in the specified directory
    pickle_files = sorted(glob.glob(os.path.join(data_directory, 'result*.pickle')))

    if not pickle_files:
        print(f"❌ Error: No pickle files found in directory: {data_directory}")
        return

    print(f"Found {len(pickle_files)} data files to analyze.\n")

    all_fits = []
    total_lc_trajectories = 0

    # Loop through each file
    for filepath in pickle_files:
        filename = os.path.basename(filepath)
        print(f"--- Processing {filename} ---")

        try:
            with open(filepath, 'rb') as f:
                trajectories_data = pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"  Could not read or load {filename}: {e}")
            continue

        file_lc_count = 0
        file_fit_count = 0

        # Loop through each trajectory in the file
        for i, (traj_data, directions) in enumerate(trajectories_data):
            # Check if it's a lane change trajectory
            if 1 in set(directions) or 2 in set(directions):
                file_lc_count += 1
                fit_result = fit_sam_to_trajectory(traj_data, directions, traj_idx=i, filename=filename)
                if fit_result:
                    all_fits.append(fit_result)
                    file_fit_count += 1

        total_lc_trajectories += file_lc_count
        print(f"  Found {file_lc_count} LC trajectories, successfully fitted {file_fit_count}.")

    print("\n" + "=" * 60)
    print("           OVERALL ANALYSIS COMPLETE")
    print("=" * 60)

    if not all_fits:
        print("\nNo trajectories could be successfully fitted with the SAM model across all files.")
        return

    # --- Final Report ---
    num_successful_fits = len(all_fits)
    success_rate = (num_successful_fits / total_lc_trajectories) * 100 if total_lc_trajectories > 0 else 0

    print(f"Total Lane Change Trajectories Found: {total_lc_trajectories}")
    print(f"Total Successful SAM w/ v₀ Fits (R² > 0.85): {num_successful_fits}")
    print(f"Overall Model Success Rate: {success_rate:.2f}%\n")

    # Performance metrics
    all_r2 = [f['sam_r2'] for f in all_fits]
    all_rmse = [f['sam_rmse'] for f in all_fits]

    print("SAM Model with v₀ Performance Statistics:")
    print(f"  Mean R²:   {np.mean(all_r2):.4f} (± {np.std(all_r2):.4f})")
    print(f"  Mean RMSE: {np.mean(all_rmse):.4f} m (± {np.std(all_rmse):.4f})\n")

    # Parameter analysis including v0
    all_W = [f['sam_W'] for f in all_fits]
    all_D = [f['sam_D'] for f in all_fits]
    all_v0_extracted = [f['v0_extracted'] for f in all_fits]
    all_v0_fitted = [f['v0_fitted'] for f in all_fits]

    print("Fitted Parameter Statistics:")
    print(f"  Mean W (Displacement): {np.mean(all_W):.3f} m (± {np.std(all_W):.3f})")
    print(f"  Mean D (Duration):     {np.mean(all_D):.3f} s (± {np.std(all_D):.3f})")
    print(f"  Mean v₀ extracted:     {np.mean(all_v0_extracted):.3f} m/s (± {np.std(all_v0_extracted):.3f})")
    print(f"  Mean v₀ fitted:        {np.mean(all_v0_fitted):.3f} m/s (± {np.std(all_v0_fitted):.3f})")

    # v0 correlation analysis
    v0_diff = np.array(all_v0_fitted) - np.array(all_v0_extracted)
    print(f"  v₀ difference (fit-ext): {np.mean(v0_diff):.3f} m/s (± {np.std(v0_diff):.3f})\n")

    # Direction analysis
    left_fits = [f for f in all_fits if f['direction'] == 'LEFT']
    right_fits = [f for f in all_fits if f['direction'] == 'RIGHT']
    print("Direction-specific Analysis:")
    print(f"  Left lane changes:  {len(left_fits)} ({len(left_fits) / len(all_fits) * 100:.1f}%)")
    print(f"  Right lane changes: {len(right_fits)} ({len(right_fits) / len(all_fits) * 100:.1f}%)")

    if left_fits:
        left_r2 = np.mean([f['sam_r2'] for f in left_fits])
        print(f"  Left LC mean R²: {left_r2:.4f}")
    if right_fits:
        right_r2 = np.mean([f['sam_r2'] for f in right_fits])
        print(f"  Right LC mean R²: {right_r2:.4f}")

    # --- Conclusion ---
    print("\n--- Conclusion ---")
    if np.mean(all_r2) > 0.95 and success_rate > 50:
        print("✅ The SAM model with v₀ demonstrates an EXCELLENT fit for a majority of the observed")
        print("   lane change deceleration profiles, consistently yielding high R² values.")
    elif np.mean(all_r2) > 0.90 and success_rate > 30:
        print("✅ The SAM model with v₀ provides a GOOD FIT for a significant portion of the trajectories.")
        print("   It appears to be a robust model for post-boundary lane change motion.")
    else:
        print("⚠️ The SAM model with v₀ provides a moderate fit. While successful in many cases,")
        print(f"   its applicability may be limited, as shown by the success rate of {success_rate:.2f}%.")

    print(f"\n📊 Key insight: The model incorporating initial velocity v₀ should better capture")
    print(f"   the non-zero lateral momentum that vehicles have when crossing the lane boundary.")

    # Visualize the best results from the entire dataset
    visualize_best_fits(all_fits, n_plots=6)


if __name__ == "__main__":
    # IMPORTANT: Set the path to the directory containing your pickle files
    data_folder_path = "../output_4sbefore_4safter/"

    run_analysis_on_all_files(data_folder_path)