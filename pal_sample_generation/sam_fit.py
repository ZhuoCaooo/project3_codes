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
def sam_deceleration_model(t, W, D):
    """
    Flexible Half-Sine SAM Model for the post-boundary (deceleration) segment.
    This model fits both W (displacement) and D (duration) as independent parameters.
    The model uses the first quarter-cycle of a sine wave: y(t) = W * sin(πt / 2D)
    """
    if D <= 0:
        return np.inf  # Prevent division by zero or invalid domain
    return W * np.sin((np.pi * t) / (2 * D))


# -------------------- FITTING FUNCTION --------------------
def fit_sam_to_trajectory(trajectory_data, directions, traj_idx=0, filename=""):
    """
    Fits the Half-Sine SAM model to the post-lane-change trajectory data.
    """
    # Basic validation of the trajectory data structure
    if len(trajectory_data) != 200:
        return None

    # Determine the direction of the lane change (1 for LEFT, 2 for RIGHT)
    direction = directions[100] if len(directions) > 100 else directions[0]
    if direction not in [1, 2]:
        return None  # Not a lane change trajectory

    # Extract relevant data columns
    delta_y = np.array([frame[2] for frame in trajectory_data])
    y_velocities = np.array([frame[3] for frame in trajectory_data])

    # Isolate the post-lane-change segment (the last 4 seconds / 100 frames)
    boundary_frame = 100
    v0 = y_velocities[boundary_frame]
    post_boundary_y = delta_y[boundary_frame:]
    delta_y_at_boundary = delta_y[boundary_frame]

    # Normalize the data for fitting: time starts at 0, displacement starts at 0
    y_relative = post_boundary_y - delta_y_at_boundary
    t_post = np.arange(len(y_relative)) / 25.0  # Time in seconds (assuming 25 Hz)
    post_displacement = y_relative[-1]

    # Filter out trajectories with very small lateral movement
    if abs(post_displacement) < 0.3:
        return None

    # --- Fit the SAM Model ---
    try:
        # Initial guess for [W, D]: Use actual displacement for W, and 4.0s for D
        p0_sam = [post_displacement, 4.0]

        # Define bounds to guide the fitter.
        # W (total displacement) should be reasonably close to the observed displacement.
        # D (duration) is the time to complete the maneuver, bounded between 2s and 8s.
        if post_displacement > 0:  # Left lane change
            bounds_sam = ([post_displacement * 0.5, 2.0], [post_displacement * 1.5, 8.0])
        else:  # Right lane change
            bounds_sam = ([post_displacement * 1.5, 2.0], [post_displacement * 0.5, 8.0])

        # Perform the curve fitting
        popt_sam, _ = curve_fit(sam_deceleration_model, t_post, y_relative,
                                p0=p0_sam, bounds=bounds_sam, maxfev=10000)

        W_fitted, D_fitted = popt_sam
        y_fitted_sam = sam_deceleration_model(t_post, W_fitted, D_fitted)

        # Calculate performance metrics
        r2 = r2_score(y_relative, y_fitted_sam)
        rmse = np.sqrt(mean_squared_error(y_relative, y_fitted_sam))

        # Only consider fits with a good R² value
        if r2 < 0.85:
            return None

        # Store results in a dictionary
        results = {
            'traj_idx': traj_idx, 'filename': filename,
            'direction': "LEFT" if direction == 1 else "RIGHT",
            'v0': v0, 'post_displacement': post_displacement,
            'y_relative': y_relative, 't_post': t_post,
            'sam_W': W_fitted, 'sam_D': D_fitted,
            'sam_fitted': y_fitted_sam,
            'sam_r2': r2, 'sam_rmse': rmse
        }
        return results

    except Exception:
        # If fitting fails for any reason, return None
        return None


# -------------------- VISUALIZATION --------------------
def visualize_best_fits(fits, n_plots=6):
    """
    Visualizes the top N best fits for the SAM model.
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
        ax1.plot(fit['t_post'], fit['sam_fitted'], 'r-', linewidth=2.5, label='SAM Fit')

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

        # Add a text box with key parameters
        param_text = f'v₀={fit["v0"]:.2f} m/s\nΔy={fit["post_displacement"]:.2f} m\nW={fit["sam_W"]:.2f}, D={fit["sam_D"]:.2f}'
        ax2.text(0.05, 0.95, param_text, transform=ax2.transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Top {n_vis} Half-Sine SAM Model Fits Across All Files', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


# -------------------- MAIN ANALYSIS SCRIPT --------------------
def run_analysis_on_all_files(data_directory):
    """
    Main function to load all data files, run the SAM fit, and report results.
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
    print(f"Total Successful SAM Fits (R² > 0.85): {num_successful_fits}")
    print(f"Overall Model Success Rate: {success_rate:.2f}%\n")

    # Performance metrics
    all_r2 = [f['sam_r2'] for f in all_fits]
    all_rmse = [f['sam_rmse'] for f in all_fits]

    print("SAM Model Performance Statistics:")
    print(f"  Mean R²:   {np.mean(all_r2):.4f} (± {np.std(all_r2):.4f})")
    print(f"  Mean RMSE: {np.mean(all_rmse):.4f} m (± {np.std(all_rmse):.4f})\n")

    # Parameter analysis
    all_W = [f['sam_W'] for f in all_fits]
    all_D = [f['sam_D'] for f in all_fits]
    print("Fitted Parameter Statistics:")
    print(f"  Mean W (Amplitude): {np.mean(all_W):.3f} m (± {np.std(all_W):.3f})")
    print(f"  Mean D (Duration):  {np.mean(all_D):.3f} s (± {np.std(all_D):.3f})\n")

    # --- Conclusion ---
    print("--- Conclusion ---")
    if np.mean(all_r2) > 0.95 and success_rate > 50:
        print("✅ The Half-Sine SAM demonstrates an EXCELLENT fit for a majority of the observed")
        print("   lane change deceleration profiles, consistently yielding high R² values.")
    elif np.mean(all_r2) > 0.90 and success_rate > 30:
        print("✅ The Half-Sine SAM provides a a NICE FIT for a significant portion of the trajectories.")
        print("   It appears to be a robust model for this type of motion.")
    else:
        print("⚠️ The Half-Sine SAM provides a moderate fit. While successful in many cases,")
        print(f"   its applicability may be limited, as shown by the success rate of {success_rate:.2f}%.")

    # Visualize the best results from the entire dataset
    visualize_best_fits(all_fits, n_plots=6)


if __name__ == "__main__":
    # IMPORTANT: Set the path to the directory containing your pickle files
    data_folder_path = "../output_4sbefore_4safter/"

    run_analysis_on_all_files(data_folder_path)