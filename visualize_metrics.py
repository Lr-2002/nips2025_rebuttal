#!/usr/bin/env python3
"""
Visualization script for trajectory and gripper stability metrics.
Creates plots to help understand the metrics and their behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from metric import TrajectoryStabilityCalculator, GripperStabilityCalculator
from data_loader import RebuttalDataLoader

def plot_trajectory_analysis():
    """Create visualizations for trajectory stability analysis."""
    
    # Create sample trajectories
    T = 200
    t = np.linspace(0, 20, T)
    
    # Smooth trajectory
    smooth_actions = np.zeros((T, 7))
    smooth_actions[:, 0] = 0.5 * np.sin(0.2 * t)
    smooth_actions[:, 1] = 0.3 * np.cos(0.15 * t)
    smooth_actions[:, 2] = 0.1 * t / 20
    
    # Noisy trajectory (VLA explosion)
    noisy_actions = smooth_actions.copy()
    explosion_times = [50, 120, 180]
    for t_exp in explosion_times:
        if t_exp < T:
            noisy_actions[t_exp:t_exp+5, :3] += np.random.uniform(-2, 2, (min(5, T-t_exp), 3))
    
    # Calculate velocities for visualization
    smooth_vel = np.linalg.norm(np.diff(smooth_actions[:, :3], axis=0), axis=1)
    noisy_vel = np.linalg.norm(np.diff(noisy_actions[:, :3], axis=0), axis=1)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Trajectory Stability Analysis Visualization', fontsize=16)
    
    # Position trajectories
    axes[0, 0].plot(smooth_actions[:, 0], smooth_actions[:, 1], 'b-', label='Smooth', linewidth=2)
    axes[0, 0].plot(noisy_actions[:, 0], noisy_actions[:, 1], 'r-', label='Noisy (VLA Explosion)', linewidth=2)
    axes[0, 0].set_title('XY Position Trajectories')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity over time
    axes[0, 1].plot(smooth_vel, 'b-', label='Smooth', linewidth=2)
    axes[0, 1].plot(noisy_vel, 'r-', label='Noisy (VLA Explosion)', linewidth=2)
    axes[0, 1].set_title('Velocity Magnitude Over Time')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Velocity Magnitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mark explosion points
    for t_exp in explosion_times:
        if t_exp < len(noisy_vel):
            axes[0, 1].axvline(x=t_exp, color='red', linestyle='--', alpha=0.7, label='Explosion' if t_exp == explosion_times[0] else "")
    
    # Z position over time
    axes[1, 0].plot(smooth_actions[:, 2], 'b-', label='Smooth', linewidth=2)
    axes[1, 0].plot(noisy_actions[:, 2], 'r-', label='Noisy (VLA Explosion)', linewidth=2)
    axes[1, 0].set_title('Z Position Over Time')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Z Position')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Calculate and display metrics
    calc = TrajectoryStabilityCalculator()
    smooth_metrics = calc.calculate_stability_metrics(smooth_actions)
    noisy_metrics = calc.calculate_stability_metrics(noisy_actions)
    
    # Metrics comparison
    metrics_names = ['Overall\nStability', 'Velocity\nSmoothness', 'Acceleration\nSmoothness', 'Jerk\nSmoothness']
    smooth_scores = [smooth_metrics.overall_stability_score, smooth_metrics.velocity_smoothness, 
                    smooth_metrics.acceleration_smoothness, smooth_metrics.jerk_smoothness]
    noisy_scores = [noisy_metrics.overall_stability_score, noisy_metrics.velocity_smoothness,
                   noisy_metrics.acceleration_smoothness, noisy_metrics.jerk_smoothness]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, smooth_scores, width, label='Smooth', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, noisy_scores, width, label='Noisy (VLA Explosion)', color='red', alpha=0.7)
    axes[1, 1].set_title('Stability Metrics Comparison')
    axes[1, 1].set_ylabel('Score (0-1)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    # Add threshold line for VLA explosion detection
    axes[1, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.8, label='VLA Explosion Threshold')
    
    plt.tight_layout()
    plt.savefig('/Users/lr-2002/project/reasoning_manipulation/rebuttal/trajectory_stability_analysis.png', dpi=300, bbox_inches='tight')
    print("Trajectory stability visualization saved to: trajectory_stability_analysis.png")

def plot_gripper_analysis():
    """Create visualizations for gripper stability analysis."""
    
    T = 100
    
    # Create smooth gripper control pattern
    smooth_gripper = np.zeros((T, 7))
    smooth_gripper[:, :3] = 0.1 * np.random.randn(T, 3)  # Small random motion
    smooth_gripper[:20, 6] = 1.0   # Open initially
    smooth_gripper[20:40, 6] = 0.0  # Close for grasp
    smooth_gripper[40:80, 6] = 0.0  # Keep closed
    smooth_gripper[80:, 6] = 1.0   # Open to release
    
    # Create erratic gripper control
    erratic_gripper = smooth_gripper.copy()
    for i in range(10, 90, 6):  # Every 6 frames
        erratic_gripper[i:i+2, 6] = 1.0 - erratic_gripper[i-1, 6]  # Flip state
    
    # Calculate arm velocities
    smooth_vel = np.linalg.norm(np.diff(smooth_gripper[:, :3], axis=0), axis=1)
    erratic_vel = np.linalg.norm(np.diff(erratic_gripper[:, :3], axis=0), axis=1)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Gripper Control Stability Analysis Visualization', fontsize=16)
    
    # Gripper states over time
    axes[0, 0].plot(smooth_gripper[:, 6], 'b-', label='Smooth Control', linewidth=3, marker='o', markersize=3)
    axes[0, 0].plot(erratic_gripper[:, 6], 'r-', label='Erratic Control', linewidth=2, marker='x', markersize=3)
    axes[0, 0].set_title('Gripper State Over Time')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Gripper State (0=Closed, 1=Open)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.1, 1.1)
    
    # Gripper state changes
    smooth_changes = np.abs(np.diff(smooth_gripper[:, 6]))
    erratic_changes = np.abs(np.diff(erratic_gripper[:, 6]))
    
    axes[0, 1].plot(smooth_changes, 'b-', label='Smooth Control', linewidth=2)
    axes[0, 1].plot(erratic_changes, 'r-', label='Erratic Control', linewidth=2)
    axes[0, 1].set_title('Gripper State Changes')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('|Change in Gripper State|')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Arm velocity (for coordination analysis)
    axes[1, 0].plot(smooth_vel, 'b-', label='Smooth Control', linewidth=2)
    axes[1, 0].plot(erratic_vel, 'r-', label='Erratic Control', linewidth=2)
    axes[1, 0].set_title('Arm Velocity (for Coordination Analysis)')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Velocity Magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Calculate and display gripper metrics
    gripper_calc = GripperStabilityCalculator()
    smooth_gripper_metrics = gripper_calc.calculate_gripper_stability(smooth_gripper)
    erratic_gripper_metrics = gripper_calc.calculate_gripper_stability(erratic_gripper)
    
    # Gripper metrics comparison
    gripper_metrics_names = ['Overall\nStability', 'Smoothness', 'Frequency', 'Coordination']
    smooth_gripper_scores = [smooth_gripper_metrics.overall_stability_score, smooth_gripper_metrics.smoothness_score,
                           smooth_gripper_metrics.frequency_score, smooth_gripper_metrics.coordination_score]
    erratic_gripper_scores = [erratic_gripper_metrics.overall_stability_score, erratic_gripper_metrics.smoothness_score,
                            erratic_gripper_metrics.frequency_score, erratic_gripper_metrics.coordination_score]
    
    x = np.arange(len(gripper_metrics_names))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, smooth_gripper_scores, width, label='Smooth Control', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, erratic_gripper_scores, width, label='Erratic Control', color='red', alpha=0.7)
    axes[1, 1].set_title('Gripper Stability Metrics Comparison')
    axes[1, 1].set_ylabel('Score (0-1)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(gripper_metrics_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    # Add threshold line for erratic gripper detection
    axes[1, 1].axhline(y=0.6, color='orange', linestyle='--', alpha=0.8, label='Erratic Gripper Threshold')
    
    # Add text annotations for change counts
    axes[1, 1].text(0.5, 0.9, f'Smooth: {smooth_gripper_metrics.total_gripper_changes} changes', 
                   transform=axes[1, 1].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 1].text(0.5, 0.85, f'Erratic: {erratic_gripper_metrics.total_gripper_changes} changes', 
                   transform=axes[1, 1].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    plt.tight_layout()
    plt.savefig('/Users/lr-2002/project/reasoning_manipulation/rebuttal/gripper_stability_analysis.png', dpi=300, bbox_inches='tight')
    print("Gripper stability visualization saved to: gripper_stability_analysis.png")

def plot_real_data_analysis():
    """Analyze and visualize real trajectory data."""
    
    print("\n=== Real Data Visualization ===")
    
    # Load real data
    loader = RebuttalDataLoader()
    if not loader.trajectories:
        print("No real trajectory data found for visualization.")
        return
    
    # Get the first trajectory
    traj_id, traj_data = next(iter(loader.trajectories.items()))
    actions = np.array(traj_data.actions)
    
    if len(actions) == 0:
        print("No actions found in trajectory data.")
        return
    
    print(f"Analyzing real trajectory: {traj_id}")
    print(f"Task: {traj_data.task_name}")
    print(f"Action shape: {actions.shape}")
    
    # Calculate metrics
    traj_calc = TrajectoryStabilityCalculator()
    gripper_calc = GripperStabilityCalculator()
    
    traj_metrics = traj_calc.calculate_stability_metrics(actions)
    gripper_metrics = gripper_calc.calculate_gripper_stability(actions)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Real Data Analysis: {traj_data.task_name}', fontsize=16)
    
    # Position trajectory
    axes[0, 0].plot(actions[:, 0], actions[:, 1], 'b-', linewidth=1, alpha=0.7)
    axes[0, 0].set_title('XY Position Trajectory')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity over time
    velocity = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)
    axes[0, 1].plot(velocity, 'g-', linewidth=1)
    axes[0, 1].set_title('Velocity Magnitude Over Time')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gripper state
    axes[0, 2].plot(actions[:, -1], 'r-', linewidth=2)
    axes[0, 2].set_title('Gripper State Over Time')
    axes[0, 2].set_xlabel('Time Step')
    axes[0, 2].set_ylabel('Gripper State')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Z position
    axes[1, 0].plot(actions[:, 2], 'purple', linewidth=1)
    axes[1, 0].set_title('Z Position Over Time')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Z Position')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Trajectory stability metrics
    traj_metrics_names = ['Overall', 'Velocity', 'Acceleration', 'Jerk']
    traj_scores = [traj_metrics.overall_stability_score, traj_metrics.velocity_smoothness,
                  traj_metrics.acceleration_smoothness, traj_metrics.jerk_smoothness]
    
    bars1 = axes[1, 1].bar(traj_metrics_names, traj_scores, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    axes[1, 1].set_title('Trajectory Stability Metrics')
    axes[1, 1].set_ylabel('Score (0-1)')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.8, label='VLA Explosion Threshold')
    
    # Add value labels on bars
    for bar, score in zip(bars1, traj_scores):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Gripper stability metrics
    gripper_metrics_names = ['Overall', 'Smoothness', 'Frequency', 'Coordination']
    gripper_scores = [gripper_metrics.overall_stability_score, gripper_metrics.smoothness_score,
                     gripper_metrics.frequency_score, gripper_metrics.coordination_score]
    
    bars2 = axes[1, 2].bar(gripper_metrics_names, gripper_scores, color=['purple', 'cyan', 'magenta', 'yellow'], alpha=0.7)
    axes[1, 2].set_title('Gripper Stability Metrics')
    axes[1, 2].set_ylabel('Score (0-1)')
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0.6, color='red', linestyle='--', alpha=0.8, label='Erratic Gripper Threshold')
    
    # Add value labels on bars
    for bar, score in zip(bars2, gripper_scores):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add summary text
    summary_text = f"""Real Data Analysis Summary:
    
Trajectory Stability: {traj_metrics.overall_stability_score:.3f}
{'⚠️ VLA ACTION EXPLOSION detected!' if traj_metrics.overall_stability_score < 0.5 else '✅ Stable trajectory'}

Gripper Stability: {gripper_metrics.overall_stability_score:.3f}
{'⚠️ ERRATIC GRIPPER CONTROL detected!' if gripper_metrics.overall_stability_score < 0.6 else '✅ Stable gripper control'}

Gripper Changes: {gripper_metrics.total_gripper_changes}/{gripper_metrics.expected_changes}
"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/lr-2002/project/reasoning_manipulation/rebuttal/real_data_analysis.png', dpi=300, bbox_inches='tight')
    print("Real data analysis visualization saved to: real_data_analysis.png")

if __name__ == "__main__":
    print("Creating trajectory and gripper stability visualizations...")
    
    try:
        # Create trajectory analysis plots
        plot_trajectory_analysis()
        
        # Create gripper analysis plots  
        plot_gripper_analysis()
        
        # Analyze real data
        plot_real_data_analysis()
        
        print("\n✅ All visualizations completed successfully!")
        print("\nGenerated files:")
        print("- trajectory_stability_analysis.png")
        print("- gripper_stability_analysis.png") 
        print("- real_data_analysis.png")
        
    except ImportError as e:
        print(f"⚠️ Matplotlib not available: {e}")
        print("Install matplotlib to generate visualizations: pip install matplotlib")
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
