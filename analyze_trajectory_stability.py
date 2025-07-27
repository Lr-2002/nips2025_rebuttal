#!/usr/bin/env python3
"""
Analyze trajectory stability for COIN benchmark data using the implemented metrics.
This script demonstrates how to use the trajectory stability metrics on real data.
"""

import numpy as np
from data_loader import RebuttalDataLoader
from metric import TrajectoryStabilityCalculator, calculate_action_smoothness_score
from metric import GripperStabilityCalculator, calculate_gripper_stability_score
import json

def analyze_trajectory_stability():
    """Analyze stability metrics for all loaded trajectories."""
    
    # Load trajectory data
    print("Loading trajectory data...")
    loader = RebuttalDataLoader()
    
    if not loader.trajectories:
        print("No trajectories found!")
        return
    
    # Initialize stability calculators
    traj_calculator = TrajectoryStabilityCalculator(dt=0.1)  # Assuming 10Hz control
    gripper_calculator = GripperStabilityCalculator()
    
    print("\n=== Trajectory Stability Analysis ===")
    
    results = {}
    
    for traj_id, traj_data in loader.trajectories.items():
        print(f"\nAnalyzing trajectory: {traj_id}")
        print(f"Task: {traj_data.task_name}")
        
        if len(traj_data.actions) == 0:
            print("  No actions found, skipping...")
            continue
        
        # Convert actions to numpy array
        actions = np.array(traj_data.actions)
        print(f"  Action shape: {actions.shape}")
        
        # Calculate stability metrics
        try:
            traj_stability_metrics = traj_calculator.calculate_stability_metrics(actions)
            gripper_stability_metrics = gripper_calculator.calculate_gripper_stability(actions)
            
            # Store results
            results[traj_id] = {
                'task_name': traj_data.task_name,
                'action_count': len(traj_data.actions),
                'traj_stability_metrics': traj_stability_metrics,
                'gripper_stability_metrics': gripper_stability_metrics
            }
            
            # Print trajectory stability results
            print(f"  Trajectory Stability Score: {traj_stability_metrics.overall_stability_score:.4f}")
            print(f"    Velocity Smoothness: {traj_stability_metrics.velocity_smoothness:.4f}")
            print(f"    Acceleration Smoothness: {traj_stability_metrics.acceleration_smoothness:.4f}")
            print(f"    Jerk Smoothness: {traj_stability_metrics.jerk_smoothness:.4f}")
            print(f"    Position Stability: {traj_stability_metrics.position_stability:.4f}")
            
            # Print gripper stability results
            print(f"  Gripper Stability Score: {gripper_stability_metrics.overall_stability_score:.4f}")
            print(f"    Smoothness: {gripper_stability_metrics.smoothness_score:.4f}")
            print(f"    Frequency: {gripper_stability_metrics.frequency_score:.4f}")
            print(f"    Coordination: {gripper_stability_metrics.coordination_score:.4f}")
            print(f"    Gripper Changes: {gripper_stability_metrics.total_gripper_changes}/{gripper_stability_metrics.expected_changes}")
            

            # Detect problematic behaviors
            if traj_stability_metrics.overall_stability_score < 0.5:
                print(f"    ⚠️  VLA ACTION EXPLOSION detected!")
            if gripper_stability_metrics.overall_stability_score < 0.6:
                print(f"    ⚠️  ERRATIC GRIPPER CONTROL detected!")

            
        except Exception as e:
            print(f"  Error calculating metrics: {e}")
            continue
    
    # Summary analysis
    if results:
        print("\n=== Summary Analysis ===")
        
        # Calculate average trajectory stability metrics
        traj_overall_scores = [r['traj_stability_metrics'].overall_stability_score for r in results.values()]
        velocity_scores = [r['traj_stability_metrics'].velocity_smoothness for r in results.values()]
        acceleration_scores = [r['traj_stability_metrics'].acceleration_smoothness for r in results.values()]
        jerk_scores = [r['traj_stability_metrics'].jerk_smoothness for r in results.values()]
        
        # Calculate average gripper stability metrics
        gripper_overall_scores = [r['gripper_stability_metrics'].overall_stability_score for r in results.values()]
        gripper_smoothness_scores = [r['gripper_stability_metrics'].smoothness_score for r in results.values()]
        gripper_frequency_scores = [r['gripper_stability_metrics'].frequency_score for r in results.values()]
        gripper_coordination_scores = [r['gripper_stability_metrics'].coordination_score for r in results.values()]
        
        print(f"Trajectory Stability Averages:")
        print(f"  Overall: {np.mean(traj_overall_scores):.4f} ± {np.std(traj_overall_scores):.4f}")
        print(f"  Velocity Smoothness: {np.mean(velocity_scores):.4f} ± {np.std(velocity_scores):.4f}")
        print(f"  Acceleration Smoothness: {np.mean(acceleration_scores):.4f} ± {np.std(acceleration_scores):.4f}")
        print(f"  Jerk Smoothness: {np.mean(jerk_scores):.4f} ± {np.std(jerk_scores):.4f}")
        
        print(f"\nGripper Stability Averages:")
        print(f"  Overall: {np.mean(gripper_overall_scores):.4f} ± {np.std(gripper_overall_scores):.4f}")
        print(f"  Smoothness: {np.mean(gripper_smoothness_scores):.4f} ± {np.std(gripper_smoothness_scores):.4f}")
        print(f"  Frequency: {np.mean(gripper_frequency_scores):.4f} ± {np.std(gripper_frequency_scores):.4f}")
        print(f"  Coordination: {np.mean(gripper_coordination_scores):.4f} ± {np.std(gripper_coordination_scores):.4f}")
        
        # Find best and worst trajectories for both metrics
        best_traj_id = max(results.keys(), key=lambda k: results[k]['traj_stability_metrics'].overall_stability_score)
        worst_traj_id = min(results.keys(), key=lambda k: results[k]['traj_stability_metrics'].overall_stability_score)
        best_gripper_id = max(results.keys(), key=lambda k: results[k]['gripper_stability_metrics'].overall_stability_score)
        worst_gripper_id = min(results.keys(), key=lambda k: results[k]['gripper_stability_metrics'].overall_stability_score)
        
        print(f"\nBest Trajectory Stability: {best_traj_id}")
        print(f"  Task: {results[best_traj_id]['task_name']}")
        print(f"  Score: {results[best_traj_id]['traj_stability_metrics'].overall_stability_score:.4f}")
        
        print(f"\nWorst Trajectory Stability: {worst_traj_id}")
        print(f"  Task: {results[worst_traj_id]['task_name']}")
        print(f"  Score: {results[worst_traj_id]['traj_stability_metrics'].overall_stability_score:.4f}")
        
        print(f"\nBest Gripper Control: {best_gripper_id}")
        print(f"  Task: {results[best_gripper_id]['task_name']}")
        print(f"  Score: {results[best_gripper_id]['gripper_stability_metrics'].overall_stability_score:.4f}")
        
        print(f"\nWorst Gripper Control: {worst_gripper_id}")
        print(f"  Task: {results[worst_gripper_id]['task_name']}")
        print(f"  Score: {results[worst_gripper_id]['gripper_stability_metrics'].overall_stability_score:.4f}")
    
    return results

def demonstrate_vla_failure_detection():
    """Demonstrate how the metric can detect VLA action 'explosions'."""
    
    print("\n=== VLA Failure Detection Demonstration ===")
    
    # Create example trajectories
    T = 200
    D = 7
    
    # Normal trajectory
    t = np.linspace(0, 20, T)
    normal_actions = np.zeros((T, D))
    normal_actions[:, 0] = 0.5 * np.sin(0.2 * t)  # Smooth motion
    normal_actions[:, 1] = 0.3 * np.cos(0.15 * t)
    normal_actions[:, 2] = 0.1 * t / 20  # Slow linear motion
    
    # VLA "explosion" trajectory - sudden large jumps
    explosive_actions = normal_actions.copy()
    # Add sudden large jumps at random times (simulating VLA failures)
    explosion_times = [50, 120, 180]
    for t_exp in explosion_times:
        if t_exp < T:
            explosive_actions[t_exp:t_exp+5, :3] += np.random.uniform(-5, 5, (min(5, T-t_exp), 3))
    
    # Calculate metrics
    normal_score = calculate_action_smoothness_score(normal_actions)
    explosive_score = calculate_action_smoothness_score(explosive_actions)
    
    print(f"Normal Trajectory Smoothness: {normal_score:.4f}")
    print(f"Explosive Trajectory Smoothness: {explosive_score:.4f}")
    print(f"Difference: {normal_score - explosive_score:.4f}")
    
    # Detailed analysis
    calculator = TrajectoryStabilityCalculator()
    normal_metrics = calculator.calculate_stability_metrics(normal_actions)
    explosive_metrics = calculator.calculate_stability_metrics(explosive_actions)
    
    print(f"\nDetailed Comparison:")
    print(f"  Jerk Smoothness - Normal: {normal_metrics.jerk_smoothness:.4f}, Explosive: {explosive_metrics.jerk_smoothness:.4f}")
    print(f"  Acceleration Smoothness - Normal: {normal_metrics.acceleration_smoothness:.4f}, Explosive: {explosive_metrics.acceleration_smoothness:.4f}")
    
    # This metric can effectively detect when VLA actions "explode"
    if explosive_score < 0.5:  # Threshold for detecting problematic trajectories
        print(f"\n⚠️  VLA FAILURE DETECTED: Smoothness score {explosive_score:.4f} below threshold 0.5")
        print("   This indicates large impact forces and action discontinuities!")
    
    return normal_score, explosive_score

if __name__ == "__main__":
    # Analyze real trajectory data
    results = analyze_trajectory_stability()
    
    # Demonstrate VLA failure detection
    demonstrate_vla_failure_detection()
    
    print("\n=== Metric Usage for COIN Benchmark ===")
    print("\nTrajectory Stability Metrics can be used to:")
    print("1. Detect VLA action 'explosions' (mentioned in rebuttal)")
    print("2. Compare trajectory quality across different models")
    print("3. Identify problematic action sequences for debugging")
    print("4. Evaluate the effectiveness of action regularization techniques")
    print("5. Provide fine-grained analysis beyond binary success metrics")
    
    print("\nGripper Stability Metrics can be used to:")
    print("1. Detect erratic gripper control (excessive open/close actions)")
    print("2. Evaluate gripper timing accuracy and coordination")
    print("3. Compare gripper control quality across VLA models")
    print("4. Identify models prone to unstable gripper behavior")
    print("5. Support the enhanced VLA-specific analysis framework from our rebuttal")
