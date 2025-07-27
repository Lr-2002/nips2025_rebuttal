#!/usr/bin/env python3
"""
Metrics for evaluating trajectory quality and model performance in COIN benchmark.
This module implements various metrics for analyzing VLA and interactive reasoning capabilities.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import warnings

@dataclass
class TrajectoryStabilityMetrics:
    """Container for trajectory stability metrics"""
    velocity_smoothness: float
    acceleration_smoothness: float
    jerk_smoothness: float
    position_stability: float
    overall_stability_score: float
    velocity_stats: Dict[str, float]
    acceleration_stats: Dict[str, float]
    jerk_stats: Dict[str, float]

class TrajectoryStabilityCalculator:
    """
    Calculator for trajectory stability metrics based on kinematic analysis.
    
    This class implements metrics to evaluate the smoothness and stability of
    robotic trajectories by analyzing velocity, acceleration, and jerk.
    """
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize the calculator.
        
        Args:
            dt: Time step between trajectory points (default: 0.1s)
        """
        self.dt = dt
    
    def calculate_stability_metrics(self, actions: np.ndarray) -> TrajectoryStabilityMetrics:
        """
        Calculate comprehensive stability metrics for a trajectory.
        
        Args:
            actions: Array of shape (T, D) where T is time steps and D is action dimensions
                    Typically D=7 for 7-DOF arm (6 pose + 1 gripper)
        
        Returns:
            TrajectoryStabilityMetrics containing all stability measures
        """
        if len(actions.shape) != 2:
            raise ValueError(f"Actions must be 2D array (T, D), got shape {actions.shape}")
        
        T, D = actions.shape
        if T < 3:
            warnings.warn("Trajectory too short for meaningful stability analysis")
            return self._create_empty_metrics()
        
        # Extract position components (first 3 or 6 dimensions typically represent position/pose)
        # For 7-DOF: [x, y, z, rx, ry, rz, gripper] or similar
        position_dims = min(6, D - 1)  # Exclude gripper dimension
        positions = actions[:, :position_dims]
        
        # Calculate kinematic derivatives
        velocities = self._calculate_velocity(positions)
        accelerations = self._calculate_acceleration(velocities)
        jerks = self._calculate_jerk(accelerations)
        
        # Calculate stability metrics
        velocity_smoothness = self._calculate_smoothness_metric(velocities)
        acceleration_smoothness = self._calculate_smoothness_metric(accelerations)
        jerk_smoothness = self._calculate_smoothness_metric(jerks)
        position_stability = self._calculate_position_stability(positions)
        
        # Calculate statistics
        velocity_stats = self._calculate_kinematic_stats(velocities)
        acceleration_stats = self._calculate_kinematic_stats(accelerations)
        jerk_stats = self._calculate_kinematic_stats(jerks)
        
        # Overall stability score (weighted combination)
        overall_stability_score = self._calculate_overall_stability(
            velocity_smoothness, acceleration_smoothness, jerk_smoothness, position_stability
        )
        
        return TrajectoryStabilityMetrics(
            velocity_smoothness=velocity_smoothness,
            acceleration_smoothness=acceleration_smoothness,
            jerk_smoothness=jerk_smoothness,
            position_stability=position_stability,
            overall_stability_score=overall_stability_score,
            velocity_stats=velocity_stats,
            acceleration_stats=acceleration_stats,
            jerk_stats=jerk_stats
        )
    
    def _calculate_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate velocity from position trajectory.
        
        Formula: v(t) = (p(t+1) - p(t)) / dt
        
        Args:
            positions: Array of shape (T, D)
        
        Returns:
            Velocities of shape (T-1, D)
        """
        return np.diff(positions, axis=0) / self.dt
    
    def _calculate_acceleration(self, velocities: np.ndarray) -> np.ndarray:
        """
        Calculate acceleration from velocity trajectory.
        
        Formula: a(t) = (v(t+1) - v(t)) / dt
        
        Args:
            velocities: Array of shape (T-1, D)
        
        Returns:
            Accelerations of shape (T-2, D)
        """
        return np.diff(velocities, axis=0) / self.dt
    
    def _calculate_jerk(self, accelerations: np.ndarray) -> np.ndarray:
        """
        Calculate jerk from acceleration trajectory.
        
        Formula: j(t) = (a(t+1) - a(t)) / dt
        
        Args:
            accelerations: Array of shape (T-2, D)
        
        Returns:
            Jerks of shape (T-3, D)
        """
        return np.diff(accelerations, axis=0) / self.dt
    
    def _calculate_smoothness_metric(self, kinematic_data: np.ndarray) -> float:
        """
        Calculate smoothness metric based on variance of kinematic data.
        
        Formula: Smoothness = 1 / (1 + σ²)
        where σ² is the variance of the kinematic data
        
        Lower variance indicates smoother motion.
        Score ranges from 0 to 1, where 1 is perfectly smooth.
        
        Args:
            kinematic_data: Array of kinematic values (velocity, acceleration, or jerk)
        
        Returns:
            Smoothness score between 0 and 1
        """
        if kinematic_data.size == 0:
            return 0.0
        
        # Calculate variance across all dimensions and time
        variance = np.var(kinematic_data)
        
        # Convert to smoothness score (higher is better)
        smoothness = 1.0 / (1.0 + variance)
        
        return float(smoothness)
    
    def _calculate_position_stability(self, positions: np.ndarray) -> float:
        """
        Calculate position stability based on trajectory deviation.
        
        Formula: Stability = 1 / (1 + mean_deviation)
        where mean_deviation is the average distance from the trajectory centroid
        
        Args:
            positions: Array of shape (T, D)
        
        Returns:
            Position stability score between 0 and 1
        """
        if positions.size == 0:
            return 0.0
        
        # Calculate centroid of trajectory
        centroid = np.mean(positions, axis=0)
        
        # Calculate distances from centroid
        distances = np.linalg.norm(positions - centroid, axis=1)
        
        # Calculate mean deviation
        mean_deviation = np.mean(distances)
        
        # Convert to stability score
        stability = 1.0 / (1.0 + mean_deviation)
        
        return float(stability)
    
    def _calculate_kinematic_stats(self, kinematic_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistical measures for kinematic data.
        
        Args:
            kinematic_data: Array of kinematic values
        
        Returns:
            Dictionary with statistical measures
        """
        if kinematic_data.size == 0:
            return {
                'mean_magnitude': 0.0,
                'std_magnitude': 0.0,
                'max_magnitude': 0.0,
                'min_magnitude': 0.0,
                'rms': 0.0
            }
        
        # Calculate magnitudes (L2 norm across dimensions)
        magnitudes = np.linalg.norm(kinematic_data, axis=1)
        
        return {
            'mean_magnitude': float(np.mean(magnitudes)),
            'std_magnitude': float(np.std(magnitudes)),
            'max_magnitude': float(np.max(magnitudes)),
            'min_magnitude': float(np.min(magnitudes)),
            'rms': float(np.sqrt(np.mean(magnitudes**2)))
        }
    
    def _calculate_overall_stability(self, velocity_smoothness: float, 
                                   acceleration_smoothness: float,
                                   jerk_smoothness: float,
                                   position_stability: float) -> float:
        """
        Calculate overall stability score as weighted combination.
        
        Formula: Overall = w1*v_smooth + w2*a_smooth + w3*j_smooth + w4*p_stable
        where weights sum to 1.0
        
        Weights are chosen based on importance:
        - Jerk (40%): Most important for smooth control
        - Acceleration (30%): Important for control stability
        - Velocity (20%): Important for motion quality
        - Position (10%): Important for overall trajectory shape
        
        Args:
            velocity_smoothness: Velocity smoothness score
            acceleration_smoothness: Acceleration smoothness score
            jerk_smoothness: Jerk smoothness score
            position_stability: Position stability score
        
        Returns:
            Overall stability score between 0 and 1
        """
        weights = {
            'jerk': 0.4,
            'acceleration': 0.3,
            'velocity': 0.2,
            'position': 0.1
        }
        
        overall_score = (
            weights['velocity'] * velocity_smoothness +
            weights['acceleration'] * acceleration_smoothness +
            weights['jerk'] * jerk_smoothness +
            weights['position'] * position_stability
        )
        
        return float(overall_score)
    
    def _create_empty_metrics(self) -> TrajectoryStabilityMetrics:
        """Create empty metrics for invalid trajectories."""
        empty_stats = {
            'mean_magnitude': 0.0,
            'std_magnitude': 0.0,
            'max_magnitude': 0.0,
            'min_magnitude': 0.0,
            'rms': 0.0
        }
        
        return TrajectoryStabilityMetrics(
            velocity_smoothness=0.0,
            acceleration_smoothness=0.0,
            jerk_smoothness=0.0,
            position_stability=0.0,
            overall_stability_score=0.0,
            velocity_stats=empty_stats.copy(),
            acceleration_stats=empty_stats.copy(),
            jerk_stats=empty_stats.copy()
        )

def calculate_action_smoothness_score(actions: np.ndarray, dt: float = 0.1) -> float:
    """
    Convenience function to calculate action smoothness score.
    
    This is the main metric mentioned in the rebuttal for detecting
    large impact forces and action discontinuities that cause VLA failures.
    
    Args:
        actions: Action trajectory array of shape (T, D)
        dt: Time step between actions
    
    Returns:
        Action smoothness score between 0 and 1 (higher is better)
    """
    calculator = TrajectoryStabilityCalculator(dt=dt)
    metrics = calculator.calculate_stability_metrics(actions)
    return metrics.overall_stability_score

# Example usage and testing
@dataclass
class GripperStabilityMetrics:
    """Container for gripper stability analysis results."""
    smoothness_score: float
    frequency_score: float
    coordination_score: float
    overall_stability_score: float
    
    # Detailed statistics
    total_gripper_changes: int
    expected_changes: int
    mean_change_magnitude: float
    coordination_events: List[Dict[str, Any]]


class GripperStabilityCalculator:
    """Calculate gripper control stability metrics to detect erratic gripper behavior.
    
    This class implements metrics to evaluate whether a model exhibits stable gripper
    control or tends to randomly open/close the gripper during task execution.
    
    The overall stability score combines:
    1. Smoothness: Penalizes abrupt gripper state changes
    2. Frequency: Penalizes excessive gripper open/close actions
    3. Coordination: Rewards gripper actions coordinated with arm motion
    """
    
    def __init__(self, 
                 gripper_threshold: float = 0.1,
                 smoothness_weight: float = 0.4,
                 frequency_weight: float = 0.3,
                 coordination_weight: float = 0.3,
                 coordination_window: int = 10):
        """
        Initialize gripper stability calculator.
        
        Args:
            gripper_threshold: Minimum change to consider a gripper state transition
            smoothness_weight: Weight for smoothness component in overall score
            frequency_weight: Weight for frequency component in overall score  
            coordination_weight: Weight for coordination component in overall score
            coordination_window: Number of frames before/after gripper change to analyze
        """
        self.gripper_threshold = gripper_threshold
        self.smoothness_weight = smoothness_weight
        self.frequency_weight = frequency_weight
        self.coordination_weight = coordination_weight
        self.coordination_window = coordination_window
        
        # Validate weights sum to 1
        total_weight = smoothness_weight + frequency_weight + coordination_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def calculate_gripper_stability(self, actions: np.ndarray) -> GripperStabilityMetrics:
        """
        Calculate comprehensive gripper stability metrics.
        
        Args:
            actions: Action trajectory array of shape (T, D) where:
                    - T is the number of time steps
                    - D is the action dimension
                    - actions[:, -1] is assumed to be gripper state
                    - actions[:, :6] are assumed to be pose (x,y,z,rx,ry,rz)
        
        Returns:
            GripperStabilityMetrics containing all stability scores and statistics
        """
        if len(actions.shape) != 2:
            raise ValueError(f"Actions must be 2D array (T, D), got shape {actions.shape}")
        
        T, D = actions.shape
        if T < 3:
            warnings.warn("Trajectory too short for meaningful gripper analysis")
            return self._create_empty_gripper_metrics()
        
        # Extract gripper states (assume last dimension)
        gripper_states = actions[:, -1]
        
        # Extract position for coordination analysis (first 3 dimensions)
        positions = actions[:, :3] if D >= 3 else actions[:, :min(3, D-1)]
        
        # Calculate individual components
        smoothness_score = self._calculate_gripper_smoothness(gripper_states)
        frequency_score, total_changes, expected_changes = self._calculate_gripper_frequency(gripper_states, T)
        coordination_score, coordination_events = self._calculate_gripper_coordination(gripper_states, positions)
        
        # Calculate overall stability score
        overall_score = (
            self.smoothness_weight * smoothness_score +
            self.frequency_weight * frequency_score +
            self.coordination_weight * coordination_score
        )
        
        # Calculate statistics
        gripper_changes = np.abs(np.diff(gripper_states))
        mean_change_magnitude = np.mean(gripper_changes[gripper_changes > self.gripper_threshold])
        if np.isnan(mean_change_magnitude):
            mean_change_magnitude = 0.0
        
        return GripperStabilityMetrics(
            smoothness_score=smoothness_score,
            frequency_score=frequency_score,
            coordination_score=coordination_score,
            overall_stability_score=overall_score,
            total_gripper_changes=total_changes,
            expected_changes=expected_changes,
            mean_change_magnitude=mean_change_magnitude,
            coordination_events=coordination_events
        )
    
    def _calculate_gripper_smoothness(self, gripper_states: np.ndarray) -> float:
        """
        Calculate gripper state smoothness.
        
        Penalizes abrupt changes in gripper state by measuring the variance
        of gripper state changes.
        
        Formula: smoothness = 1 / (1 + variance(state_changes))
        """
        state_changes = np.abs(np.diff(gripper_states))
        variance = np.var(state_changes)
        smoothness = 1.0 / (1.0 + variance)
        return float(smoothness)
    
    def _calculate_gripper_frequency(self, gripper_states: np.ndarray, trajectory_length: int) -> Tuple[float, int, int]:
        """
        Calculate gripper action frequency score.
        
        Penalizes excessive gripper state transitions by comparing actual
        transitions to expected reasonable number.
        
        Formula: frequency_score = min(1.0, expected_transitions / max(1, actual_transitions))
        """
        # Count significant gripper state changes
        state_changes = np.abs(np.diff(gripper_states))
        significant_changes = np.sum(state_changes > self.gripper_threshold)
        
        # Expected reasonable number of gripper changes
        # Assume 2-4 changes per 100 frames is reasonable for most tasks
        expected_changes = max(2, trajectory_length // 50)
        
        # Calculate frequency penalty
        if significant_changes == 0:
            frequency_score = 1.0  # No changes is perfect
        else:
            frequency_score = min(1.0, expected_changes / significant_changes)
        
        return float(frequency_score), int(significant_changes), int(expected_changes)
    
    def _calculate_gripper_coordination(self, gripper_states: np.ndarray, positions: np.ndarray) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Calculate gripper-motion coordination score.
        
        Analyzes whether gripper state changes are coordinated with arm motion
        by examining velocity changes in the 10 frames before and after each
        gripper state transition.
        
        Good coordination patterns:
        - Gripper closing: Arm decelerates before gripper closes (approaching object)
        - Gripper opening: Arm may accelerate after gripper opens (moving away)
        """
        # Calculate arm velocity
        arm_velocity = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        
        # Find gripper state change times
        gripper_changes = np.abs(np.diff(gripper_states))
        change_indices = np.where(gripper_changes > self.gripper_threshold)[0]
        
        if len(change_indices) == 0:
            return 1.0, []  # No gripper changes = perfect coordination
        
        coordination_scores = []
        coordination_events = []
        
        for t in change_indices:
            # Analyze coordination for this gripper change
            coord_score, event_info = self._analyze_single_gripper_coordination(
                t, gripper_states, arm_velocity
            )
            coordination_scores.append(coord_score)
            coordination_events.append(event_info)
        
        overall_coordination = np.mean(coordination_scores)
        return float(overall_coordination), coordination_events
    
    def _analyze_single_gripper_coordination(self, t: int, gripper_states: np.ndarray, arm_velocity: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze coordination for a single gripper state change.
        
        Args:
            t: Time index of gripper change
            gripper_states: Full gripper state trajectory
            arm_velocity: Arm velocity trajectory
        
        Returns:
            Coordination score (0-1) and detailed event information
        """
        # Define analysis windows
        before_start = max(0, t - self.coordination_window)
        after_end = min(len(arm_velocity), t + self.coordination_window)
        
        # Calculate velocities in before/after windows
        before_velocities = arm_velocity[before_start:t] if t > before_start else np.array([])
        after_velocities = arm_velocity[t:after_end] if after_end > t else np.array([])
        
        before_vel = np.mean(before_velocities) if len(before_velocities) > 0 else 0.0
        after_vel = np.mean(after_velocities) if len(after_velocities) > 0 else 0.0
        
        # Determine gripper change type
        gripper_change = gripper_states[min(t+1, len(gripper_states)-1)] - gripper_states[t]
        is_closing = gripper_change < -self.gripper_threshold
        is_opening = gripper_change > self.gripper_threshold
        
        # Calculate coordination score based on velocity patterns
        coordination_score = self._score_velocity_coordination(
            before_vel, after_vel, is_closing, is_opening
        )
        
        # Create event information
        event_info = {
            'time_index': int(t),
            'gripper_change': float(gripper_change),
            'is_closing': bool(is_closing),
            'is_opening': bool(is_opening),
            'before_velocity': float(before_vel),
            'after_velocity': float(after_vel),
            'coordination_score': float(coordination_score)
        }
        
        return coordination_score, event_info
    
    def _score_velocity_coordination(self, before_vel: float, after_vel: float, 
                                   is_closing: bool, is_opening: bool) -> float:
        """
        Score the coordination between gripper action and velocity changes.
        
        Coordination patterns:
        - Closing gripper: Good if arm was moving and slows down (approaching object)
        - Opening gripper: Good if arm speeds up after opening (moving away)
        - General: Any significant velocity change around gripper action is good
        """
        # Avoid division by zero
        velocity_epsilon = 0.01
        
        if before_vel < velocity_epsilon and after_vel < velocity_epsilon:
            # Both velocities very low - static gripper operation
            return 0.8  # Reasonable coordination
        
        # Calculate velocity change ratio
        if before_vel > velocity_epsilon:
            velocity_change_ratio = abs(before_vel - after_vel) / before_vel
        else:
            velocity_change_ratio = min(1.0, after_vel / velocity_epsilon)
        
        # Score based on gripper action type and velocity pattern
        if is_closing:
            # Closing gripper: reward deceleration (approaching object)
            if before_vel > after_vel:
                return min(1.0, 0.8 + 0.2 * velocity_change_ratio)
            else:
                return min(0.7, 0.5 + 0.2 * velocity_change_ratio)
        
        elif is_opening:
            # Opening gripper: reward acceleration (moving away) or maintaining speed
            if after_vel >= before_vel * 0.8:  # Not slowing down much
                return min(1.0, 0.8 + 0.2 * velocity_change_ratio)
            else:
                return min(0.6, 0.4 + 0.2 * velocity_change_ratio)
        
        else:
            # Neutral gripper change: any velocity adjustment is reasonable
            return min(1.0, 0.6 + 0.4 * velocity_change_ratio)
    
    def _create_empty_gripper_metrics(self) -> GripperStabilityMetrics:
        """Create empty metrics for edge cases."""
        return GripperStabilityMetrics(
            smoothness_score=1.0,
            frequency_score=1.0,
            coordination_score=1.0,
            overall_stability_score=1.0,
            total_gripper_changes=0,
            expected_changes=0,
            mean_change_magnitude=0.0,
            coordination_events=[]
        )


def calculate_gripper_stability_score(actions: np.ndarray, 
                                     gripper_threshold: float = 0.1) -> float:
    """
    Convenience function to calculate gripper stability score.
    
    Args:
        actions: Action trajectory array of shape (T, D)
        gripper_threshold: Minimum change to consider a gripper state transition
    
    Returns:
        Overall gripper stability score (0-1, higher is better)
    """
    calculator = GripperStabilityCalculator(gripper_threshold=gripper_threshold)
    metrics = calculator.calculate_gripper_stability(actions)
    return metrics.overall_stability_score


if __name__ == "__main__":
    # Create sample trajectory data
    T = 100  # 100 time steps
    D = 7    # 7-dimensional actions (6 pose + 1 gripper)
    
    # Generate smooth trajectory
    t = np.linspace(0, 10, T)
    smooth_actions = np.zeros((T, D))
    
    # Smooth sinusoidal motion for pose
    smooth_actions[:, 0] = 0.5 * np.sin(0.5 * t)  # x
    smooth_actions[:, 1] = 0.3 * np.cos(0.3 * t)  # y
    smooth_actions[:, 2] = 0.1 * t / 10           # z (slow rise)
    smooth_actions[:, 3] = 0.2 * np.sin(0.2 * t)  # rx
    smooth_actions[:, 4] = 0.1 * np.cos(0.4 * t)  # ry
    smooth_actions[:, 5] = 0.05 * t / 10          # rz
    
    # Smooth gripper control (reasonable open/close pattern)
    smooth_actions[:20, 6] = 1.0   # Open initially
    smooth_actions[20:40, 6] = 0.0  # Close for grasp
    smooth_actions[40:80, 6] = 0.0  # Keep closed during manipulation
    smooth_actions[80:, 6] = 1.0   # Open to release
    
    # Generate noisy trajectory (simulating VLA action explosion)
    noisy_actions = smooth_actions.copy()
    # Add random large jumps to simulate action explosions
    explosion_times = [20, 45, 70]
    for t_exp in explosion_times:
        noisy_actions[t_exp:t_exp+3, :6] += np.random.uniform(-2, 2, (3, 6))
    
    # Generate erratic gripper behavior
    erratic_gripper_actions = smooth_actions.copy()
    # Add random gripper openings/closings
    for i in range(10, 90, 8):  # Every 8 frames
        erratic_gripper_actions[i:i+2, 6] = 1.0 - erratic_gripper_actions[i-1, 6]  # Flip state
    
    # Calculate trajectory stability metrics
    traj_calculator = TrajectoryStabilityCalculator()
    
    smooth_traj_metrics = traj_calculator.calculate_stability_metrics(smooth_actions)
    noisy_traj_metrics = traj_calculator.calculate_stability_metrics(noisy_actions)
    
    print("=== Trajectory Stability Analysis ===")
    print()
    print("Smooth Trajectory:")
    print(f"  Overall Stability: {smooth_traj_metrics.overall_stability_score:.4f}")
    print(f"  Velocity Smoothness: {smooth_traj_metrics.velocity_smoothness:.4f}")
    print(f"  Acceleration Smoothness: {smooth_traj_metrics.acceleration_smoothness:.4f}")
    print(f"  Jerk Smoothness: {smooth_traj_metrics.jerk_smoothness:.4f}")
    print()
    print("Noisy Trajectory:")
    print(f"  Overall Stability: {noisy_traj_metrics.overall_stability_score:.4f}")
    print(f"  Velocity Smoothness: {noisy_traj_metrics.velocity_smoothness:.4f}")
    print(f"  Acceleration Smoothness: {noisy_traj_metrics.acceleration_smoothness:.4f}")
    print(f"  Jerk Smoothness: {noisy_traj_metrics.jerk_smoothness:.4f}")
    print()
    print("Convenience Function:")
    smooth_score = calculate_action_smoothness_score(smooth_actions)
    noisy_score = calculate_action_smoothness_score(noisy_actions)
    print(f"  Smooth Action Score: {smooth_score:.4f}")
    print(f"  Noisy Action Score: {noisy_score:.4f}")
    
    # Calculate gripper stability metrics
    gripper_calculator = GripperStabilityCalculator()
    
    smooth_gripper_metrics = gripper_calculator.calculate_gripper_stability(smooth_actions)
    erratic_gripper_metrics = gripper_calculator.calculate_gripper_stability(erratic_gripper_actions)
    
    print("\n=== Gripper Control Stability Analysis ===")
    print()
    print("Smooth Gripper Control:")
    print(f"  Overall Stability: {smooth_gripper_metrics.overall_stability_score:.4f}")
    print(f"  Smoothness Score: {smooth_gripper_metrics.smoothness_score:.4f}")
    print(f"  Frequency Score: {smooth_gripper_metrics.frequency_score:.4f}")
    print(f"  Coordination Score: {smooth_gripper_metrics.coordination_score:.4f}")
    print(f"  Total Gripper Changes: {smooth_gripper_metrics.total_gripper_changes}")
    print(f"  Expected Changes: {smooth_gripper_metrics.expected_changes}")
    print()
    print("Erratic Gripper Control:")
    print(f"  Overall Stability: {erratic_gripper_metrics.overall_stability_score:.4f}")
    print(f"  Smoothness Score: {erratic_gripper_metrics.smoothness_score:.4f}")
    print(f"  Frequency Score: {erratic_gripper_metrics.frequency_score:.4f}")
    print(f"  Coordination Score: {erratic_gripper_metrics.coordination_score:.4f}")
    print(f"  Total Gripper Changes: {erratic_gripper_metrics.total_gripper_changes}")
    print(f"  Expected Changes: {erratic_gripper_metrics.expected_changes}")
    
    # Show detailed coordination events for erratic gripper
    print("\n  Coordination Events (first 3):")
    for i, event in enumerate(erratic_gripper_metrics.coordination_events[:3]):
        print(f"    Event {i+1}: t={event['time_index']}, "
              f"closing={event['is_closing']}, "
              f"coord_score={event['coordination_score']:.3f}")
    
    print("\n=== Gripper Stability Detection ===")
    smooth_gripper_score = calculate_gripper_stability_score(smooth_actions)
    erratic_gripper_score = calculate_gripper_stability_score(erratic_gripper_actions)
    
    print(f"Smooth Gripper Score: {smooth_gripper_score:.4f}")
    print(f"Erratic Gripper Score: {erratic_gripper_score:.4f}")
    
    # Detection thresholds
    if erratic_gripper_score < 0.6:
        print(f"\n⚠️  ERRATIC GRIPPER DETECTED: Score {erratic_gripper_score:.4f} below threshold 0.6")
        print("   This indicates unstable gripper control with excessive open/close actions!")
    
    if smooth_gripper_score > 0.8:
        print(f"\n✅ STABLE GRIPPER CONTROL: Score {smooth_gripper_score:.4f} above threshold 0.8")
        print("   This indicates well-coordinated gripper behavior!")
