# Trajectory Stability Metrics for COIN Benchmark

## Overview

This document describes the trajectory stability metrics implemented for the COIN benchmark, specifically designed to detect VLA action "explosions" and evaluate trajectory quality as mentioned in our rebuttal.

## Core Metrics

### 1. Action Smoothness Score (Trajectory Stability)

**Purpose**: Detect large impact forces and action discontinuities that cause VLA model failures.

**Formula**: 
$$
S_{\text{traj}} = 0.3 \cdot S_{\text{vel}} + 0.3 \cdot S_{\text{acc}} + 0.2 \cdot S_{\text{jerk}} + 0.2 \cdot S_{\text{pos}}
$$

Where:
- $S_{\text{vel}}$ = velocity smoothness
- $S_{\text{acc}}$ = acceleration smoothness  
- $S_{\text{jerk}}$ = jerk smoothness
- $S_{\text{pos}}$ = position stability

### 2. Kinematic Analysis Components

#### Velocity Calculation
$$
\mathbf{v}_t = \frac{\mathbf{p}_{t+1} - \mathbf{p}_t}{\Delta t}
$$

#### Acceleration Calculation  
$$
\mathbf{a}_t = \frac{\mathbf{v}_{t+1} - \mathbf{v}_t}{\Delta t}
$$

#### Jerk Calculation
$$
\mathbf{j}_t = \frac{\mathbf{a}_{t+1} - \mathbf{a}_t}{\Delta t}
$$

### 3. Smoothness Metric
$$
S_{\text{smooth}} = \exp\left(-\alpha \cdot \frac{\sigma(|\mathbf{q}_t|)}{\mu(|\mathbf{q}_t|) + \epsilon}\right)
$$
Where $\sigma$ is standard deviation, $\mu$ is mean, and $\epsilon = 10^{-6}$.

### 4. Position Stability
$$
S_{\text{pos}} = \exp\left(-\beta \cdot \frac{1}{T-k} \sum_{t=k+1}^{T} \|\mathbf{p}_t - \mathbf{p}_{t-k}\|_2\right)
$$
Where $\beta = 1.0$.

### 5. Gripper Control Stability Score

**Purpose**: Detect erratic gripper control behavior, such as excessive open/close actions that indicate unstable VLA gripper control.

**Formula**:
$$
S_{\text{gripper}} = 0.4 \cdot S_{\text{smooth}} + 0.3 \cdot S_{\text{freq}} + 0.3 \cdot S_{\text{coord}}
$$

Where:
- $S_{\text{smooth}}$ = gripper smoothness
- $S_{\text{freq}}$ = gripper frequency score
- $S_{\text{coord}}$ = gripper-motion coordination

#### 5.1 Gripper Smoothness
$$
S_{\text{smooth}} = \exp\left(-\gamma \cdot \frac{N_{\text{abrupt}}}{N_{\text{total}} + \epsilon}\right)
$$
Where $\gamma = 3.0$.

#### 5.2 Gripper Frequency Score
$$
S_{\text{freq}} = \min\left(1.0, \frac{N_{\text{expected}}}{N_{\text{total}}}\right)
$$
Where $N_{\text{expected}} = \left\lfloor \frac{T}{50} \right\rfloor$.

#### 5.3 Gripper-Motion Coordination
$$
S_{\text{coord}} = \frac{\sum_{t} (R_{\text{decel}}(t) + R_{\text{accel}}(t))}{N_{\text{total}}}
$$

## Implementation

### Key Classes

1. **`TrajectoryStabilityCalculator`**: Main trajectory stability calculator class
2. **`TrajectoryStabilityMetrics`**: Data container for trajectory results
3. **`GripperStabilityCalculator`**: Main gripper stability calculator class
4. **`GripperStabilityMetrics`**: Data container for gripper results
5. **`calculate_action_smoothness_score()`**: Convenience function for trajectory stability
6. **`calculate_gripper_stability_score()`**: Convenience function for gripper stability

### Usage Example

```python
from metric import calculate_action_smoothness_score, calculate_gripper_stability_score
import numpy as np

# Load your action trajectory (T, D) where T=time, D=dimensions
actions = np.array(trajectory_data)  # Shape: (410, 7) for example

# Calculate trajectory stability score
traj_score = calculate_action_smoothness_score(actions, dt=0.1)

# Calculate gripper stability score  
gripper_score = calculate_gripper_stability_score(actions)

# Interpret results
if traj_score < 0.5:
    print("⚠️ VLA ACTION EXPLOSION detected!")
if gripper_score < 0.6:
    print("⚠️ ERRATIC GRIPPER CONTROL detected!")
```

## Results on Real Data

### Current COIN Trajectory Analysis

From our analysis of trajectory `20250511_013942` (Tabletop-Seek-Holder-InCabinet-v1):

**Trajectory Stability:**
- **Overall Stability Score**: 0.1475 (Low - indicates problematic actions)
- **Velocity Smoothness**: 0.3317
- **Acceleration Smoothness**: 0.0017 (Very low - indicates large accelerations)
- **Jerk Smoothness**: 0.0000 (Critical - indicates severe jerk)

**Gripper Stability:**
- **Overall Stability Score**: 0.2498 (Low - indicates erratic gripper control)
- **Smoothness Score**: 0.0545 (Very low - indicates abrupt gripper changes)
- **Frequency Score**: 0.0214 (Critical - 374 changes vs 8 expected)
- **Coordination Score**: 0.7385 (Good - gripper changes are coordinated with arm motion)

### VLA Failure Detection Capability

Our metrics successfully distinguish between different failure modes:

**Trajectory Stability Detection:**
- **Normal Trajectory**: 0.9761 smoothness score
- **Explosive Trajectory**: 0.0636 smoothness score
- **Detection Threshold**: 0.5 (trajectories below this indicate VLA action explosions)

**Gripper Control Detection:**
- **Stable Gripper Control**: 0.8982 stability score
- **Erratic Gripper Control**: 0.5915 stability score
- **Detection Threshold**: 0.6 (scores below this indicate erratic gripper behavior)

## Applications in COIN Benchmark

### 1. VLA-Specific Analysis (Rebuttal Metrics)
- **Action Smoothness Score**: Detects when VLA actions "explode"
  - **Threshold**: < 0.5 indicates problematic trajectory
  - **Use Case**: Identify models prone to unstable control
- **Gripper Stability Score**: Detects erratic gripper control behavior
  - **Threshold**: < 0.6 indicates excessive gripper open/close actions
  - **Use Case**: Evaluate gripper timing accuracy and coordination

### 2. Model Comparison
- Compare trajectory quality across different VLA models
- Compare gripper control stability across different models
- Quantify improvement from action regularization techniques
- Evaluate curriculum learning effectiveness on both motion and gripper control

### 3. Debugging and Analysis
- Identify specific time points where actions become unstable
- Detect excessive gripper open/close patterns and their timing
- Analyze correlation between instability and task failure
- Evaluate gripper-motion coordination quality
- Guide development of more stable VLA architectures

### 4. Enhanced Evaluation Metrics
- Move beyond binary success/failure metrics
- Provide fine-grained analysis of model behavior
- Support the enhanced metrics framework proposed in our rebuttal

## Integration with COIN Benchmark

### Data Pipeline
1. **Load trajectories** using `RebuttalDataLoader`
2. **Extract actions** from trajectory data
3. **Calculate stability metrics** using `TrajectoryStabilityCalculator`
4. **Analyze results** for model evaluation

### Metric Categories (from Rebuttal)

This implementation directly supports:

- **VLA-Specific Analysis Metrics**:
  - ✅ Action Smoothness Score (implemented)
  - ✅ Gripper Timing Accuracy (implemented as Gripper Stability Score)
  - Generalization Capability Score (future work)

- **Interactive Reasoning Analysis Metrics**:
  - Spatial Information Utilization Score (future work)
  - Instruction Changing Score (future work)
  - Task Decomposition Score (future work)

## Technical Details

### Input Requirements
- **Action Array**: Shape (T, D) where T = time steps, D = action dimensions
- **Time Step**: dt parameter (default: 0.1s for 10Hz control)
- **Position Dimensions**: First 6 dimensions assumed to be pose (x,y,z,rx,ry,rz)

### Output Metrics
- **Scores**: All metrics range from 0 to 1 (higher = better)
- **Statistics**: Mean, std, max, min, RMS for velocity/acceleration/jerk
- **Detailed Analysis**: Component-wise breakdown of stability issues

### Performance
- **Computational Complexity**: O(T×D) where T = trajectory length, D = dimensions
- **Memory Usage**: Linear in trajectory size
- **Real-time Capable**: Can process 400-step trajectories in milliseconds

## Future Extensions

1. **Gripper-Specific Metrics**: Analyze gripper timing and force application
2. **Multi-Modal Integration**: Incorporate visual and language information
3. **Task-Specific Thresholds**: Adaptive thresholds based on task difficulty
4. **Comparative Analysis**: Benchmarking against human demonstration quality

## References

This implementation supports the enhanced evaluation framework proposed in our COIN benchmark rebuttal, specifically addressing:

- Reviewer u7xR's concerns about evaluation metrics
- Reviewer hSnT's request for fine-grained analysis beyond binary metrics
- Our proposed VLA-specific diagnostic capabilities

The metrics provide quantitative evidence for the "action explosion" phenomenon we identified in VLA models and support our argument for more nuanced evaluation approaches in interactive reasoning benchmarks.
