# COIN Benchmark Scoring System Summary

本文档总结了为COIN benchmark开发的所有评估指标和打分系统，包括公式、阈值和使用方法。

## 1. 轨迹稳定性评估 (Trajectory Stability Metrics)

### 1.1 总体稳定性得分
**公式：**
```
Overall_Stability = 0.3×velocity_smoothness + 0.3×acceleration_smoothness + 0.2×jerk_smoothness + 0.2×position_stability
```

**组成部分：**

#### 1.1.1 速度平滑度 (Velocity Smoothness)
```
velocity_smoothness = exp(-α × velocity_variation)
其中：velocity_variation = std(||v_t||) / (mean(||v_t||) + ε)
α = 2.0, ε = 1e-6
```

#### 1.1.2 加速度平滑度 (Acceleration Smoothness)  
```
acceleration_smoothness = exp(-α × acceleration_variation)
其中：acceleration_variation = std(||a_t||) / (mean(||a_t||) + ε)
```

#### 1.1.3 急动度平滑度 (Jerk Smoothness)
```
jerk_smoothness = exp(-α × jerk_variation)
其中：jerk_variation = std(||j_t||) / (mean(||j_t||) + ε)
```

#### 1.1.4 位置稳定性 (Position Stability)
```
position_stability = exp(-β × position_drift)
其中：position_drift = mean(||p_t - p_{t-k}||), k=5
β = 1.0
```

**检测阈值：**
- `< 0.5`: VLA动作爆炸 (VLA Action Explosion)
- `> 0.8`: 良好的轨迹稳定性

---

## 2. 夹爪控制稳定性评估 (Gripper Control Stability)

### 2.1 总体夹爪稳定性得分
**公式：**
```
Gripper_Stability = 0.4×smoothness + 0.3×frequency + 0.3×coordination
```

**组成部分：**

#### 2.1.1 夹爪平滑度 (Gripper Smoothness)
```
smoothness = exp(-γ × abrupt_changes / total_changes)
其中：abrupt_changes = count(|gripper_{t+1} - gripper_t| > threshold)
γ = 3.0, threshold = 0.3
```

#### 2.1.2 夹爪频率得分 (Gripper Frequency)
```
frequency = min(1.0, expected_changes / actual_changes)
其中：expected_changes = trajectory_length / 50  # 假设每50步一次合理的夹爪动作
```

#### 2.1.3 夹爪协调性 (Gripper Coordination)
```
coordination = (deceleration_reward + acceleration_reward) / total_gripper_changes
```

**协调性分析：**
- 分析夹爪状态变化前后10帧的手臂运动
- 夹爪闭合前减速：+1分 (接近物体)
- 夹爪打开后加速：+1分 (离开物体)

**检测阈值：**
- `< 0.6`: 夹爪控制不稳定 (Erratic Gripper Control)
- `> 0.8`: 良好的夹爪控制

---

## 3. 任务分解质量评估 (Task Decomposition Quality) [独立模块]

### 3.1 总体任务分解得分
**公式：**
```
Task_Decomposition = 0.3×logical_coherence + 0.25×granularity + 0.25×completeness + 0.2×primitive_alignment
```

**组成部分：**

#### 3.1.1 逻辑连贯性 (Logical Coherence)
- 检查子任务的逻辑顺序 (如：开门→拿取→放置→关门)
- 奖励合理模式：+10%
- 惩罚反逻辑模式：-20%

#### 3.1.2 粒度适当性 (Granularity)
```
granularity = f(avg_words_per_subtask, word_variance)
理想范围：4-8个词/子任务，低方差
```

#### 3.1.3 完整性 (Completeness)
```
completeness = covered_key_actions / total_key_actions
从高级指令中提取关键动作，检查子任务覆盖率
```

#### 3.1.4 原始任务对齐 (Primitive Alignment)
```
primitive_alignment = matching_subtasks / total_subtasks
检查子任务与预定义原始任务的匹配度
```

**检测阈值：**
- `< 0.6`: 任务分解质量差
- `> 0.8`: 良好的任务分解

---

## 4. 指令变化分析 (Instruction Change Analysis) [独立模块]

### 4.1 指令变化检测
**检测类型：**
1. **初始指令** (Initial Instruction)
2. **子任务变化** (Subtask Change) 
3. **计划更新** (Plan Update)
4. **高级变化** (High-level Change)

### 4.2 人工标注评分标准
**评分维度：**
1. **指令清晰度** (Instruction Clarity): 1-5分
2. **过渡平滑度** (Transition Smoothness): 1-5分  
3. **执行质量** (Execution Quality): 1-5分
4. **整体处理** (Overall Handling): 1-5分

**问题行为标记：**
- 指令模糊 (Ambiguous Instructions)
- 突然变化 (Abrupt Changes)
- 执行错误 (Execution Errors)
- 恢复失败 (Recovery Failures)

---

## 5. 集成评估系统

### 5.1 当前集成的指标
**主要评估脚本：** `analyze_trajectory_stability.py`

**包含指标：**
1. ✅ 轨迹稳定性评估
2. ✅ 夹爪控制稳定性评估

**独立模块：**
1. 📋 任务分解质量评估 (`task_decomposition_scorer.py`)
2. 📋 指令变化分析 (`instruction_change_analyzer.py`)
3. 📋 人工标注系统 (`annotation_scorer.py`)

### 5.2 使用方法

#### 运行综合分析：
```bash
cd /Users/lr-2002/project/reasoning_manipulation/rebuttal
python analyze_trajectory_stability.py
```

#### 运行独立模块：
```bash
# 任务分解评估
python task_decomposition_scorer.py

# 指令变化分析
python instruction_change_analyzer.py

# 人工标注
python annotation_scorer.py
```

---

## 6. 检测能力总结

### 6.1 VLA模型问题检测
| 问题类型 | 检测指标 | 阈值 | 状态 |
|---------|---------|------|------|
| VLA动作爆炸 | 轨迹稳定性 | < 0.5 | ✅ 已实现 |
| 夹爪控制不稳定 | 夹爪稳定性 | < 0.6 | ✅ 已实现 |
| 任务分解质量差 | 分解得分 | < 0.6 | 📋 独立模块 |
| 指令处理问题 | 人工标注 | < 3.0 | 📋 独立模块 |

### 6.2 实际数据验证结果
**示例轨迹分析：**
- 轨迹稳定性：0.8982 (良好)
- 夹爪稳定性：0.5915 (检测到不稳定)
  - 374次夹爪变化 vs 8次预期 (频率得分: 0.0214)
- 任务分解：0.8750 (良好)

---

## 7. 与Rebuttal的对应关系

### 7.1 审稿人关注点对应
| 审稿人问题 | 对应指标 | 实现状态 |
|-----------|---------|---------|
| 夹爪时序准确性 | 夹爪稳定性评估 | ✅ 已实现 |
| VLA动作质量 | 轨迹稳定性评估 | ✅ 已实现 |
| 交互推理分析 | 任务分解评估 | 📋 独立模块 |
| 失败恢复能力 | 指令变化分析 | 📋 独立模块 |

### 7.2 定量评估支持
- 提供细粒度分析，超越二元成功指标
- 支持VLA特定行为分析
- 为rebuttal提供定量证据支持

---

## 8. 未来扩展方向

1. **多模态集成评估**
2. **动态环境适应性评估** 
3. **人机协作质量评估**
4. **长期推理一致性评估**

---

*本文档总结了COIN benchmark的完整评估框架，支持论文rebuttal中提出的增强评估指标。*
