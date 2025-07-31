# Rebuttal: COIN Benchmark for Interactive Reasoning in Embodied AI


## Reviewer u7xR

### Q1: The success rates across different models on COIN-50 are consistently below 3%, indicating that the tasks may be overly difficult and offering limited insight into model capabilities.

**Response:**

We thank you for your constructive feedback ,but we want to argue that these low success rates are **highly informative** and reveal critical limitations that binary metrics alone cannot capture. Our comprehensive analysis of 50 VLA trajectories provides rich experimental evidence(All of the following response could find related provf at the response for Q3):

**Key Finding 1: Universal VLA Action Instability**
Models produce **erratic, high-variance movements** leading to systematic task failures(All of their action smoothness are under Human Performance). And even some model will give out very large jerk, which will hurt the robot boyd. This represents a fundamental control problem requiring immediate attention

**Key Finding 2: Severe Generalization Failures** 
Models fail to adapt manipulation strategies for different visual/physical properties and different instructions. We decompose 20 mid-term tasks from COIN-50, which have only visual difference or instruction difference with COIN-Primitive and measure the decay of their success rate, their **generalization score is very low**: CogACT 0.08, PI0 0.40, GR00T 0.0. This is another reason why they could not handle the final COIN-50 properly

**Key Finding 3: VLM vs VLA Performance Gap**
VLM components could achieve reasonable task decomposition  (GPT4o 0.6+-, Gemini 0.4+-) Meaning the current bottleneck of H-VLA or CodeAsPolicy lies in low-level manipulation lies in VLA execution layer, not high-level reasoning;While the task **decomposition score did not change a lot** along the task procedure going on according to the Figure 13 in the Appendix. So may be we need to improve the ability of Memory using and decision making of the VLM. 

**Actionable Insights for the Community:**
Rather than focusing solely on end-to-end training, or large scale data training. 
Researchers should also pay attention to: (1) **Action stability problems and visual/instruction stability** for VLAs (2) **VLM-VLA integration and close-loop feedback(integrating memory strageties and video understanding )** for Hierachical-VLAs. (3)**Feedback control** which could change the plan according to the change of the situation for CodeAsPolicy model. 


### Q2: The benchmark currently makes a sharp leap from primitive tasks to full interactive reasoning, without including tasks of intermediate complexity. This gap hinders the ability to evaluate model progress gradually or apply curriculum-based learning strategies.

**Response:**

We have addressed this by implementing **mid-term tasks** as intermediate complexity levels. These tasks bridge primitive and full interactive reasoning by introducing controlled complexity increases:

**Detail Implementation:**
We built **20 composition tasks** decomposed from the original 50 interactive reasoning tasks. These task have only **small visual difference or instruction difference** from COIN-Primitive. For example, in the Pick-Cube-WithDoor task, we build Pick-Cube-WithDoor-OD(need to only open the door) and Pick-Cube-WithDoor-PC tasks(the door have been opened initially, robot only need to pick the cube and put it on the marker). While there success rate have great decay. 
| Model | Mid-term Success Rate | Mid-term finish tasks | Primitive Success Rate | Generalization Score |
|-------|---------------------|----------------|----------------------|---------------------|
| CogACT | 0.015 | 3 | 0.19 | 0.0789 |
| PI0 | 0.065 | 4 | 0.161 | 0.4037 |
| GR00T | 0.0 | 0 | 0.167 | 0.0 |

And finally find that even these seemingly minor changes cause VLA models to struggle significantly, revealing that **visual and instruction generalization must be solved first** before tackling full interactive reasoning.

**Future Expansion:** 
- Plan to expand beyond current 20 mid-term tasks
- Framework enables community contributions for richer task diversity
- Supports curriculum-based learning strategies for models improvement

### Q3: The evaluation relies primarily on binary success metrics (SR/CSR), which are insufficient to capture partial task completion, the quality of subtask execution, or more nuanced failure behaviors. Introducing more fine-grained metrics would significantly enhance the benchmark's diagnostic value.

**Response:**

We agree that binary metrics are limiting and have implemented **comprehensive fine-grained evaluation metrics** that provide rich diagnostic insights beyond simple success/failure:
**1. Trajectory Stability Score** - Measures action quality and smoothness:
   - **Formula**: $S_{traj} = 0.3 \cdot S_{vel} + 0.3 \cdot S_{acc} + 0.2 \cdot S_{jerk} + 0.2 \cdot S_{pos}$
   - **Components**: Each component uses $\text{Smooth}(x) = \exp(-CV_x)$ where $CV_x = \frac{\sigma_x}{\mu_x}$ (coefficient of variation)
	 - $S_{vel}, S_{acc}, S_{jerk}, S_{pos}$ apply this function to velocity, acceleration, jerk (3rd derivative), and position respectively
   - **Weight Design**: Velocity and acceleration (0.3 each) are primary indicators of VLA action explosions; jerk (0.2); position (0.2) provides additional stability assessment
   - **Interpretation**: Higher scores indicate better trajectory stability 

**2. Gripper Control Stability** - Assesses manipulation quality:
   - **Formula**: $S_{gripper} = 0.4 \cdot S_{smooth} + 0.3 \cdot S_{freq} + 0.3 \cdot S_{coord}$
   - **Components**: 
	 - $S_{smooth} = \exp(-\text{abrupt changes})$ penalizes sudden gripper state transitions
	 - $S_{freq} = \exp(-\frac{N_{changes}}{N_{expected}})$ where $N_{changes}$ is actual gripper actions, $N_{expected}$ is task-appropriate frequency
	 - $S_{coord} = \frac{1}{N}\sum \text{coordination score}$ analyzing arm deceleration before gripper close and acceleration after gripper open in 10-frame windows
   - **Weight Design**: Smoothness (0.4) is most critical for erratic behavior detection; frequency and coordination (0.3 each) balance overuse and timing accuracy
   - **Interpretation**: Higher scores indicate better gripper control quality

**3. Task Decomposition Quality** - Evaluates planning capabilities:
   - **Method**: GPT-4o assessment of completeness, correctness, clarity
   - **Interpretation**: Higher scores indicate better planning quality


**4. Generalization Capability Score**: Evaluates model adaptability through controlled task variations:
- **Method**: We extracted 20 mid-term tasks from COIN-Interactive tasks, creating variants of primitive tasks
- **Formula**: $S_{gen} = \frac{{\text{SR}}_{\text{mid-term}}}{\text{SR}_{\text{primitive}}}$ where success rates are averaged across all tasks in each category 
- **Interpretation**: Scores close to 1.0 indicate good generalization; lower scores reveal generalization failures


**Empirical Results**: We applied these metrics to analyze VLA across different task difficulties:

| Model | Task Type | Trajectory Stability | Gripper Stability |
|-------|-----------|---------------------|-------------------|
| **CogACT** | Primitive | **0.150 ± 0.055** | **0.872 ± 0.134** |
| **CogACT** | Composition | **0.138 ± 0.039** | **0.796 ± 0.136** |
| **CogACT** | Interactive | **0.146 ± 0.041** | **0.782 ± 0.141** |
| GR00T | Primitive | 0.082 ± 0.015 | 0.318 ± 0.116 |
| GR00T | Composition | 0.086 ± 0.002 | 0.327 ± 0.058 |
| GR00T | Interactive | 0.084 ± 0.002 | 0.294 ± 0.050 |
| PI0 | Primitive | 0.084 ± 0.067 | 0.440 ± 0.198 |
| PI0 | Composition | 0.035 ± 0.043 | 0.465 ± 0.253 |
| PI0 | Interactive | 0.061 ± 0.050 | 0.440 ± 0.219 |


What's more  the **Generalization Capability Evaluation** have been shown on the response for Q2.

**Key Insights for Incremental Improvements:**

1. VLA related
- **Overall Performance Issues**: All models exhibit significant challenges in trajectory smoothness and gripper control, reflecting fundamental learning difficulties from the dataset that expose limitations in current VLA training paradigms
- **Generalization Limitations**: Current models demonstrate weak generalization capabilities, particularly when trained on non-large-scale datasets, affecting both visual and instruction-following generalization. 
- **Task Complexity Impact**: Performance degradation is evident as task complexity increases from primitive to interactive, highlighting the scalability challenges of current VLA approaches
2. VLM related

- **VLM-VLA Interface Challenges**: The connection between VLM reasoning and underlying VLA execution shows significant issues, particularly when instructions serve as the intermediate medium. VLAs may not have encountered specific instructions during training or exhibit excessive sensitivity to instruction variations, leading to performance degradation
- **Limited Historical Context Utilization**: VLMs demonstrate insufficient ability to leverage historical information effectively. While theoretically VLMs should improve instruction switching decisions over time, Figure 13 shows relatively flat performance curves, indicating poor utilization of accumulated context for adaptive reasoning
- **CodeAsPolicy Feedback Limitations**: Analysis from Figure 5 reveals that while CodeAsPolicy approaches can achieve relatively consistent results on individual tasks, their task coverage is quite limited. A critical issue is the **lack of feedback mechanisms** - once the high-level structure plans a trajectory, it typically remains static. However, our tasks require dynamic re-planning after completing certain stages, highlighting the need for researchers to incorporate feedback and closed-loop considerations into hierarchical planning systems

These metrics provide the nuanced evaluation you requested, revealing incremental improvements and failure modes that binary success rates would miss.

---

## Reviewer VAih

### Q1: The authors introduce a low-cost mobile AR teleoperation pipeline, but they don't thoroughly evaluate its reliability, the fidelity of the captured data, or how it compares to other collection methods. Such analyses are crucial, since the community impact of a dataset hinges on its quality.

**Response:**

We thank you for your constructive feedback and provide comprehensive response of our AR teleoperation system:

**System Reliability and Foundation:**
- Built on **commercial AR solutions** (ARCore/ARKit) with mature VIO algorithms
- Inspired by validated project like the Mujoco_AR mentioned in the paper with demonstrated robustness
- We provide Cross-device compatibility for Android and IOS phone ensuring broad accessibility and stability

**Data Quality Validation:**
- **Successful VLA training**: Models trained on our data consistently perform intended tasks, model could achieve 12 or more tasks success, validating the quality of the data .
- **High fidelity demonstrations**: 1,000+ carefully curated trajectories with validation of the trajectory and gripper stability. What's more ,we did careful replay provied in Maniskill , **90% of original data could be replayed**.

**Advantages over Traditional Methods:**
- **Accessibility**: No specialized hardware required, any Android or IOS phone published after 2016 can be used.
- **Scalability**: Easy deployment across different environments, only need to install the app on phone and use wifi to connected to PC.
- **Quality**: Comparable data fidelity at fraction of the cost, and **90% of original data could be replayed**.
### Q2: In Table 1, the paper only lists a few qualitative feature comparisons with prior datasets. It lacks detailed, quantitative metrics—e.g., overall scale, number of demonstrations, task diversity, and data quality—information that's essential for assessing the dataset's true value.

**Response:**

You're absolutely right about the need for detailed quantitative comparisons. Here's a comprehensive comparison with existing benchmarks:

| Benchmark | Tasks | Demonstrations | Avg. Steps |
|-----------|-------|----------------|------------|
| **COIN** | **50** | **1,000+** | **988.9** |
| RLBench | 100 | 100 per task | 180.2 |
| CALVIN | 34 | 200K | 30 |
| Libero | 130 | 50 per task | 77.3 |
| ARNOLD | 8 | 40 | 125.8 |
| VLABench | 100 | 163 | 157.2 |
| Behavior-1K | 1000 | 9331 | N/A |

While COIN may not have the largest scale in terms of raw task count or demonstration volume, our **core contribution lies in the systematic decomposition and construction of interactive reasoning tasks commonly encountered in daily life**. Each of our 50 tasks represents a carefully designed challenge that requires extensive interactive design and multi-modal reasoning capabilities which is called **interactive reasoning**. What's more, our **988.9 average steps per task** far exceeds all other benchmarks, demonstrating that COIN prioritizes task depth and complexity over sheer volume. Each task involves intricate multi-step reasoning that cannot be achieved through simple primitive skill combinations. And at least 50% tasks of the COIN-50 have multiple solutions with solution diversity which will be introduced more in the response for Q3.

### Q3: While the authors stress that the large number of steps demands strong reasoning, if the dataset lacks procedural diversity, models could simply memorize repeated step sequences instead of performing genuine causal inference. For each goal, multiple valid action plans should exist; a dataset with overly uniform patterns cannot adequately evaluate reasoning capability.

**Response:**

We address the procedural diversity concern with concrete Multi-solution tasks, Firstly our task are **test-only** which are composition of COIN-Primitive tasks and could only get related visual feature and skill learning on training stage with our dataset. While they will face totally different tasks when evaluating on such COIN-50 tasks. For instance, the model might see the door ,cabinet solely on the training data, while didn't ever see the composition of them on the training data. 

What's more, for these COIN-50 tasks, they indeed have multiple valid solutions.
**50%+ of tasks** exhibit substantial trajectory diversity with multiple valid solution paths.For example, The task Tabletop-Find-Dice need the agent find the dice which have 2-4 on the opposite face, there are few solutions for this task. It could view all the face on the desktop and find the right one, or put all the dice to the marker directly. The task Tabletop-finish-Hanobi is the task need to put all the circle on the right stick with order, there is no limit for the space using, so the robot could directly move them to some space and stack all of them. And for such tasks we provide **Diverse workflow annotations**  to highlight the richness of possible approaches, For every tasks with diverse workflow, we provide multiple workflow using language which is useful to solve the final task.

### Q4: Minor Issues - The bottom-right inset of Figure 2 is illegible. Equations are missing terminal punctuation.

**Response:**
Thank you for pointing out these issues. We will address them in the revision:
- Figure 2 inset will be enlarged and clarified in the revision
- Mathematical notation will be corrected with proper punctuation
---

## Reviewer hSnT
### Q1: The chosen binary success metrics (success/failure) are overly simplistic, lacking nuances like partial successes or graceful recovery. Such simplistic metrics may obscure incremental improvements critical to interactive reasoning tasks. It would be helpful to introduce more nuanced or graded metrics (e.g., progress scores, task-specific milestones, or time efficiency), and this might provide deeper insights into models' capabilities and behaviors.

**Response:**

We completely agree and have implemented **comprehensive fine-grained evaluation metrics** that directly address your concerns about partial successes, graceful recovery, and incremental improvements:

**1. Trajectory Stability Score** - Measures action quality and smoothness:
   - **Formula**: $S_{traj} = 0.3 \cdot S_{vel} + 0.3 \cdot S_{acc} + 0.2 \cdot S_{jerk} + 0.2 \cdot S_{pos}$
   - **Components**: Each component uses $\text{Smooth}(x) = \exp(-CV_x)$ where $CV_x = \frac{\sigma_x}{\mu_x}$ (coefficient of variation)
	 - $S_{vel}, S_{acc}, S_{jerk}, S_{pos}$ apply this function to velocity, acceleration, jerk (3rd derivative), and position respectively
   - **Weight Design**: Velocity and acceleration (0.3 each) are primary indicators of VLA action explosions; jerk (0.2); position (0.2) provides additional stability assessment
   - **Interpretation**: Higher scores indicate better trajectory stability 

**2. Gripper Control Stability** - Assesses manipulation quality:
   - **Formula**: $S_{gripper} = 0.4 \cdot S_{smooth} + 0.3 \cdot S_{freq} + 0.3 \cdot S_{coord}$
   - **Components**: 
	 - $S_{smooth} = \exp(-\text{abrupt changes})$ penalizes sudden gripper state transitions
	 - $S_{freq} = \exp(-\frac{N_{changes}}{N_{expected}})$ where $N_{changes}$ is actual gripper actions, $N_{expected}$ is task-appropriate frequency
	 - $S_{coord} = \frac{1}{N}\sum \text{coordination score}$ analyzing arm deceleration before gripper close and acceleration after gripper open in 10-frame windows
   - **Weight Design**: Smoothness (0.4) is most critical for erratic behavior detection; frequency and coordination (0.3 each) balance overuse and timing accuracy
   - **Interpretation**: Higher scores indicate better gripper control quality

**3. Task Decomposition Quality** - Evaluates planning capabilities:
   - **Method**: GPT-4o assessment of completeness, correctness, clarity
   - **Interpretation**: Higher scores indicate better planning quality

**4. Generalization Capability Score**: Evaluates model adaptability through controlled task variations:
- **Method**: We extracted 20 mid-term tasks from COIN-50 tasks, creating variants of primitive tasks
- **Formula**: $S_{gen} = \frac{\text{SR}_{\text{mid-term}}}{\text{SR}_{\text{primitive}}}$ where success rates are averaged across all tasks in each category 
- **Interpretation**: Scores close to 1.0 indicate good generalization; lower scores reveal generalization failures

**Empirical Results**: We applied these metrics to analyze VLA across different task difficulties:

| Model | Task Type | Trajectory Stability | Gripper Stability |
|-------|-----------|---------------------|-------------------|
| **CogACT** | Primitive | **0.150 ± 0.055** | **0.872 ± 0.134** |
| **CogACT** | Composition | **0.138 ± 0.039** | **0.796 ± 0.136** |
| **CogACT** | Interactive | **0.146 ± 0.041** | **0.782 ± 0.141** |
| GR00T | Primitive | 0.082 ± 0.015 | 0.318 ± 0.116 |
| GR00T | Composition | 0.086 ± 0.002 | 0.327 ± 0.058 |
| GR00T | Interactive | 0.084 ± 0.002 | 0.294 ± 0.050 |
| PI0 | Primitive | 0.084 ± 0.067 | 0.440 ± 0.198 |
| PI0 | Composition | 0.035 ± 0.043 | 0.465 ± 0.253 |
| PI0 | Interactive | 0.061 ± 0.050 | 0.440 ± 0.219 |

**Generalization Capability Evaluation**: We evaluated model generalization using our 20 mid-term tasks extracted from COIN-50. The results demonstrate significant generalization challenges:

| Model | Mid-term Success Rate | Mid-term finish tasks | Primitive Success Rate | Generalization Score |
|-------|---------------------|----------------|----------------------|---------------------|
| CogACT | 0.015 | 3 | 0.19 | 0.0789 |
| PI0 | 0.065 | 4 | 0.161 | 0.4037 |
| GR00T | 0.0 | 0 | 0.167 | 0.0 |

**Key Findings**:
1. VLA related
- **Overall Performance Issues**: All models exhibit significant challenges in trajectory smoothness and gripper control, reflecting fundamental learning difficulties from the dataset that expose limitations in current VLA training paradigms, meaning we need to pay more attention to how to do better action chunking or action tokenization.
- **Generalization Limitations**: Current models demonstrate weak generalization capabilities according to the generation capability evalution, particularly when trained on non large-scale datasets, affecting both visual and instruction-following generalization causing the performance degradation along task complexity increases from primitive to interactive. So we need to find better ways to do the generalization. 
2. VLM related
- **VLM-VLA Interface Challenges**: The connection between VLM reasoning and underlying VLA execution 
relay on language only. Although the VLM could did reasonable task decomposition , the low level VLA could not accpet and process well, meaning we need more attention on the VLM-VLA interface like Pi0.5[1]. - **Limited Historical Context Utilization**: VLMs demonstrate insufficient ability to leverage historical information effectively. While theoretically VLMs should improve instruction switching decisions over time, Figure 13 shows relatively flat performance curves, indicating poor utilization of accumulated context for adaptive reasoning
- **CodeAsPolicy Feedback Limitations**: Analysis from Figure 5 reveals that while CodeAsPolicy approaches can achieve relatively consistent results on individual tasks, their task coverage is quite limited. A critical issue is the **lack of feedback mechanisms** - once the high-level structure plans a trajectory, it typically remains static. However, our tasks require dynamic re-planning after completing certain stages, highlighting the need for researchers to incorporate feedback and closed-loop considerations into hierarchical planning systems

These metrics provide fine-grained analysis of different failure modes while maintaining the diagnostic value of our challenging benchmark.

### Q2: The proposed tasks generally focus on short-term manipulative reasoning (within a few hundred steps). Long-term reasoning and planning spanning extended periods or involving more complex temporal reasoning are insufficiently explored.

**Response:**
We thank you for your constructive feedback and understand it may arise from the compact presentation of our temporal analysis. We would like to clarify that COIN actually demonstrates substantially longer and more complex temporal reasoning than existing benchmarks. The detailed comparison may not be immediately apparent from the main figures, so we provide the specific data here, As shown in Figure 2 (top-left panel):

| Benchmark | Average Length |
|-----------|----------------|
| ManiSkill | 52.3 |
| Libero | 77.3 |
| RoboCasa A. | 123 |
| ARNOLD | 125.8 |
| VLABench Primitive | 157.2 |
| RLBench | 180.2 |
| RoboCasa Composition | 371.9 |
| RLBench Composition | 502.5 |
| **COIN-50** | **988.9** |


As shown on the above table, our benchmark features longer trajectories with rich temporal dependencies and subgoal structures.
Among our 50 tasks, at least **40 exhibit strong temporal dependencies** where earlier information/actions are essential for later stages, ensuring models must reason over extended horizons beyond local cues. 
For example, in the task **Pick-Cylinder-WithObstacle**, the robot could not directly pick the center cylinder, while the arounding cylinder have different type, some of them are static, some are dynamic, the robot need to interact with them one by one and decide how to move the dynamic one, and then find the right position to pick the center cylinder. 

**Multi-Level Subgoal Decomposition:** Our design philosophy emphasizes both interactive reasoning depth and genuine temporal complexity. Each COIN task is composed of multiple subgoals, requiring models to:
- Plan across multiple subgoals (typically 2–5 per task)
- Maintain and update goal hierarchies throughout extended interactions
- Adapt subgoal execution based on environmental feedback
The following table summarizes the distribution of subgoal counts across tasks in COIN-50:

| Subtask Length | Percentage | Number of Tasks |
|----------------|------------|----------------|
| 2 | 0.36 | 18 |
| 3 | 0.46 | 23 |
| 4 | 0.12 | 6 |
| 5 | 0.06 | 3 |


### Q3: The diversity in object categories, materials, textures, and physical interactions is relatively limited. More diverse object and scene categories would strengthen the generalization claims of the benchmark.
**Response:**

Thank you for your suggestion. We agree that object diversity is important for generalization. While our primary focus was constructing a controlled benchmark evaluating **test-time adaptation** and emphasizing **reasoning process diversity**—including reasoning targets (friction, mass, shape), object relationships (containment, comparison), and robot-object interactions and so on, so we did not pay too much attention to the object diversity. But we recognize the importance of broader diversity. Our assets are selected from Objaverse[2] and PartMobility[3] supporting easy expansion, and we plan to add an automatic asset integration pipeline with diverse object format input to support community usage.
### Q4: While the paper robustly identifies model failures (e.g., inadequate adaptation, visual-motor mismatches), it offers limited guidance or strategies on how these failures might be practically mitigated or resolved.
**Response:**
Thank you for your suggestion. Our updated analysis with new metrics in response to Q1 highlights key directions for addressing observed model failures, and we provide some detail improve direction for these failure :

1. **Urgent Need to Improve VLA Generalization**: According to the analysis of the generalization table, Even small new objects (e.g., a cube) significantly degrade VLA performance, severely restricting scalability and creating fundamental bottlenecks for interactive reasoning. 

2. **Action Stability and Gripper Timing**: Both are critical for pick-place operations. Instabilities drastically affect interactive manipulation success. As shown on the stabilty analysis, **CogACT** could maintain better action stability, this might be related to the **action ensemble stragety**. What's more, some related work have provide more insight for this problem, Dyna-1 with RL could improve the stability of VLA which might be learnt from human data. 

3. **Adaptive CodeAsPolicy**: Current open-loop methods lack adaptability. We recommend online closed-loop strategies and CoT reasoning in VLMs, enabling models to summarize execution feedback for better decision-making beyond simple backtracking. To achieve this, it need the VLM to achieve video or temopral memorization strategy.

4. **HVLA Task Switching**: Models need sophisticated task switching modules. Instruction chaining should incorporate reflection on recent changes (e.g., video segment summaries) to guide future planning. What's more, we need consider how to build better hierarchical integration between VLA and VLMs like [1]

These recommendations, informed by our expanded metrics, provide concrete strategies for improving model robustness in interactive reasoning settings.
### Q5: No comparison or baseline involving human performance or human-level task demonstrations is provided. Such a baseline would help contextualize the difficulty level of tasks and set meaningful targets for model performance.
**Response:**
Indeed, we acknowledge that this aspect was previously underexplored. To address this, we selected 10 representative tasks from COIN-50, ensuring coverage of long-horizon reasoning, tool use, and other key challenges. We then recruited 3 participants who have get their B.S. degree. with no prior exposure to our tasks and asked each to attempt every task twice via gello, recording their success rates.

**Human Baseline Results**: Each participant completed 10 tasks with 2 attempts per task, providing a total of 20 trials per participant across diverse interactive reasoning scenarios:

| Participant | Success Rate | Performance Level | Real SR| 
|-------------|--------------|------------------|----|
| A | 30% | 6/20 tasks completed | 100% |
| B | 50% | 10/20 tasks completed | 100% |
| C | 40% | 8/20 tasks completed | 100% |
| **Average** | **40%** | **8/20 tasks completed** | **100%** |

The 40% average human success rate in simulation is caused by manually operating robots using **teleoperation**. To elimate the error caused by the second-hand view and operation, we built up a similar test env in real-world, and let these to did the same test, the success rate is 100%. 
As shown in the table robotic systems typically achieve success rates below 3%, demonstrating that current robots perform far below human capabilities


# Reference
[1] Physical Intelligence et al. "π_0.5: A Vision-Language-Action Model with Open-World Generalization." 
[2] Deitke, Matt et al. "Objaverse: A Universe of Annotated 3D Objects." 
[3] Xiang, Fanbo et al. "SAPIEN: A SimulAted Part-based Interactive ENvironment." 
[4] Dyna Robotic et al. "Dynamism v1 (DYNA-1) Model: A Breakthrough in Performance and Production-Ready Embodied AI"
---

## Reviewer 8aJw

### Q1: Experiments focus on tabletop scenarios, neglecting dynamic environments (e.g., outdoor or cluttered spaces). This limits the benchmark's generalizability to real-world robotics tasks. In addition, tasks assume fixed environmental setups, lacking dynamic elements (e.g., moving obstacles or changing object states). Real-world interactions often require handling unpredictability, which COIN does not fully capture. It is interesting to introduce dynamic environments (e.g., kitchen with running water, outdoor construction sites) to test adaptability. This would better reflect real-world manipulation challenges.

**Response:**
Thank you for your constructive suggestions. We fully acknowledge that real-world experiments are crucial for embodied intelligence research and are essential for demonstrating the practical value of robotics. However, the primary focus of our work is to build up an **interactive-reasoning environments** which need lots of interaction and reasoning. To this end, we have constructed a rich set of interactive tasks and systematically tested model performance within these scenarios.

While regarding your point about real-robot experiments and more dynamic features, there are some related response for the detail plans.

Firstly for the **simulation experiment part**, tasks in COIN-50 have involved some dynamic feature, for example, the dynamic physical properties(friction, velocity, initial position), and the nowadays H-VLA and CodeAsPolicy could not achieve reasonable results (with models could only achieve less than 3% SR) according to the Table 2 in the paper, so we think it might be more prioritied to solve the interaction and reasoning ability of models on such less-dynamic and fixed tabletop scenario. And we will import more scenario and dynamic feature in the future if the models could achieve reasonable results. 
Secondly for the **real-world experiments part**, we hope to built up an novel test environment for current models which could be easily accessible for every researchers, so our benchmark are constructed on the simulation like [1] [2] [3]. What's more, our environments are designed to be easily transferable to physical hardware. We provide STL files for all assets used in our benchmark, we provide the different mass, friction version and corresponding guide to build such environment, enabling any researcher to 3D print the necessary models and conduct real-world experiments.
### Q2: While COIN uses physics-based simulation, it may not fully replicate real-world robot dynamics (e.g., sensor noise, actuator limitations). This could affect transferability to physical robots. It is cool to integrate COIN with physical robot platforms (e.g., Franka Emika Panda) for real-world validation. This could involve hybrid simulation-reality setups to reduce the sim-to-real gap.
Thank you very much for your question and suggestion. As mentioned in our previous response regarding real-robot experiments, users can leverage the provided STL files to 3D print physical models and conduct corresponding real-world experiments.

Additionally, our entire platform is built on ManiSkill and SAPIEN, both of which are highly extensible and support a wide range of commonly used sensors and configurations. Within our environment, it is possible to simulate various sensor noises, physical lighting conditions, and actuator limitations. These capabilities help to partially bridge the sim-to-real gap, and, in fact, some recent studies have already demonstrated successful sim-to-real transfer using this environment [1](#references).

---

## References

[1] Stone Tao et al. "ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI." *Robotics: Science and Systems*, 2025.
[2] Li, Xuanlin et al. "Evaluating Real-World Robot Manipulation Policies in Simulation." arXiv:2405.05941, 2024.
[3] Liu, Bo et al. "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning." arXiv:2306.03310, 2023.
[4] Zhou, Zhiyuan et al. "AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World." arXiv:2503.24278, 2025.
