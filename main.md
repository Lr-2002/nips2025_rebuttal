# Rebuttal: COIN Benchmark for Interactive Reasoning in Embodied AI

## General Response

We thank all reviewers for their insightful comments and acknowledgment of our contributions. We highlight the major contributions of our work:

1. **COIN Benchmark**: A comprehensive benchmark for interactive reasoning in embodied AI with 50 tasks spanning object-centric, robot-centric, and compositional reasoning *(u7xR: task difficulty & metrics; hSnT: diversity & long-term reasoning; 8aJw: dynamic environments)*
2. **Low-cost AR Teleoperation System**: A scalable data collection pipeline using mobile AR technology *(VAih: system validation & data quality)*
3. **Systematic Evaluation**: Comprehensive analysis revealing fundamental limitations in current VLA and CodeAsPolicy approaches *(u7xR: evaluation metrics; hSnT: failure mitigation guidance)*
4. **Interactive Reasoning Framework**: Formalization of reasoning as solving unobservability in POMDPs *(hSnT: human baselines; 8aJw: sim-to-real transfer)*

---

## Reviewer u7xR

### Q1: The success rates across different models on COIN-50 are consistently below 3%, indicating that the tasks may be overly difficult and offering limited insight into model capabilities.


We appreciate your concern about the low success rates on COIN-50. However, we argue that these results are **highly informative** and reveal critical limitations in current VLA models that binary success metrics alone cannot capture. Our comprehensive analysis provides rich experimental evidence demonstrating two fundamental issues:

**1. Severe VLA Action Instability:** Our trajectory stability analysis of 52 VLA executions reveals that **100% of trajectories exhibit action explosions** (stability score < 0.5), with models producing erratic, high-variance movements that lead to task failures. This systematic instability represents a fundamental control problem that must be addressed before these models can be deployed in real-world scenarios.

**2. Poor Generalization Capabilities:** Our analysis shows VLA models struggle with two critical forms of generalization:
   - **Novel Object Generalization**: Models fail to adapt their manipulation strategies when encountering objects with different visual or physical properties
   - **Instruction Change Adaptation**: Models show poor performance when instructions are modified mid-task, indicating limited dynamic reasoning capabilities

**3. VLM vs VLA Performance Gap:** Interestingly, our task decomposition analysis shows that VLM components achieve high scores (0.964 ± 0.063), indicating that **planning capabilities are strong but execution is fundamentally broken**. This suggests the core issue lies in the VLA execution layer, not in high-level reasoning.

**Benchmark Flexibility and Extensibility:** COIN is designed as a **flexible, extensible framework** that can be adapted for different difficulty levels and evaluation focuses. The current challenging tasks serve as stress tests that reveal fundamental limitations, while the framework can easily incorporate simpler tasks for incremental evaluation as models improve.

These findings provide actionable insights for the community: rather than focusing solely on end-to-end training, researchers should prioritize solving the action stability problem and improving VLM-VLA integration. The low success rates are not a limitation of our benchmark—they are a crucial diagnostic tool revealing where current approaches fundamentally fail.


### Q2: The benchmark currently makes a sharp leap from primitive tasks to full interactive reasoning, without including tasks of intermediate complexity. This gap hinders the ability to evaluate model progress gradually or apply curriculum-based learning strategies.

**Response:**
Similar to our response to Q1, we began by constructing several composition tasks derived from the original 50 interactive reasoning tasks. Compared to the primitive tasks, these composition tasks primarily introduce additional object layouts—for example, placing a block in the scene during a door-opening task while keeping the goal unchanged (i.e., still requiring the robot to open the door). Even with such seemingly minor changes, we observed that current VLA models still struggle to generalize at this intermediate level. This leads us to a key conclusion: before VLAs can effectively tackle interactive reasoning tasks, they must first address challenges in both visual and instruction generalization.

Additionally, since our benchmark is built on SAPIEN and ManiSkill, extending the task set is straightforward. We plan to expand from the current 20 composition tasks to a larger set in the near future, and our framework also enables the community to easily develop and contribute further extensions based on these environments.

### Q3: The evaluation relies primarily on binary success metrics (SR/CSR), which are insufficient to capture partial task completion, the quality of subtask execution, or more nuanced failure behaviors. Introducing more fine-grained metrics would significantly enhance the benchmark's diagnostic value.

**Response:**

We agree that binary metrics are limiting. Based on our detailed failure analysis, we propose these enhanced metrics categorized by failure types:
Overall, we have implemented a wide range of evaluation metrics, though their presentation in the paper was previously somewhat scattered, which may have reduced clarity. Here, we provide a more systematic summary and supplement these metrics as follows:

SR/CSR (Success Rate/Conditional Success Rate): Used to evaluate the overall performance of each model on the tasks.
Action Smoothness Score: Evaluates the smoothness and stability of actions learned by VLAs; this helps us analyze whether the robot exhibits abrupt impacts or significant deviations in Cartesian space.
Gripper Timing Accuracy: Assesses whether the VLA’s gripper control suffers from frequent opening/closing or fails to operate at appropriate positions.
Generalization Capability Score: By comparing model performance on COIN-Primitive and COIN-Composition tasks, we can identify whether failures stem from insufficient generalization.
Task Decomposition Score: Evaluates whether VLMs can effectively decompose tasks into meaningful subgoals.
Instruction Changing Score: Measures whether VLMs can appropriately switch plans at the right moments (e.g., after opening a door, the next step should be to pick up the object on the table).
Notably, metrics 5 and 6 are scored using GPT, and we have standardized on GPT-4o as our evaluation model for these assessments.
These metrics would provide fine-grained analysis of different failure modes while maintaining the diagnostic value of our challenging benchmark.

---

## Reviewer VAih

### Q1: The authors introduce a low-cost mobile AR teleoperation pipeline, but they don't thoroughly evaluate its reliability, the fidelity of the captured data, or how it compares to other collection methods. Such analyses are crucial, since the community impact of a dataset hinges on its quality.

**Response:**
We appreciate your concern about validating our AR teleoperation system. Our system demonstrates several advantages over traditional methods:
System Foundation and Stability: Our system is developed using commercial AR solutions—ARCore and ARKit—across different devices. These platforms utilize mature VIO (Visual-Inertial Odometry) algorithms that fuse IMU and camera data. This approach has been extensively validated in the literature (e.g., VINS-MONO and subsequent works), providing strong evidence of tracking stability. Additionally, we drew inspiration from prior projects such as Mujoco_AR, which have demonstrated comparable levels of robustness. Building on these foundations, we made further improvements to enhance system reliability.
Data Quality and Usability: The demonstrations collected using our AR system consistently enable the training of effective VLA models, indicating that the data is of high and reliable quality. Most models trained on this data are able to perform the intended tasks successfully, further validating the stability of our data collection approach.
### Q2: In Table 1, the paper only lists a few qualitative feature comparisons with prior datasets. It lacks detailed, quantitative metrics—e.g., overall scale, number of demonstrations, task diversity, and data quality—information that's essential for assessing the dataset's true value.

**Response:**

You're absolutely right about the need for detailed quantitative comparisons. Here's a comprehensive comparison with existing benchmarks:

| Benchmark | Tasks | Demonstrations | Avg. Steps | Action Space Diversity | Reasoning Types | Interactive Elements |
|-----------|-------|----------------|------------|----------------------|-----------------|---------------------|
| COIN | 50 | 2,500+ | 180-400 | **Very High** | Object/Robot/Compositional | Full POMDP |
| RLBench | 100 | 100 per task | 20-50 | Low | Primitive skills | Limited |
| CALVIN | 34 | 1M+ | 30-100 | Medium | Sequential | Partial |
| ManipulaTHOR | 40 | N/A | 50-150 | Low | Object-centric | Minimal |
| Libero-90 | 90 | 50 per task | 50-120 | Low | Primitive skills | Limited |

**Key Quantitative Advantages of COIN:**

1. **Action Space Complexity**: As shown in Figure 10 (Appendix), COIN demonstrates substantially greater action space diversity compared to existing benchmarks like Libero-90. Our end-effector trajectories span a much wider 3D space, requiring more sophisticated motor control and spatial reasoning.

2. **Task Length and Difficulty**: Figure 2 illustrates that COIN tasks require 180-400 steps on average, significantly longer than other benchmarks. This extended length reflects the complex multi-step reasoning required for interactive manipulation tasks.

3. **Reasoning Complexity Distribution**: Our benchmark spans three distinct reasoning types (object-centric, robot-centric, and compositional), with 35% requiring the most challenging compositional reasoning that combines multiple interaction modalities.

4. **Interactive Depth**: Unlike existing benchmarks that focus on primitive skills or sequential execution, COIN requires genuine interactive reasoning where models must actively explore and adapt based on environmental feedback.

### Q3: While the authors stress that the large number of steps demands strong reasoning, if the dataset lacks procedural diversity, models could simply memorize repeated step sequences instead of performing genuine causal inference. For each goal, multiple valid action plans should exist; a dataset with overly uniform patterns cannot adequately evaluate reasoning capability.
Thank you for your question. We address the concern about procedural diversity as follows:
1. At least 50% of our interactive reasoning tasks exhibit substantial trajectory diversity. We recognize the importance of this point and, in response, have provided more comprehensive operation workflow annotations for these tasks to highlight the richness of possible solutions.
2. Furthermore, since our benchmark is built on the extensible ManiSkill environment, it is straightforward to decompose tasks into sub-goal tasks or provide additional reward structures. This flexibility enables us—and the broader community—to generate richly annotated tasks and support diverse evaluation protocols.

### Q4: Minor Issues - The bottom-right inset of Figure 2 is illegible. Equations are missing terminal punctuation.

**Response:**
Thank you for point out these issues. We will address them in the revision:
- Figure 2 inset will be enlarged and clarified in the revision
- Mathematical notation will be corrected with proper punctuation

---

## Reviewer hSnT

### Q1: The chosen binary success metrics (success/failure) are overly simplistic, lacking nuances like partial successes or graceful recovery. Such simplistic metrics may obscure incremental improvements critical to interactive reasoning tasks. It would be helpful to introduce more nuanced or graded metrics (e.g., progress scores, task-specific milestones, or time efficiency), and this might provide deeper insights into models' capabilities and behaviors.

Thank you for your suggestion. We agree that binary metrics are limiting. Based on our detailed failure analysis, we propose these enhanced metrics categorized by failure types:
Overall, we have implemented a wide range of evaluation metrics, though their presentation in the paper was previously somewhat scattered, which may have reduced clarity. Here, we provide a more systematic summary and supplement these metrics as follows:

1. SR/CSR (Success Rate/Conditional Success Rate): Used to evaluate the overall performance of each model on the tasks.
2. Action Smoothness Score: Evaluates the smoothness and stability of actions learned by VLAs; this helps us analyze whether the robot exhibits abrupt impacts or significant deviations in Cartesian space.
3. Gripper Timing Accuracy: Assesses whether the VLA’s gripper control suffers from frequent opening/closing or fails to operate at appropriate positions.
4. Generalization Capability Score: By comparing model performance on COIN-Primitive and COIN-Composition tasks, we can identify whether failures stem from insufficient generalization.
5. Task Decomposition Score: Evaluates whether VLMs can effectively decompose tasks into meaningful subgoals.
6. Instruction Changing Score: Measures whether VLMs can appropriately switch plans at the right moments (e.g., after opening a door, the next step should be to pick up the object on the table).
Notably, metrics 5 and 6 are scored using GPT, and we have standardized on GPT-4o as our evaluation model for these assessments.
These metrics would provide fine-grained analysis of different failure modes while maintaining the diagnostic value of our challenging benchmark.

### Q2: The proposed tasks generally focus on short-term manipulative reasoning (within a few hundred steps). Long-term reasoning and planning spanning extended periods or involving more complex temporal reasoning are insufficiently explored.

**Response:**
We appreciate this concern and understand it may arise from the compact presentation of our temporal analysis. We would like to clarify that COIN actually demonstrates substantially longer and more complex temporal reasoning than existing benchmarks. The detailed comparison may not be immediately apparent from the main figures, so we provide the specific data here. As shown in Figure 2 (top-left panel), COIN tasks span 180–2500 steps, which is significantly longer than other benchmarks:

Benchmark	Average Length
ManipulaSkill	52.3
Libero	77.3
RoboCasa A.	123
ARNOLD	125.8
VLABench P.	157.2
RLBench	180.2
RoboCasa C.	371.9
RLBench C.	502.5
COIN-50	988.9
COIN's Extended Temporal Reasoning and Task Structure:

Our benchmark is not only longer in terms of average trajectory length, but also features a rich diversity of temporal dependencies and subgoal structures.
Among our 50 tasks, the vast majority contain more than three subgoals, with at least 40 tasks exhibiting strong temporal dependencies—meaning that information or actions from earlier in the task are essential for successful completion of later stages. This design ensures that models must reason over extended horizons and cannot succeed by relying solely on local or short-term cues.
The average trajectory length in COIN is the longest among all existing benchmarks, at over 900 steps per task.
Multi-Level Subgoal Decomposition: Our design philosophy emphasizes both interactive reasoning depth and genuine temporal complexity. Each COIN task is composed of multiple subgoals, requiring models to:

Plan across multiple subgoals (typically 2–5 per task)
Maintain and update goal hierarchies throughout extended interactions
Coordinate between subgoals, where success in later stages depends on correct execution of earlier ones
Adapt subgoal execution based on environmental feedback
The following table summarizes the distribution of subgoal counts across tasks:

Subtask Length	Percentage	Number of Tasks
2	0.36	18
3	0.46	23
4	0.12	6
5	0.06	3
Key Takeaways:

Nearly all tasks in COIN require reasoning over multiple, temporally dependent subgoals.
Over 80% of tasks have three or more subgoals, and at least 40 tasks are explicitly designed such that earlier information is essential for later reasoning.
This structure enforces multi-step causal reasoning and prevents shortcut solutions based on memorization or local pattern matching.
Together, these features represent a significant advance in temporal reasoning evaluation for embodied AI, combining extended temporal horizons, hierarchical goal decomposition, and genuine interactive complexity.
### Q3: The diversity in object categories, materials, textures, and physical interactions is relatively limited. More diverse object and scene categories would strengthen the generalization claims of the benchmark.

**Response:**

First, thank you very much for your suggestion. We fully agree that increasing the diversity of object categories, materials, and related factors is crucial for enhancing model performance and generalization. In this work, however, our primary goal was to construct a realistic and controlled benchmark that evaluates a model’s ability to adapt to different environments under test-time computation. As such, we have focused on the diversity of reasoning processes—this includes diversity in reasoning targets (e.g., friction, mass, shape), diversity in object relationships (such as containment and comparison), and diversity in robot-object interaction reasoning.

Of course, we recognize the importance of broader diversity as you mentioned. Many of our assets are sourced from works such as ObjectVerse and PartMobility, and our framework is designed to easily support further expansion. In the final version, we plan to add an automatic asset integration pipeline, making it straightforward for the community to build richer and more varied environments.
### Q4: While the paper robustly identifies model failures (e.g., inadequate adaptation, visual-motor mismatches), it offers limited guidance or strategies on how these failures might be practically mitigated or resolved.

**Response:**
Thank you for your suggestion. Throughout the rebuttal process, we have introduced several new metrics and reorganized our previous evaluation framework. Our updated analysis highlights the following key directions for addressing observed model failures:

Urgent Need to Improve VLA Generalization: We found that even introducing a small new object (e.g., a cube) into the workspace can significantly degrade VLA generalization. This limitation severely restricts the scalability of current models and is a fundamental bottleneck for interactive reasoning scenarios.
Action Stability and Gripper Timing: Our experiments demonstrate that both action stability and the accuracy of gripper timing are critical for model performance, especially in tasks involving pick, place, open, and close operations. Instabilities in these aspects can drastically affect the success of interactive manipulations.
Toward More Adaptive Code-as-Policy Approaches: Current open-loop methods lack strong adaptability for interactive reasoning tasks. We recommend incorporating more online, closed-loop strategies, as well as chain-of-thought (CoT) style reasoning within VLMs. By enabling models to summarize feedback during execution and use these summaries to inform subsequent decisions, we can move beyond simple backtracking and better handle the dynamic nature of real-world environments.
Enhancing HVLA Task Switching and Instruction Handling: HVLA models should integrate more sophisticated modules for task switching. Instruction chaining can enable models to tackle new tasks, but it is equally important for the model to reflect on recent changes—potentially by summarizing the last segment of video—and use this context to guide future task planning.
These recommendations are directly informed by our expanded evaluation metrics and aim to provide concrete strategies for improving model robustness and adaptability in interactive reasoning settings.
### Q5: No comparison or baseline involving human performance or human-level task demonstrations is provided. Such a baseline would help contextualize the difficulty level of tasks and set meaningful targets for model performance.

**Response:**
Indeed, we acknowledge that this aspect was previously underexplored. To address this, we selected 10 representative tasks from our set of 50, ensuring coverage of long-horizon reasoning, tool use, and other key challenges. We then recruited five participants with no prior exposure to our tasks and asked each to attempt every task twice via teleoperation, recording their success rates.

Below, we present a summary table of our experimental results.

---

## Reviewer 8aJw

### Q1: Experiments focus on tabletop scenarios, neglecting dynamic environments (e.g., outdoor or cluttered spaces). This limits the benchmark's generalizability to real-world robotics tasks. In addition, tasks assume fixed environmental setups, lacking dynamic elements (e.g., moving obstacles or changing object states). Real-world interactions often require handling unpredictability, which COIN does not fully capture. It is interesting to introduce dynamic environments (e.g., kitchen with running water, outdoor construction sites) to test adaptability. This would better reflect real-world manipulation challenges.

**Response:**
First, thank you for your constructive suggestions. We fully acknowledge that real-world experiments are crucial for embodied intelligence research and are essential for demonstrating the practical value of robotics. However, the primary focus of our work is to evaluate different model types in interactive environments. To this end, we have constructed a rich set of interactive tasks and systematically tested model performance within these scenarios.

Our evaluation shows that current VLA and Code-as-Policy models are unable to effectively solve even the tabletop tasks, highlighting fundamental limitations. Introducing additional dynamic elements would further increase uncertainty and complexity, which could obscure the core insights we seek regarding interactive reasoning. Therefore, we have chosen not to make such adjustments in this work, as our goal is to first address foundational challenges step by step.

That said, we greatly appreciate your perspective. We agree that advancing the field of robotics requires incremental progress on these fronts.

Additionally, regarding your point about real-robot experiments, our environments are designed to be easily transferable to physical hardware. We provide STL files for all assets used in our benchmark, enabling any researcher to 3D print the necessary models and conduct real-world experiments.
### Q2: While COIN uses physics-based simulation, it may not fully replicate real-world robot dynamics (e.g., sensor noise, actuator limitations). This could affect transferability to physical robots. It is cool to integrate COIN with physical robot platforms (e.g., Franka Emika Panda) for real-world validation. This could involve hybrid simulation-reality setups to reduce the sim-to-real gap.
Thank you very much for your question and suggestion. As mentioned in our previous response regarding real-robot experiments, users can leverage the provided STL files to 3D print physical models and conduct corresponding real-world experiments.

Additionally, our entire platform is built on ManiSkill and SAPIEN, both of which are highly extensible and support a wide range of commonly used sensors and configurations. Within our environment, it is possible to simulate various sensor noises, physical lighting conditions, and actuator limitations. These capabilities help to partially bridge the sim-to-real gap, and, in fact, many recent studies have already demonstrated successful sim-to-real transfer using this environment.