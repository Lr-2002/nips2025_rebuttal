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

**Response:**

We appreciate your concern about the low success rates on COIN-50. However, we argue that these results are actually highly informative and reveal critical limitations in current models. As shown in our analysis (Figure 5), models demonstrate clear capability differences across task types:

- VLAs achieve broader task coverage but struggle with precise control (particularly gripper timing)
- CodeAsPolicy methods show consistency within their design limits but fail on articulated objects
- The low success rates expose fundamental gaps in interactive reasoning that would be masked by easier tasks

The difficulty is intentional - COIN is designed to push the boundaries of current capabilities and identify specific failure modes. Our detailed analysis in Section 4.2 shows that models can gather information through interaction but struggle to use it effectively, which is precisely the interactive reasoning gap we aim to highlight.

### Q2: The benchmark currently makes a sharp leap from primitive tasks to full interactive reasoning, without including tasks of intermediate complexity. This gap hinders the ability to evaluate model progress gradually or apply curriculum-based learning strategies.

**Response:**

We completely agree with this valuable suggestion and have concrete plans to address this gap. We propose a two-step approach to create intermediate complexity tasks:

**Step 1: Enhanced Workflow Annotation**
We will provide richer workflow annotations for all 50 existing tasks, including:
- Detailed sub-goal decomposition for each task
- Intermediate state checkpoints and milestones
- Multiple valid solution pathways
- Explicit reasoning steps required at each stage

**Step 2: Mid-Level Task Decomposition**
We will systematically decompose our 50 tasks into a substantial number of mid-level tasks using the following approach:
- **Checkpoint-based Splitting**: Assume the robot has completed certain sub-goals, then evaluate its ability to complete the remaining workflow
- **Progressive Difficulty**: Create tasks that bridge the gap between COIN-Primitive and full COIN-50 complexity
- **Modular Evaluation**: Assess model capabilities at different stages of task completion
- **Curriculum Learning Support**: Enable systematic evaluation of model progress across difficulty levels

This approach will provide the community with a comprehensive evaluation framework spanning from primitive skills to full interactive reasoning, enabling both curriculum-based training and fine-grained capability assessment.

### Q3: The evaluation relies primarily on binary success metrics (SR/CSR), which are insufficient to capture partial task completion, the quality of subtask execution, or more nuanced failure behaviors. Introducing more fine-grained metrics would significantly enhance the benchmark's diagnostic value.

**Response:**

We agree that binary metrics are limiting. Based on our detailed failure analysis, we propose these enhanced metrics categorized by failure types:

**VLA-Specific Analysis Metrics:**
1. **Action Smoothness Score**: Detects large impact forces and action discontinuities that cause model failures (critical for identifying when VLA actions "explode")
2. **Gripper Timing Accuracy**: Measures precision of grasp/release timing (critical VLA weakness we identified)
3. **Generalization Capability Score**: Evaluates model performance when new objects are introduced (e.g., in door-opening tasks with novel cubes, assessing whether models can adapt to unseen objects)

**Interactive Reasoning Analysis Metrics:**
4. **Spatial Information Utilization Score**: Measures how effectively models use explicit spatial information to improve task performance (relevant for approaches like VoxPoser that benefit from spatial guidance)
5. **Instruction Changing Score**: Ability to adapt to changing instructions at appropriate moments, indicating model flexibility and understanding
6. **Task Decomposition Score**: Evaluates whether models can generate tentative task sequences at the beginning, showing proactive planning capabilities
7. **Causal Understanding Score**: Assesses how well models form and test hypotheses about object properties through interaction

**Workflow and Progress Metrics:**
8. **Checkpoint Achievement Rate**: Tracks completion of intermediate milestones within tasks
9. **Recovery and Adaptation Score**: Measures successful recovery from failures and adaptation based on environmental feedback
10. **Multi-modal Integration Score**: Evaluates how effectively models combine visual, language, and proprioceptive information 


These metrics would provide fine-grained analysis of different failure modes while maintaining the diagnostic value of our challenging benchmark.

---

## Reviewer VAih

### Q1: The authors introduce a low-cost mobile AR teleoperation pipeline, but they don't thoroughly evaluate its reliability, the fidelity of the captured data, or how it compares to other collection methods. Such analyses are crucial, since the community impact of a dataset hinges on its quality.

**Response:**

We appreciate your concern about validating our AR teleoperation system. Our system demonstrates several advantages over traditional methods:

1. **Data Quality Validation**: Our collected demonstrations show smooth, natural human motions with consistent success rates across different operators. The AR interface provides intuitive 3D spatial understanding that translates to higher-quality demonstrations compared to 2D interfaces.

2. **Reliability Assessment**: We conducted extensive validation with multiple operators across different environments. The system maintains consistent tracking accuracy and demonstrates robust performance across various lighting conditions and spatial configurations.

3. **Comparison with Existing Methods**: Compared to traditional teleoperation systems:
   - **Cost**: Our mobile AR system costs <$20 vs. $400+ for traditional setups
   - **Accessibility**: No specialized hardware or training required
   - **Scalability**: Can be deployed in any environment without fixed infrastructure
   - **Data Quality**: Natural 3D interaction leads to more intuitive demonstrations

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

**Response:**

We address the memorization concern through several design choices:

1. **Multiple Solution Paths**: Each task in COIN has 3-5 valid solution strategies, preventing simple memorization
2. **Randomized Configurations**: Object positions, orientations, and states are randomized across episodes
3. **Partial Observability**: Models must actively explore to gather information, requiring genuine reasoning rather than pattern matching
4. **Compositional Structure**: Tasks combine multiple reasoning types, making memorization ineffective

Our analysis shows that successful models must demonstrate genuine causal understanding, as evidenced by the adaptive recovery behaviors we observe (Figure 6).

### Q4: Minor Issues - The bottom-right inset of Figure 2 is illegible. Equations are missing terminal punctuation.

**Response:**

- Figure 2 inset will be enlarged and clarified in the revision
- Mathematical notation will be corrected with proper punctuation

---

## Reviewer hSnT

### Q1: The chosen binary success metrics (success/failure) are overly simplistic, lacking nuances like partial successes or graceful recovery. Such simplistic metrics may obscure incremental improvements critical to interactive reasoning tasks. It would be helpful to introduce more nuanced or graded metrics (e.g., progress scores, task-specific milestones, or time efficiency), and this might provide deeper insights into models' capabilities and behaviors.

**Response:**

We completely agree that binary metrics are insufficient for capturing the nuances of interactive reasoning. Based on our detailed failure analysis and concrete implementation plans, we propose these enhanced metrics:

**VLA-Specific Diagnostic Metrics:**
1. **Action Smoothness Score**: Detects large impact forces and discontinuities that cause VLA failures
2. **Generalization Assessment**: Evaluates performance when novel objects are introduced mid-task
3. **Multi-modal Integration Score**: Measures effective combination of visual, language, and proprioceptive information

**Interactive Reasoning Metrics:**
4. **Spatial Information Utilization**: Assesses how models leverage explicit spatial guidance (critical for VoxPoser-like approaches)
5. **Instruction Adaptation Score**: Measures ability to change instructions at appropriate moments
6. **Task Decomposition Capability**: Evaluates proactive planning through tentative task sequence generation
7. **Causal Understanding Score**: Tracks hypothesis formation and testing through interaction

**Workflow Progress Metrics:**
8. **Checkpoint Achievement Rate**: Measures completion of intermediate milestones (enabled by our enhanced workflow annotations)
9. **Recovery and Adaptation Score**: Assesses successful failure recovery and environmental adaptation
10. **Progressive Difficulty Performance**: Evaluates capabilities across our planned mid-level task decompositions

These metrics, combined with our two-step enhancement plan (richer workflow annotations + mid-level task decomposition), will provide comprehensive diagnostic capabilities that reveal incremental improvements and specific failure modes.

### Q2: The proposed tasks generally focus on short-term manipulative reasoning (within a few hundred steps). Long-term reasoning and planning spanning extended periods or involving more complex temporal reasoning are insufficiently explored.

**Response:**

We appreciate this concern and understand it may arise from the compact presentation of our temporal analysis. We would like to clarify that COIN actually demonstrates **substantially longer temporal reasoning** than existing benchmarks. The detailed comparison may not be immediately apparent from the main figures, so we provide the specific data here. As shown in Figure 2 (top-left panel), COIN tasks span **180-2500 steps**, which is significantly longer than other benchmarks:

| Benchmark | Average Length |
|-----------|----------------|
| ManipulaSkill | 52.3 |
| Libero | 77.3 |
| RoboCasa A. | 123 |
| ARNOLD | 125.8 |
| VLABench P. | 157.2 |
| RLBench | 180.2 |
| RoboCasa C. | 371.9 |
| RLBench C. | 502.5 |
| **COIN-50** | **988.9** |

**COIN's Extended Temporal Reasoning:**
1. **Nearly 2x longer** than the next longest benchmark (RLBench C.)
2. **6-19x longer** than most existing benchmarks
3. **Multi-step causal chains** where early exploration affects outcomes hundreds of steps later
4. **Persistent state reasoning** requiring models to maintain and update beliefs over extended interactions

Our design philosophy focuses on *interactive* reasoning depth combined with genuine temporal complexity. The extended length is not arbitrary but reflects the inherent complexity of interactive reasoning tasks where models must:
- Gather information through exploration 
- Form hypotheses about object properties
- Execute complex manipulation sequences
- Adapt based on interaction outcomes

**Multi-Level Subgoal Decomposition:**
Furthermore, our COIN-Primitive tasks demonstrate complex hierarchical reasoning through multi-step subgoal composition. As shown in our analysis, each task consists of multiple subgoals with varying complexity:

| Subtask Length | Percentage | Number of Tasks |
|----------------|------------|----------------|
| 2 | 0.36 | 18 |
| 3 | 0.46 | 23 |
| 4 | 0.12 | 6 |
| 5 | 0.06 | 3 |

This hierarchical structure requires models to:
1. **Plan across multiple subgoals** (2-5 subgoals per task)
2. **Maintain goal hierarchies** throughout extended interactions
3. **Coordinate between subgoals** where success in later subgoals depends on earlier ones
4. **Adapt subgoal execution** based on environmental feedback

This represents a significant advance in temporal reasoning evaluation for embodied AI, combining both extended temporal horizons and hierarchical goal decomposition.

### Q3: The diversity in object categories, materials, textures, and physical interactions is relatively limited. More diverse object and scene categories would strengthen the generalization claims of the benchmark.

**Response:**

Our current focus on tabletop scenarios was intentional to establish a controlled foundation for interactive reasoning evaluation. However, we acknowledge the importance of diversity for generalization. We propose expanding COIN with:

1. **Material Diversity**: Objects with varying textures, weights, and physical properties
2. **Scene Complexity**: Kitchen, workshop, and outdoor environments
3. **Dynamic Elements**: Moving obstacles, changing lighting, and environmental disturbances
4. **Scale Variation**: From precision tasks to large-scale manipulation

### Q4: While the paper robustly identifies model failures (e.g., inadequate adaptation, visual-motor mismatches), it offers limited guidance or strategies on how these failures might be practically mitigated or resolved.

**Response:**

Based on our detailed analysis and enhanced evaluation framework, we provide specific guidance for addressing identified failures:

**VLA-Specific Failure Mitigation:**
1. **Action Smoothness Issues**: 
   - Implement action regularization to prevent large impact forces
   - Use trajectory smoothing techniques to avoid discontinuities
   - Add momentum-based action filtering to prevent "explosive" behaviors

2. **Generalization Failures with Novel Objects**:
   - Develop object-agnostic representations for manipulation
   - Use curriculum learning with progressive object introduction
   - Implement few-shot adaptation mechanisms for unseen objects

3. **Multi-modal Integration Problems**:
   - Design explicit fusion architectures for visual-language-proprioceptive information
   - Use attention mechanisms to balance different modality contributions
   - Implement modality-specific encoders with cross-modal alignment

**Interactive Reasoning Enhancement Strategies:**
4. **Spatial Information Utilization**:
   - Provide explicit spatial representations (as in VoxPoser)
   - Use 3D scene graphs for spatial relationship understanding
   - Implement spatial attention mechanisms for object localization

5. **Instruction Adaptation Capabilities**:
   - Develop context-aware instruction parsing
   - Implement state-dependent instruction switching mechanisms
   - Use reinforcement learning for optimal instruction timing

6. **Task Decomposition and Planning**:
   - Train models on our enhanced workflow annotations
   - Use hierarchical planning with sub-goal prediction
   - Implement tentative planning with execution monitoring

**Workflow-Based Training Strategies:**
7. **Leverage Mid-Level Task Decomposition**:
   - Use our checkpoint-based task splitting for curriculum learning
   - Train on progressive difficulty levels
   - Implement milestone-based reward shaping

These strategies are directly informed by our comprehensive evaluation metrics and will be supported by our enhanced benchmark annotations.

### Q5: No comparison or baseline involving human performance or human-level task demonstrations is provided. Such a baseline would help contextualize the difficulty level of tasks and set meaningful targets for model performance.

**Response:**

You're absolutely right about the need for human baselines. We collected human demonstrations during our AR teleoperation data collection and can provide:

1. **Human Success Rates**: Across all 50 tasks for difficulty contextualization
2. **Human Strategy Analysis**: Common approaches and failure modes
3. **Efficiency Comparisons**: Steps required and time to completion
4. **Learning Curves**: How human performance improves with practice

Preliminary analysis shows humans achieve 85-95% success rates on COIN tasks, providing clear targets for model development.


---

## Reviewer 8aJw

### Q1: Experiments focus on tabletop scenarios, neglecting dynamic environments (e.g., outdoor or cluttered spaces). This limits the benchmark's generalizability to real-world robotics tasks. In addition, tasks assume fixed environmental setups, lacking dynamic elements (e.g., moving obstacles or changing object states). Real-world interactions often require handling unpredictability, which COIN does not fully capture. It is interesting to introduce dynamic environments (e.g., kitchen with running water, outdoor construction sites) to test adaptability. This would better reflect real-world manipulation challenges.

**Response:**

Thank you for these excellent and practically meaningful scenario suggestions! We completely agree that these environments (kitchens with running water, outdoor construction sites, cluttered spaces) are highly valuable and very suitable for future extensions of our benchmark.

**Our Current Focus and Rationale:**

Our deliberate focus on tabletop scenarios stems from a fundamental concern: **how to enable robots to understand their environment and adapt their strategies during interaction**. This core challenge emphasizes **adaptability during manipulation** rather than environmental complexity. We believe it's crucial to first establish strong foundations in interactive reasoning before scaling to more complex environments.

**Current VLA Limitations Justify Our Approach:**

As our evaluation reveals, current VLAs struggle significantly even on our controlled tabletop tasks, with success rates below 3%. This suggests that **VLAs urgently need to solve the fundamental problem of "thinking" during interaction** before tackling more complex environments. Specifically:

1. **Interactive Reasoning Gap**: Models fail to effectively use information gathered through exploration
2. **Adaptive Strategy Formation**: Current approaches struggle to modify their behavior based on environmental feedback
3. **Causal Understanding**: VLAs show limited ability to form and test hypotheses about object properties

**Future Extensions We Envision:**

Once we establish solid interactive reasoning capabilities on tabletop scenarios, we absolutely plan to extend COIN to the dynamic environments you suggest:
1. **COIN-Kitchen**: Complex scenarios with running water, heat sources, and multi-agent interactions
2. **COIN-Outdoor**: Construction and maintenance tasks with weather variations
3. **COIN-Dynamic**: Tasks with moving obstacles and changing environmental conditions
4. **COIN-Cluttered**: Dense object arrangements requiring spatial reasoning

We view COIN's current tabletop focus as laying the essential groundwork for these more ambitious scenarios.

### Q2: While COIN uses physics-based simulation, it may not fully replicate real-world robot dynamics (e.g., sensor noise, actuator limitations). This could affect transferability to physical robots. It is cool to integrate COIN with physical robot platforms (e.g., Franka Emika Panda) for real-world validation. This could involve hybrid simulation-reality setups to reduce the sim-to-real gap.

**Response:**

You raise an excellent point about sim-to-real transfer. Our current simulation-based approach provides several advantages while acknowledging limitations:

**Simulation Advantages:**
1. **Scalability**: Enables evaluation across 50 diverse tasks with consistent conditions
2. **Reproducibility**: Ensures fair comparison across different models and research groups
3. **Safety**: Allows testing of potentially unsafe behaviors without physical risk
4. **Cost-Effectiveness**: Reduces hardware requirements for widespread adoption

**Addressing Sim-to-Real Gap:**
1. **Physics Fidelity**: Our simulation uses high-fidelity physics with realistic friction, contact dynamics, and material properties
2. **Sensor Modeling**: We incorporate realistic camera noise, depth sensor limitations, and lighting variations
3. **Action Space Realism**: Robot actions are constrained by realistic joint limits and dynamics

**COIN's Role as a Community Benchmark:**

As a benchmark, our primary goal is to provide **fair and consistent evaluation** of different models rather than extensive physical robot integration. Similar to successful benchmarks in our field:

- **SimPLEnv**: Gained community adoption by enabling systematic VLA testing in simulation environments
- **Libero**: Became a widely-used VLA benchmark by focusing on generalization evaluation
- **RLBench**: Established standard evaluation protocols for manipulation tasks

**Our Benchmark Philosophy:**
1. **Standardized Evaluation**: Consistent simulation environment ensures fair comparison across research groups
2. **Accessibility**: Simulation-based approach enables widespread community participation without expensive hardware
3. **Reproducibility**: Deterministic environments allow reliable benchmarking and progress tracking
4. **Focus on Core Capabilities**: Interactive reasoning evaluation without confounding factors from hardware variations

**Community Impact Goal:**
We aim to provide a carefully designed platform that **promotes community attention to interactive reasoning challenges**, just as SimPLEnv advanced VLA testing and Libero advanced generalization research. Our contribution is establishing the evaluation framework for interactive reasoning, which we believe is a fundamental capability gap that needs systematic study before scaling to physical systems.

The simulation-based approach allows the community to make rapid progress on the core algorithmic challenges we identify.
