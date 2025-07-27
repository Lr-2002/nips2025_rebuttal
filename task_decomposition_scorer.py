#!/usr/bin/env python3
"""
Task Decomposition Scorer for COIN Benchmark

This module evaluates the quality of task decomposition in hierarchical VLA systems,
specifically analyzing how well high-level instructions are broken down into subtasks.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from data_loader import RebuttalDataLoader

@dataclass
class TaskDecompositionMetrics:
    """Container for task decomposition analysis results."""
    overall_decomposition_score: float
    
    # Individual components
    logical_coherence_score: float
    granularity_score: float
    completeness_score: float
    primitive_alignment_score: float
    
    # Detailed statistics
    total_subtasks: int
    avg_subtask_length: float
    primitive_task_coverage: float
    logical_flow_violations: int
    
    # Per-step analysis
    step_scores: List[float]
    problematic_steps: List[int]

class TaskDecompositionScorer:
    """
    Evaluate task decomposition quality in hierarchical VLA systems.
    
    This scorer analyzes how well high-level instructions are decomposed into
    coherent, executable subtasks, supporting the Interactive Reasoning Analysis
    metrics mentioned in our rebuttal.
    """
    
    def __init__(self):
        """Initialize the task decomposition scorer."""
        # Define primitive task patterns from the system prompt
        self.primitive_tasks = {
            "close the drawer", "close the door", "close the cabinet door", "close the microwave",
            "open the drawer", "open the cabinet door", "open the microwave", "open the door",
            "pull the pivot to the target area", "pick up the pen and put it to the marker",
            "pick up the bottle and put it on the marker", "pick the apple to the marker",
            "pick up the cube, put it in the holder", "rotate the holder till the hole upward",
            "rotate the cube till the blue face upward", "rotate the USB body for 90 degree with plug right ward",
            "put the ball into the container", "put the fork on the plate", "turn on the trigger",
            "stack all the cube", "find and pick the book from the bookshelf and put it on the marker"
        }
        
        # Action categories for logical flow analysis
        self.action_categories = {
            'open': ['open the drawer', 'open the cabinet door', 'open the microwave', 'open the door'],
            'close': ['close the drawer', 'close the door', 'close the cabinet door', 'close the microwave'],
            'pick': ['pick up the pen and put it to the marker', 'pick up the bottle and put it on the marker', 
                    'pick the apple to the marker', 'pick up the cube, put it in the holder'],
            'put': ['put the ball into the container', 'put the fork on the plate'],
            'rotate': ['rotate the holder till the hole upward', 'rotate the cube till the blue face upward',
                      'rotate the USB body for 90 degree with plug right ward'],
            'other': ['pull the pivot to the target area', 'turn on the trigger', 'stack all the cube',
                     'find and pick the book from the bookshelf and put it on the marker']
        }
    
    def calculate_decomposition_score(self, trajectory_data) -> TaskDecompositionMetrics:
        """
        Calculate comprehensive task decomposition score for a trajectory.
        
        Args:
            trajectory_data: Trajectory data from RebuttalDataLoader
            
        Returns:
            TaskDecompositionMetrics with detailed analysis
        """
        if not trajectory_data.chat_data or 'data' not in trajectory_data.chat_data:
            return self._create_empty_metrics()
        
        chat_steps = trajectory_data.chat_data['data']
        
        # Extract all subtask decompositions
        all_subtasks = []
        step_scores = []
        problematic_steps = []
        
        for i, step in enumerate(chat_steps):
            response = step.get('response', {})
            subtasks = response.get('subtasks', [])
            
            if subtasks:
                all_subtasks.extend(subtasks)
                
                # Calculate score for this decomposition step
                step_score = self._evaluate_single_decomposition(
                    step.get('input_instruction', {}),
                    subtasks,
                    i
                )
                step_scores.append(step_score)
                
                if step_score < 0.6:  # Threshold for problematic decomposition
                    problematic_steps.append(i)
        
        if not all_subtasks:
            return self._create_empty_metrics()
        
        # Calculate individual component scores
        logical_coherence = self._calculate_logical_coherence(all_subtasks)
        granularity_score = self._calculate_granularity_score(all_subtasks)
        completeness_score = self._calculate_completeness_score(chat_steps)
        primitive_alignment = self._calculate_primitive_alignment(all_subtasks)
        
        # Calculate overall score (weighted combination)
        overall_score = (
            0.3 * logical_coherence +
            0.25 * granularity_score +
            0.25 * completeness_score +
            0.2 * primitive_alignment
        )
        
        # Calculate statistics
        total_subtasks = len(all_subtasks)
        avg_subtask_length = np.mean([len(task.split()) for task in all_subtasks])
        primitive_coverage = self._calculate_primitive_coverage(all_subtasks)
        logical_violations = self._count_logical_violations(all_subtasks)
        
        return TaskDecompositionMetrics(
            overall_decomposition_score=overall_score,
            logical_coherence_score=logical_coherence,
            granularity_score=granularity_score,
            completeness_score=completeness_score,
            primitive_alignment_score=primitive_alignment,
            total_subtasks=total_subtasks,
            avg_subtask_length=avg_subtask_length,
            primitive_task_coverage=primitive_coverage,
            logical_flow_violations=logical_violations,
            step_scores=step_scores,
            problematic_steps=problematic_steps
        )
    
    def _evaluate_single_decomposition(self, instruction: Dict, subtasks: List[str], step_idx: int) -> float:
        """Evaluate a single decomposition step."""
        if not subtasks:
            return 0.0
        
        # Check if subtasks are reasonable for the instruction
        high_level = instruction.get('High-level instruction', '')
        
        # Basic quality checks
        score = 1.0
        
        # Penalize very short or very long subtask lists
        if len(subtasks) < 2:
            score *= 0.7  # Too simple
        elif len(subtasks) > 8:
            score *= 0.8  # Too complex
        
        # Check for primitive task alignment
        primitive_matches = sum(1 for task in subtasks if self._matches_primitive_task(task))
        primitive_ratio = primitive_matches / len(subtasks)
        score *= (0.5 + 0.5 * primitive_ratio)  # Reward primitive task usage
        
        # Check for logical ordering (open before pick, pick before put, etc.)
        if self._has_logical_ordering(subtasks):
            score *= 1.1
        else:
            score *= 0.9
        
        return min(1.0, score)
    
    def _calculate_logical_coherence(self, subtasks: List[str]) -> float:
        """
        Calculate logical coherence of subtask sequence.
        
        Evaluates whether subtasks follow a logical order and make sense together.
        """
        if len(subtasks) < 2:
            return 1.0
        
        coherence_score = 1.0
        
        # Check for common logical patterns
        for i in range(len(subtasks) - 1):
            current_task = subtasks[i].lower()
            next_task = subtasks[i + 1].lower()
            
            # Pattern 1: Open before pick/put
            if 'open' in current_task and ('pick' in next_task or 'put' in next_task):
                coherence_score *= 1.1  # Reward good pattern
            
            # Pattern 2: Pick before put
            if 'pick' in current_task and 'put' in next_task:
                coherence_score *= 1.1
            
            # Pattern 3: Close after other actions
            if 'close' in next_task and any(action in current_task for action in ['pick', 'put', 'rotate']):
                coherence_score *= 1.1
            
            # Anti-pattern: Close before pick/put
            if 'close' in current_task and ('pick' in next_task or 'put' in next_task):
                coherence_score *= 0.8  # Penalize illogical pattern
        
        return min(1.0, coherence_score)
    
    def _calculate_granularity_score(self, subtasks: List[str]) -> float:
        """
        Calculate granularity appropriateness of subtasks.
        
        Evaluates whether subtasks are at the right level of detail.
        """
        if not subtasks:
            return 0.0
        
        # Analyze subtask complexity
        word_counts = [len(task.split()) for task in subtasks]
        avg_words = np.mean(word_counts)
        std_words = np.std(word_counts)
        
        # Ideal granularity: 4-8 words per subtask, low variance
        granularity_score = 1.0
        
        # Penalize too simple (< 3 words) or too complex (> 12 words)
        if avg_words < 3:
            granularity_score *= 0.7
        elif avg_words > 12:
            granularity_score *= 0.8
        elif 4 <= avg_words <= 8:
            granularity_score *= 1.1  # Reward good granularity
        
        # Penalize high variance (inconsistent granularity)
        if std_words > 3:
            granularity_score *= 0.9
        
        return min(1.0, granularity_score)
    
    def _calculate_completeness_score(self, chat_steps: List[Dict]) -> float:
        """
        Calculate completeness of task decomposition.
        
        Evaluates whether the decomposition covers all necessary steps.
        """
        if not chat_steps:
            return 0.0
        
        # Check if high-level instruction is fully addressed
        first_step = chat_steps[0]
        high_level_instruction = first_step.get('input_instruction', {}).get('High-level instruction', '')
        
        # Extract key components from high-level instruction
        key_actions = self._extract_key_actions(high_level_instruction)
        
        # Check if subtasks cover these key actions
        all_subtasks = []
        for step in chat_steps:
            response = step.get('response', {})
            subtasks = response.get('subtasks', [])
            all_subtasks.extend(subtasks)
        
        covered_actions = 0
        for action in key_actions:
            if any(action.lower() in task.lower() for task in all_subtasks):
                covered_actions += 1
        
        if not key_actions:
            return 0.8  # Default score if no clear actions identified
        
        completeness = covered_actions / len(key_actions)
        return completeness
    
    def _calculate_primitive_alignment(self, subtasks: List[str]) -> float:
        """
        Calculate alignment with primitive task vocabulary.
        
        Evaluates how well subtasks match the available primitive tasks.
        """
        if not subtasks:
            return 0.0
        
        aligned_count = 0
        for task in subtasks:
            if self._matches_primitive_task(task):
                aligned_count += 1
        
        alignment_ratio = aligned_count / len(subtasks)
        return alignment_ratio
    
    def _matches_primitive_task(self, subtask: str) -> bool:
        """Check if a subtask matches or closely resembles a primitive task."""
        subtask_lower = subtask.lower().strip()
        
        # Exact match
        if subtask_lower in [task.lower() for task in self.primitive_tasks]:
            return True
        
        # Partial match (key words present)
        for primitive in self.primitive_tasks:
            primitive_words = set(primitive.lower().split())
            subtask_words = set(subtask_lower.split())
            
            # If most key words match, consider it aligned
            if len(primitive_words.intersection(subtask_words)) >= len(primitive_words) * 0.7:
                return True
        
        return False
    
    def _has_logical_ordering(self, subtasks: List[str]) -> bool:
        """Check if subtasks follow logical ordering patterns."""
        if len(subtasks) < 2:
            return True
        
        # Look for common logical patterns
        has_open_before_action = False
        has_action_before_close = False
        
        for i, task in enumerate(subtasks):
            task_lower = task.lower()
            
            # Check if open comes before pick/put actions
            if 'open' in task_lower:
                remaining_tasks = subtasks[i+1:]
                if any('pick' in t.lower() or 'put' in t.lower() for t in remaining_tasks):
                    has_open_before_action = True
            
            # Check if close comes after other actions
            if 'close' in task_lower and i > 0:
                previous_tasks = subtasks[:i]
                if any(action in t.lower() for t in previous_tasks for action in ['pick', 'put', 'open']):
                    has_action_before_close = True
        
        return has_open_before_action or has_action_before_close
    
    def _extract_key_actions(self, instruction: str) -> List[str]:
        """Extract key actions from high-level instruction."""
        instruction_lower = instruction.lower()
        key_actions = []
        
        # Common action patterns
        action_patterns = {
            'find': ['find', 'locate', 'search'],
            'pick': ['pick', 'grab', 'take', 'get'],
            'put': ['put', 'place', 'move'],
            'open': ['open'],
            'close': ['close'],
            'rotate': ['rotate', 'turn'],
        }
        
        for action_type, patterns in action_patterns.items():
            if any(pattern in instruction_lower for pattern in patterns):
                key_actions.append(action_type)
        
        return key_actions
    
    def _calculate_primitive_coverage(self, subtasks: List[str]) -> float:
        """Calculate what percentage of subtasks use primitive task vocabulary."""
        if not subtasks:
            return 0.0
        
        primitive_count = sum(1 for task in subtasks if self._matches_primitive_task(task))
        return primitive_count / len(subtasks)
    
    def _count_logical_violations(self, subtasks: List[str]) -> int:
        """Count logical ordering violations in subtask sequence."""
        violations = 0
        
        for i in range(len(subtasks) - 1):
            current = subtasks[i].lower()
            next_task = subtasks[i + 1].lower()
            
            # Violation: Close before pick/put
            if 'close' in current and ('pick' in next_task or 'put' in next_task):
                violations += 1
            
            # Violation: Put before pick (without object specification)
            if 'put' in current and 'pick' in next_task:
                violations += 1
        
        return violations
    
    def _create_empty_metrics(self) -> TaskDecompositionMetrics:
        """Create empty metrics for edge cases."""
        return TaskDecompositionMetrics(
            overall_decomposition_score=0.0,
            logical_coherence_score=0.0,
            granularity_score=0.0,
            completeness_score=0.0,
            primitive_alignment_score=0.0,
            total_subtasks=0,
            avg_subtask_length=0.0,
            primitive_task_coverage=0.0,
            logical_flow_violations=0,
            step_scores=[],
            problematic_steps=[]
        )

def calculate_task_decomposition_score(trajectory_data) -> float:
    """
    Convenience function to calculate task decomposition score.
    
    Args:
        trajectory_data: Trajectory data from RebuttalDataLoader
        
    Returns:
        Overall task decomposition score (0-1, higher is better)
    """
    scorer = TaskDecompositionScorer()
    metrics = scorer.calculate_decomposition_score(trajectory_data)
    return metrics.overall_decomposition_score

# Example usage and testing
if __name__ == "__main__":
    print("=== Task Decomposition Scoring Analysis ===")
    
    # Load trajectory data
    loader = RebuttalDataLoader()
    
    if not loader.trajectories:
        print("No trajectories found!")
        exit()
    
    # Initialize scorer
    scorer = TaskDecompositionScorer()
    
    # Analyze each trajectory
    for traj_id, traj_data in loader.trajectories.items():
        print(f"\n=== Analyzing Trajectory: {traj_id} ===")
        print(f"Task: {traj_data.task_name}")
        
        # Calculate decomposition metrics
        metrics = scorer.calculate_decomposition_score(traj_data)
        
        print(f"\nTask Decomposition Analysis:")
        print(f"  Overall Score: {metrics.overall_decomposition_score:.4f}")
        print(f"  Logical Coherence: {metrics.logical_coherence_score:.4f}")
        print(f"  Granularity: {metrics.granularity_score:.4f}")
        print(f"  Completeness: {metrics.completeness_score:.4f}")
        print(f"  Primitive Alignment: {metrics.primitive_alignment_score:.4f}")
        
        print(f"\nStatistics:")
        print(f"  Total Subtasks: {metrics.total_subtasks}")
        print(f"  Avg Subtask Length: {metrics.avg_subtask_length:.1f} words")
        print(f"  Primitive Coverage: {metrics.primitive_task_coverage:.2%}")
        print(f"  Logical Violations: {metrics.logical_flow_violations}")
        
        if metrics.problematic_steps:
            print(f"  Problematic Steps: {metrics.problematic_steps}")
        
        # Show example subtasks
        if traj_data.chat_data and 'data' in traj_data.chat_data:
            first_step = traj_data.chat_data['data'][0]
            response = first_step.get('response', {})
            subtasks = response.get('subtasks', [])
            
            if subtasks:
                print(f"\nExample Subtask Decomposition:")
                for i, subtask in enumerate(subtasks, 1):
                    primitive_match = "✓" if scorer._matches_primitive_task(subtask) else "✗"
                    print(f"  {i}. {subtask} {primitive_match}")
        
        # Detection thresholds
        if metrics.overall_decomposition_score < 0.6:
            print(f"\n⚠️  POOR TASK DECOMPOSITION detected!")
            print("   Issues may include: illogical ordering, poor granularity, or incomplete coverage")
        elif metrics.overall_decomposition_score > 0.8:
            print(f"\n✅ GOOD TASK DECOMPOSITION: Well-structured and logical subtask breakdown")
    
    print(f"\n=== Task Decomposition Scoring Complete ===")
    print("This metric supports the Interactive Reasoning Analysis framework from our rebuttal.")
    print("It evaluates how well hierarchical VLA systems decompose complex tasks into executable subtasks.")
