#!/usr/bin/env python3
"""
Human Annotation Scorer for COIN Instruction Change Analysis

This script provides tools for human annotators to score instruction changes
and analyze the collected annotations.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class AnnotationStats:
    """Statistics for annotation analysis."""
    total_annotations: int
    avg_instruction_clarity: float
    avg_transition_smoothness: float
    avg_execution_quality: float
    avg_overall_score: float
    common_problems: List[str]
    score_distribution: Dict[str, Dict[int, int]]

class AnnotationScorer:
    """Handle human annotation scoring and analysis."""
    
    def __init__(self, annotation_dir: str = "instruction_analysis/annotations"):
        """
        Initialize the annotation scorer.
        
        Args:
            annotation_dir: Directory containing annotation files
        """
        self.annotation_dir = annotation_dir
    
    def load_annotation_template(self, trajectory_id: str) -> Optional[Dict]:
        """Load annotation template for a trajectory."""
        template_path = os.path.join(self.annotation_dir, f"{trajectory_id}_annotation_template.json")
        
        if not os.path.exists(template_path):
            print(f"Template not found: {template_path}")
            return None
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def interactive_annotation(self, trajectory_id: str):
        """
        Interactive command-line annotation interface.
        
        Args:
            trajectory_id: ID of the trajectory to annotate
        """
        template = self.load_annotation_template(trajectory_id)
        if not template:
            return
        
        print(f"\\n=== Annotating Trajectory: {trajectory_id} ===")
        print(f"Task: {template.get('task_name', 'Unknown')}")
        print(f"Total instruction changes: {len(template['instruction_changes'])}\\n")
        
        # Display annotation instructions
        print("Annotation Guidelines:")
        for key, instruction in template['annotation_instructions'].items():
            print(f"  {key}: {instruction}")
        print()\n        
        annotator_name = input("Enter your name: ").strip()
        
        # Annotate each instruction change
        for i, change_info in enumerate(template['instruction_changes']):\n            print(f"\\n{'='*60}")
            print(f"Instruction Change {i+1}/{len(template['instruction_changes'])}")
            print(f"{'='*60}")
            print(f"Step: {change_info['step_number']}")
            print(f"Type: {change_info['change_type']}")
            print(f"Description: {change_info['description']}")
            print(f"Video segment: {change_info.get('video_segment', 'N/A')}")
            print(f"Frames: {change_info['frames']}")
            
            print("\\nPrevious instruction:")
            self._print_instruction(change_info['previous_instruction'])
            
            print("\\nNew instruction:")
            self._print_instruction(change_info['new_instruction'])
            
            # Get video confirmation
            video_path = change_info.get('video_segment')
            if video_path and os.path.exists(video_path):
                watch = input(f"\\nWatch video segment? ({video_path}) [y/N]: ").strip().lower()
                if watch == 'y':
                    self._open_video(video_path)
            
            # Collect annotations
            annotation = template['annotations'][i]
            
            print("\\nPlease provide ratings (1-5 scale):")
            annotation['instruction_clarity'] = self._get_rating("Instruction clarity", 1, 5)
            annotation['transition_smoothness'] = self._get_rating("Transition smoothness", 1, 5)
            annotation['execution_quality'] = self._get_rating("Execution quality", 1, 5)
            annotation['overall_score'] = self._get_rating("Overall score", 1, 5)
            
            # Get notes
            annotation['notes'] = input("\\nAdditional notes (optional): ").strip()
            
            # Get problematic behaviors
            print("\\nProblematic behaviors (enter numbers, space-separated):")
            behaviors = ['hesitation', 'wrong_action', 'confusion', 'repetition', 'freezing', 'other']
            for j, behavior in enumerate(behaviors):
                print(f"  {j+1}. {behavior}")
            
            behavior_input = input("Select behaviors (e.g., '1 3 5'): ").strip()
            selected_behaviors = []
            if behavior_input:
                try:
                    indices = [int(x) - 1 for x in behavior_input.split()]
                    selected_behaviors = [behaviors[i] for i in indices if 0 <= i < len(behaviors)]
                except ValueError:
                    print("Invalid input, skipping problematic behaviors")
            
            annotation['problematic_behaviors'] = selected_behaviors
            annotation['annotator'] = annotator_name
            annotation['timestamp'] = datetime.now().isoformat()
            
            # Show summary
            print(f"\\nAnnotation summary:")
            print(f"  Clarity: {annotation['instruction_clarity']}/5")
            print(f"  Smoothness: {annotation['transition_smoothness']}/5")
            print(f"  Execution: {annotation['execution_quality']}/5")
            print(f"  Overall: {annotation['overall_score']}/5")
            print(f"  Problems: {', '.join(selected_behaviors) if selected_behaviors else 'None'}")
        
        # Save annotations
        output_path = os.path.join(self.annotation_dir, f"{trajectory_id}_annotations.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        print(f"\\n✅ Annotations saved to: {output_path}")
    
    def _print_instruction(self, instruction: Dict):
        """Pretty print an instruction dictionary."""
        if not instruction:
            print("  (No instruction)")
            return
        
        for key, value in instruction.items():
            if isinstance(value, list):
                print(f"  {key}: {', '.join(value)}")
            else:
                print(f"  {key}: {value}")
    
    def _get_rating(self, prompt: str, min_val: int, max_val: int) -> int:
        """Get a rating from user input with validation."""
        while True:
            try:
                rating = int(input(f"{prompt} ({min_val}-{max_val}): ").strip())
                if min_val <= rating <= max_val:
                    return rating
                else:
                    print(f"Please enter a number between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")
    
    def _open_video(self, video_path: str):
        """Attempt to open video file with default system player."""
        try:
            import subprocess
            import sys
            
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", video_path])
            elif sys.platform == "linux":
                subprocess.run(["xdg-open", video_path])
            elif sys.platform == "win32":
                subprocess.run(["start", video_path], shell=True)
            else:
                print(f"Please manually open: {video_path}")
        except Exception as e:
            print(f"Could not open video automatically: {e}")
            print(f"Please manually open: {video_path}")
    
    def analyze_annotations(self, trajectory_id: str) -> Optional[AnnotationStats]:
        """
        Analyze completed annotations for a trajectory.
        
        Args:
            trajectory_id: ID of the trajectory
            
        Returns:
            AnnotationStats object with analysis results
        """
        annotation_path = os.path.join(self.annotation_dir, f"{trajectory_id}_annotations.json")
        
        if not os.path.exists(annotation_path):
            print(f"Annotations not found: {annotation_path}")
            return None
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = data['annotations']
        completed_annotations = [a for a in annotations if a.get('overall_score') is not None]
        
        if not completed_annotations:
            print("No completed annotations found")
            return None
        
        # Calculate statistics
        clarity_scores = [a['instruction_clarity'] for a in completed_annotations]
        smoothness_scores = [a['transition_smoothness'] for a in completed_annotations]
        execution_scores = [a['execution_quality'] for a in completed_annotations]
        overall_scores = [a['overall_score'] for a in completed_annotations]
        
        # Collect problematic behaviors
        all_problems = []
        for a in completed_annotations:
            all_problems.extend(a.get('problematic_behaviors', []))
        
        problem_counts = {}
        for problem in all_problems:
            problem_counts[problem] = problem_counts.get(problem, 0) + 1
        
        common_problems = sorted(problem_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Score distributions
        score_distribution = {
            'instruction_clarity': {i: clarity_scores.count(i) for i in range(1, 6)},
            'transition_smoothness': {i: smoothness_scores.count(i) for i in range(1, 6)},
            'execution_quality': {i: execution_scores.count(i) for i in range(1, 6)},
            'overall_score': {i: overall_scores.count(i) for i in range(1, 6)}
        }
        
        stats = AnnotationStats(
            total_annotations=len(completed_annotations),
            avg_instruction_clarity=sum(clarity_scores) / len(clarity_scores),
            avg_transition_smoothness=sum(smoothness_scores) / len(smoothness_scores),
            avg_execution_quality=sum(execution_scores) / len(execution_scores),
            avg_overall_score=sum(overall_scores) / len(overall_scores),
            common_problems=[p[0] for p in common_problems[:5]],  # Top 5 problems
            score_distribution=score_distribution
        )
        
        return stats
    
    def generate_analysis_report(self, trajectory_id: str):
        """Generate a comprehensive analysis report."""
        stats = self.analyze_annotations(trajectory_id)
        if not stats:
            return
        
        print(f"\\n=== Annotation Analysis Report: {trajectory_id} ===")
        print(f"Total annotations: {stats.total_annotations}")
        print(f"\\nAverage scores:")
        print(f"  Instruction clarity: {stats.avg_instruction_clarity:.2f}/5")
        print(f"  Transition smoothness: {stats.avg_transition_smoothness:.2f}/5")
        print(f"  Execution quality: {stats.avg_execution_quality:.2f}/5")
        print(f"  Overall score: {stats.avg_overall_score:.2f}/5")
        
        print(f"\\nMost common problems:")
        for problem in stats.common_problems:
            print(f"  - {problem}")
        
        print(f"\\nScore distributions:")
        for metric, distribution in stats.score_distribution.items():
            print(f"  {metric}:")
            for score, count in distribution.items():
                if count > 0:
                    print(f"    {score}: {count} annotations")
        
        # Save report
        report_path = os.path.join("instruction_analysis/analysis_reports", f"{trajectory_id}_annotation_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        report_data = {
            "trajectory_id": trajectory_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_annotations": stats.total_annotations,
                "average_scores": {
                    "instruction_clarity": stats.avg_instruction_clarity,
                    "transition_smoothness": stats.avg_transition_smoothness,
                    "execution_quality": stats.avg_execution_quality,
                    "overall_score": stats.avg_overall_score
                },
                "common_problems": stats.common_problems,
                "score_distributions": stats.score_distribution
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\\n📊 Report saved to: {report_path}")

def main():
    """Main function for command-line usage."""
    import sys
    
    scorer = AnnotationScorer()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python annotation_scorer.py annotate <trajectory_id>")
        print("  python annotation_scorer.py analyze <trajectory_id>")
        print("\\nAvailable trajectories:")
        
        # List available templates
        if os.path.exists(scorer.annotation_dir):
            for filename in os.listdir(scorer.annotation_dir):
                if filename.endswith("_annotation_template.json"):
                    traj_id = filename.replace("_annotation_template.json", "")
                    print(f"  - {traj_id}")
        return
    
    command = sys.argv[1]
    trajectory_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    if command == "annotate" and trajectory_id:
        scorer.interactive_annotation(trajectory_id)
    elif command == "analyze" and trajectory_id:
        scorer.generate_analysis_report(trajectory_id)
    else:
        print("Invalid command or missing trajectory ID")

if __name__ == "__main__":
    main()
