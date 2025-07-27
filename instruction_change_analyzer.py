#!/usr/bin/env python3
"""
Instruction Change Analyzer for COIN Benchmark

This script analyzes instruction changes in trajectory data, extracts corresponding video segments,
and provides a framework for human annotation and scoring.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from data_loader import RebuttalDataLoader

@dataclass
class InstructionChange:
    """Container for instruction change information."""
    step_number: int
    previous_instruction: Dict[str, Any]
    new_instruction: Dict[str, Any]
    change_type: str  # 'subtask_change', 'plan_update', 'initial'
    description: str
    video_start_frame: int
    video_end_frame: int
    video_segment_path: Optional[str] = None

@dataclass
class HumanAnnotation:
    """Container for human annotation data."""
    trajectory_id: str
    instruction_change_id: int
    
    # Quality scores (1-5 scale)
    instruction_clarity: int  # How clear is the new instruction?
    transition_smoothness: int  # How smooth is the transition between instructions?
    execution_quality: int  # How well does the robot execute the new instruction?
    overall_score: int  # Overall quality of instruction change handling
    
    # Additional annotations
    notes: str
    problematic_behaviors: List[str]  # e.g., ['hesitation', 'wrong_action', 'confusion']
    timestamp: str

class InstructionChangeAnalyzer:
    """Analyze instruction changes in trajectory data and prepare for human annotation."""
    
    def __init__(self, output_dir: str = "instruction_analysis"):
        """
        Initialize the analyzer.
        
        Args:
            output_dir: Directory to save analysis results and video segments
        """
        self.output_dir = output_dir
        self.ensure_output_directory()
        
    def ensure_output_directory(self):
        """Create output directory structure."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "video_segments"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "annotations"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "analysis_reports"), exist_ok=True)
    
    def analyze_trajectory_instructions(self, trajectory_data) -> List[InstructionChange]:
        """
        Analyze instruction changes in a single trajectory.
        
        Args:
            trajectory_data: Trajectory data from RebuttalDataLoader
            
        Returns:
            List of InstructionChange objects
        """
        if not trajectory_data.chat_data or 'data' not in trajectory_data.chat_data:
            return []
        
        chat_steps = trajectory_data.chat_data['data']
        instruction_changes = []
        
        previous_instruction = None
        
        for i, step in enumerate(chat_steps):
            step_number = step.get('step', i * 50)  # Default step increment
            current_instruction = step.get('input_instruction', {})
            
            if i == 0:
                # Initial instruction
                change = InstructionChange(
                    step_number=step_number,
                    previous_instruction={},
                    new_instruction=current_instruction,
                    change_type='initial',
                    description='Initial instruction provided',
                    video_start_frame=max(0, step_number - 25),
                    video_end_frame=min(len(trajectory_data.actions), step_number + 25)
                )
                instruction_changes.append(change)
            else:
                # Check for instruction changes
                change_type, description = self._analyze_instruction_change(
                    previous_instruction, current_instruction
                )
                
                if change_type != 'no_change':
                    change = InstructionChange(
                        step_number=step_number,
                        previous_instruction=previous_instruction,
                        new_instruction=current_instruction,
                        change_type=change_type,
                        description=description,
                        video_start_frame=max(0, step_number - 25),
                        video_end_frame=min(len(trajectory_data.actions), step_number + 25)
                    )
                    instruction_changes.append(change)
            
            previous_instruction = current_instruction
        
        return instruction_changes
    
    def _analyze_instruction_change(self, prev_inst: Dict, curr_inst: Dict) -> Tuple[str, str]:
        """
        Analyze the type of instruction change.
        
        Args:
            prev_inst: Previous instruction dictionary
            curr_inst: Current instruction dictionary
            
        Returns:
            Tuple of (change_type, description)
        """
        if not isinstance(prev_inst, dict) or not isinstance(curr_inst, dict):
            return 'format_change', 'Instruction format changed'
        
        # Check for subtask changes
        prev_subtask = prev_inst.get('Current subtask', '')
        curr_subtask = curr_inst.get('Current subtask', '')
        
        if prev_subtask != curr_subtask and curr_subtask:
            return 'subtask_change', f'Subtask changed: "{prev_subtask}" → "{curr_subtask}"'
        
        # Check for plan updates
        prev_plan = prev_inst.get('Current plan (remaining subtasks)', [])
        curr_plan = curr_inst.get('Current plan (remaining subtasks)', [])
        
        if prev_plan != curr_plan:
            return 'plan_update', f'Plan updated: {len(prev_plan)} → {len(curr_plan)} remaining subtasks'
        
        # Check for high-level instruction changes
        prev_high_level = prev_inst.get('High-level instruction', '')
        curr_high_level = curr_inst.get('High-level instruction', '')
        
        if prev_high_level != curr_high_level:
            return 'high_level_change', 'High-level instruction changed'
        
        return 'no_change', 'No significant instruction change'
    
    def extract_video_segment(self, video_path: str, start_frame: int, end_frame: int, 
                            output_path: str, fps: int = 30) -> bool:
        """
        Extract a video segment from the full trajectory video.
        
        Args:
            video_path: Path to the full trajectory video
            start_frame: Starting frame number
            end_frame: Ending frame number
            output_path: Path for the output video segment
            fps: Frames per second of the video
            
        Returns:
            True if extraction successful, False otherwise
        """
        try:
            import cv2
            
            # Open the video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Cannot open video {video_path}")
                return False
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Adjust frame range
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Extract frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add frame number overlay
                cv2.putText(frame, f'Frame: {frame_idx}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                out.write(frame)
            
            # Clean up
            cap.release()
            out.release()
            
            print(f"Video segment extracted: {output_path}")
            return True
            
        except ImportError:
            print("OpenCV not available. Install with: pip install opencv-python")
            return False
        except Exception as e:
            print(f"Error extracting video segment: {e}")
            return False
    
    def create_annotation_template(self, trajectory_id: str, instruction_changes: List[InstructionChange]) -> str:
        """
        Create an annotation template for human scoring.
        
        Args:
            trajectory_id: ID of the trajectory
            instruction_changes: List of instruction changes
            
        Returns:
            Path to the created annotation template
        """
        template_path = os.path.join(self.output_dir, "annotations", f"{trajectory_id}_annotation_template.json")
        
        template = {
            "trajectory_id": trajectory_id,
            "instruction_changes": [],
            "annotation_instructions": {
                "instruction_clarity": "Rate how clear and understandable the new instruction is (1=very unclear, 5=very clear)",
                "transition_smoothness": "Rate how smoothly the robot transitions between instructions (1=very abrupt, 5=very smooth)",
                "execution_quality": "Rate how well the robot executes the new instruction (1=very poor, 5=excellent)",
                "overall_score": "Overall quality of instruction change handling (1=very poor, 5=excellent)",
                "problematic_behaviors": "Select from: ['hesitation', 'wrong_action', 'confusion', 'repetition', 'freezing', 'other']"
            },
            "annotations": []
        }
        
        for i, change in enumerate(instruction_changes):
            change_info = {
                "change_id": i,
                "step_number": change.step_number,
                "change_type": change.change_type,
                "description": change.description,
                "previous_instruction": change.previous_instruction,
                "new_instruction": change.new_instruction,
                "video_segment": change.video_segment_path,
                "frames": f"{change.video_start_frame}-{change.video_end_frame}"
            }
            template["instruction_changes"].append(change_info)
            
            # Add empty annotation template
            annotation_template = {
                "change_id": i,
                "instruction_clarity": None,
                "transition_smoothness": None,
                "execution_quality": None,
                "overall_score": None,
                "notes": "",
                "problematic_behaviors": [],
                "annotator": "",
                "timestamp": ""
            }
            template["annotations"].append(annotation_template)
        
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        print(f"Annotation template created: {template_path}")
        return template_path
    
    def analyze_all_trajectories(self) -> Dict[str, List[InstructionChange]]:
        """
        Analyze instruction changes for all trajectories.
        
        Returns:
            Dictionary mapping trajectory IDs to their instruction changes
        """
        loader = RebuttalDataLoader()
        all_changes = {}
        
        print("Analyzing instruction changes in all trajectories...")
        
        for traj_id, traj_data in loader.trajectories.items():
            print(f"\nAnalyzing trajectory: {traj_id}")
            print(f"Task: {traj_data.task_name}")
            
            # Analyze instruction changes
            changes = self.analyze_trajectory_instructions(traj_data)
            all_changes[traj_id] = changes
            
            print(f"Found {len(changes)} instruction changes")
            
            # Extract video segments if video exists
            if traj_data.video_path and os.path.exists(traj_data.video_path):
                for i, change in enumerate(changes):
                    segment_filename = f"{traj_id}_change_{i}_{change.step_number}.mp4"
                    segment_path = os.path.join(self.output_dir, "video_segments", segment_filename)
                    
                    success = self.extract_video_segment(
                        traj_data.video_path,
                        change.video_start_frame,
                        change.video_end_frame,
                        segment_path
                    )
                    
                    if success:
                        change.video_segment_path = segment_path
            else:
                print(f"Video not found: {traj_data.video_path}")
            
            # Create annotation template
            self.create_annotation_template(traj_id, changes)
            
            # Print change details
            for i, change in enumerate(changes):
                print(f"  Change {i+1}: {change.description} (Step {change.step_number})")
        
        # Create summary report
        self.create_summary_report(all_changes)
        
        return all_changes
    
    def create_summary_report(self, all_changes: Dict[str, List[InstructionChange]]):
        """Create a summary report of all instruction changes."""
        report_path = os.path.join(self.output_dir, "analysis_reports", "instruction_changes_summary.json")
        
        summary = {
            "total_trajectories": len(all_changes),
            "total_instruction_changes": sum(len(changes) for changes in all_changes.values()),
            "change_types": {},
            "trajectories": {}
        }
        
        # Analyze change types
        for traj_id, changes in all_changes.items():
            summary["trajectories"][traj_id] = {
                "total_changes": len(changes),
                "change_types": {}
            }
            
            for change in changes:
                # Global change type count
                if change.change_type not in summary["change_types"]:
                    summary["change_types"][change.change_type] = 0
                summary["change_types"][change.change_type] += 1
                
                # Per-trajectory change type count
                if change.change_type not in summary["trajectories"][traj_id]["change_types"]:
                    summary["trajectories"][traj_id]["change_types"][change.change_type] = 0
                summary["trajectories"][traj_id]["change_types"][change.change_type] += 1
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nSummary report created: {report_path}")
        print(f"Total trajectories: {summary['total_trajectories']}")
        print(f"Total instruction changes: {summary['total_instruction_changes']}")
        print("Change types:")
        for change_type, count in summary["change_types"].items():
            print(f"  {change_type}: {count}")

def create_annotation_interface():
    """Create a simple web-based annotation interface."""
    interface_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COIN Instruction Change Annotation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .change-item { border: 1px solid #ccc; margin: 20px 0; padding: 20px; border-radius: 5px; }
        .instruction-box { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 3px; }
        .rating { margin: 10px 0; }
        .rating input { margin: 0 5px; }
        video { max-width: 100%; height: auto; }
        .submit-btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .submit-btn:hover { background: #45a049; }
    </style>
</head>
<body>
    <h1>COIN Benchmark: Instruction Change Annotation</h1>
    <p>Please rate each instruction change on a scale of 1-5 for the following criteria:</p>
    
    <div id="annotation-form">
        <!-- Annotation items will be loaded here -->
    </div>
    
    <button class="submit-btn" onclick="saveAnnotations()">Save Annotations</button>
    
    <script>
        function loadAnnotationData(templatePath) {
            // Load annotation template and create form
            // This would be implemented with actual file loading
        }
        
        function saveAnnotations() {
            // Collect all annotation data and save
            alert('Annotations saved! (Implementation needed)');
        }
    </script>
</body>
</html>
"""
    
    interface_path = os.path.join("instruction_analysis", "annotation_interface.html")
    with open(interface_path, 'w', encoding='utf-8') as f:
        f.write(interface_html)
    
    print(f"Annotation interface created: {interface_path}")
    return interface_path

if __name__ == "__main__":
    print("=== COIN Instruction Change Analyzer ===")
    
    # Initialize analyzer
    analyzer = InstructionChangeAnalyzer()
    
    # Analyze all trajectories
    all_changes = analyzer.analyze_all_trajectories()
    
    # Create annotation interface
    create_annotation_interface()
    
    print("\n=== Analysis Complete ===")
    print("Next steps:")
    print("1. Review the generated annotation templates in 'instruction_analysis/annotations/'")
    print("2. Watch the extracted video segments in 'instruction_analysis/video_segments/'")
    print("3. Fill out the annotation templates with human scores")
    print("4. Use the annotation interface for easier scoring")
    
    print(f"\nTotal instruction changes found: {sum(len(changes) for changes in all_changes.values())}")
