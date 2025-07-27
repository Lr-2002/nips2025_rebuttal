#!/usr/bin/env python3
"""
Visualization tool for instruction changes in COIN benchmark trajectories.
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Any
import numpy as np

class InstructionChangeVisualizer:
    """Visualize instruction changes and annotation results."""
    
    def __init__(self, analysis_dir: str = "instruction_analysis"):
        """Initialize the visualizer."""
        self.analysis_dir = analysis_dir
        self.output_dir = os.path.join(analysis_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_instruction_timeline(self, trajectory_id: str):
        """Create a timeline visualization of instruction changes."""
        template_path = os.path.join(self.analysis_dir, "annotations", f"{trajectory_id}_annotation_template.json")
        
        if not os.path.exists(template_path):
            print(f"Template not found: {template_path}")
            return
        
        with open(template_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        changes = data['instruction_changes']
        
        # Create timeline plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot timeline
        steps = [change['step_number'] for change in changes]
        y_pos = [1] * len(steps)
        
        # Color mapping for change types
        color_map = {
            'initial': 'green',
            'subtask_change': 'blue',
            'plan_update': 'orange',
            'high_level_change': 'red'
        }
        
        colors = [color_map.get(change['change_type'], 'gray') for change in changes]
        
        # Plot points
        scatter = ax.scatter(steps, y_pos, c=colors, s=200, alpha=0.7, zorder=3)
        
        # Add timeline line
        if len(steps) > 1:
            ax.plot([min(steps), max(steps)], [1, 1], 'k-', alpha=0.3, linewidth=2, zorder=1)
        
        # Add annotations
        for i, change in enumerate(changes):
            # Add change description
            ax.annotate(f"Step {change['step_number']}\\n{change['change_type']}", 
                       (change['step_number'], 1), 
                       xytext=(0, 30), textcoords='offset points',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3),
                       fontsize=9)
            
            # Add instruction text
            new_inst = change['new_instruction']
            if isinstance(new_inst, dict):
                subtask = new_inst.get('Current subtask', '')
                if subtask:
                    ax.annotate(f'"{subtask}"', 
                               (change['step_number'], 1), 
                               xytext=(0, -40), textcoords='offset points',
                               ha='center', va='top',
                               fontsize=8, style='italic',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.5))
        
        # Customize plot
        ax.set_xlabel('Step Number', fontsize=12)
        ax.set_title(f'Instruction Changes Timeline: {trajectory_id}', fontsize=14, fontweight='bold')
        ax.set_ylim(0.5, 1.5)
        ax.set_yticks([])
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=10, label=change_type.replace('_', ' ').title())
                          for change_type, color in color_map.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{trajectory_id}_timeline.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Timeline visualization saved: {output_path}")
    
    def plot_annotation_scores(self, trajectory_id: str):
        """Plot annotation scores if available."""
        annotation_path = os.path.join(self.analysis_dir, "annotations", f"{trajectory_id}_annotations.json")
        
        if not os.path.exists(annotation_path):
            print(f"Annotations not found: {annotation_path}")
            return
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = data['annotations']
        completed = [a for a in annotations if a.get('overall_score') is not None]
        
        if not completed:
            print("No completed annotations found")
            return
        
        # Extract scores
        change_ids = [a['change_id'] for a in completed]
        clarity_scores = [a['instruction_clarity'] for a in completed]
        smoothness_scores = [a['transition_smoothness'] for a in completed]
        execution_scores = [a['execution_quality'] for a in completed]
        overall_scores = [a['overall_score'] for a in completed]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Annotation Scores: {trajectory_id}', fontsize=16, fontweight='bold')
        
        # Plot individual metrics
        x = np.arange(len(change_ids))
        width = 0.8
        
        ax1.bar(x, clarity_scores, width, color='skyblue', alpha=0.7)
        ax1.set_title('Instruction Clarity')
        ax1.set_ylabel('Score (1-5)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Change {i+1}' for i in change_ids])
        ax1.set_ylim(0, 5)
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(x, smoothness_scores, width, color='lightgreen', alpha=0.7)
        ax2.set_title('Transition Smoothness')
        ax2.set_ylabel('Score (1-5)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Change {i+1}' for i in change_ids])
        ax2.set_ylim(0, 5)
        ax2.grid(True, alpha=0.3)
        
        ax3.bar(x, execution_scores, width, color='orange', alpha=0.7)
        ax3.set_title('Execution Quality')
        ax3.set_ylabel('Score (1-5)')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Change {i+1}' for i in change_ids])
        ax3.set_ylim(0, 5)
        ax3.grid(True, alpha=0.3)
        
        ax4.bar(x, overall_scores, width, color='coral', alpha=0.7)
        ax4.set_title('Overall Score')
        ax4.set_ylabel('Score (1-5)')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'Change {i+1}' for i in change_ids])
        ax4.set_ylim(0, 5)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for ax, scores in [(ax1, clarity_scores), (ax2, smoothness_scores), 
                          (ax3, execution_scores), (ax4, overall_scores)]:
            for i, score in enumerate(scores):
                ax.text(i, score + 0.1, str(score), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{trajectory_id}_scores.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Score visualization saved: {output_path}")
    
    def plot_score_distribution(self, trajectory_id: str):
        """Plot distribution of annotation scores."""
        annotation_path = os.path.join(self.analysis_dir, "annotations", f"{trajectory_id}_annotations.json")
        
        if not os.path.exists(annotation_path):
            print(f"Annotations not found: {annotation_path}")
            return
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = data['annotations']
        completed = [a for a in annotations if a.get('overall_score') is not None]
        
        if not completed:
            print("No completed annotations found")
            return
        
        # Collect all scores
        all_scores = {
            'Instruction Clarity': [a['instruction_clarity'] for a in completed],
            'Transition Smoothness': [a['transition_smoothness'] for a in completed],
            'Execution Quality': [a['execution_quality'] for a in completed],
            'Overall Score': [a['overall_score'] for a in completed]
        }
        
        # Create distribution plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        positions = []
        labels = []
        colors = ['skyblue', 'lightgreen', 'orange', 'coral']
        
        for i, (metric, scores) in enumerate(all_scores.items()):
            positions.extend([i + 1] * len(scores))
            labels.extend([metric] * len(scores))
        
        # Create violin plot
        parts = ax.violinplot([scores for scores in all_scores.values()], 
                             positions=range(1, len(all_scores) + 1),
                             showmeans=True, showmedians=True)
        
        # Customize violin plot
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(1, len(all_scores) + 1))
        ax.set_xticklabels(all_scores.keys(), rotation=45, ha='right')
        ax.set_ylabel('Score (1-5)')
        ax.set_title(f'Score Distribution: {trajectory_id}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 6)
        ax.grid(True, alpha=0.3)
        
        # Add mean values as text
        for i, (metric, scores) in enumerate(all_scores.items()):
            mean_score = np.mean(scores)
            ax.text(i + 1, 5.5, f'μ={mean_score:.1f}', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{trajectory_id}_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distribution visualization saved: {output_path}")
    
    def create_summary_dashboard(self, trajectory_id: str):
        """Create a comprehensive dashboard with all visualizations."""
        try:
            self.plot_instruction_timeline(trajectory_id)
            self.plot_annotation_scores(trajectory_id)
            self.plot_score_distribution(trajectory_id)
            
            print(f"\\n✅ All visualizations created for {trajectory_id}")
            print(f"Check the '{self.output_dir}' directory for generated plots.")
            
        except ImportError:
            print("⚠️ Matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")

def main():
    """Main function for command-line usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_instruction_changes.py <trajectory_id>")
        return
    
    trajectory_id = sys.argv[1]
    visualizer = InstructionChangeVisualizer()
    visualizer.create_summary_dashboard(trajectory_id)

if __name__ == "__main__":
    main()
