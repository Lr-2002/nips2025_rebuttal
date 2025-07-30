#!/usr/bin/env python3
"""
Model Performance Visualization for COIN Benchmark

Analyzes and visualizes model performance across primitive-composition-interactive tasks
focusing on trajectory stability and gripper performance metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelPerformanceAnalyzer:
    """Analyze and visualize model performance across different task difficulties."""
    
    def __init__(self, results_dir: str):
        """Initialize analyzer with results directory."""
        self.results_dir = Path(results_dir)
        self.data = {}
        self.task_orders = ['primitive', 'composition', 'interactive']
        
        # Model name mapping for cleaner display
        self.model_mapping = {
            'cogact': 'CogACT',
            'gr00t': 'GR00T', 
            'pi0': 'PI0'
        }
        
        # Color mapping for models
        self.model_colors = {
            'CogACT': '#FF6B6B',
            'GR00T': '#4ECDC4', 
            'PI0': '#45B7D1'
        }
        
    def load_all_data(self):
        """Load all summary statistics from different models and task difficulties."""
        print("Loading performance data...")
        
        for task_type in self.task_orders:
            task_dir = self.results_dir / task_type
            if not task_dir.exists():
                print(f"Warning: {task_type} directory not found")
                continue
                
            self.data[task_type] = {}
            
            for model_dir in task_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                # Extract model name from directory
                model_name = self._extract_model_name(model_dir.name)
                summary_file = model_dir / 'summary_statistics.json'
                
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            data = json.load(f)
                            self.data[task_type][model_name] = data
                            print(f"Loaded {task_type}/{model_name}")
                    except Exception as e:
                        print(f"Error loading {summary_file}: {e}")
                        
        print(f"Data loading complete. Found {sum(len(v) for v in self.data.values())} model results.")
        
    def _extract_model_name(self, dir_name: str) -> str:
        """Extract clean model name from directory name."""
        dir_lower = dir_name.lower()
        for key, clean_name in self.model_mapping.items():
            if key in dir_lower:
                return clean_name
        return dir_name  # fallback
        
    def create_comprehensive_analysis(self):
        """Create comprehensive visualization analysis."""
        if not self.data:
            self.load_all_data()
            
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Overall Trajectory Stability Comparison
        ax1 = plt.subplot(3, 2, 1)
        self._plot_trajectory_stability(ax1)
        
        # 2. Velocity and Acceleration Performance
        ax2 = plt.subplot(3, 2, 2)
        self._plot_kinematic_performance(ax2)
        
        # 3. Gripper Control Performance
        ax3 = plt.subplot(3, 2, 3)
        self._plot_gripper_performance(ax3)
        
        # 4. Gripper Coordination Analysis
        ax4 = plt.subplot(3, 2, 4)
        self._plot_gripper_coordination(ax4)
        
        # 5. Problem Detection Rates
        ax5 = plt.subplot(3, 2, 5)
        self._plot_problem_detection(ax5)
        
        # 6. Gripper Usage Patterns
        ax6 = plt.subplot(3, 2, 6)
        self._plot_gripper_usage(ax6)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comprehensive_model_analysis.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Comprehensive analysis saved to: {self.results_dir / 'comprehensive_model_analysis.png'}")
        
        # Create detailed individual plots
        self._create_detailed_plots()
        
    def _plot_trajectory_stability(self, ax):
        """Plot overall trajectory stability across models and difficulties."""
        data_matrix = []
        models = []
        
        for task_type in self.task_orders:
            if task_type not in self.data:
                continue
                
            for model, stats in self.data[task_type].items():
                if 'traj_overall_stability_mean' in stats:
                    data_matrix.append([
                        task_type,
                        model,
                        stats['traj_overall_stability_mean'],
                        stats.get('traj_overall_stability_std', 0)
                    ])
                    
        if not data_matrix:
            ax.text(0.5, 0.5, 'No trajectory stability data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        df = pd.DataFrame(data_matrix, columns=['Task', 'Model', 'Stability', 'Std'])
        
        # Create grouped bar plot
        task_positions = {task: i for i, task in enumerate(self.task_orders)}
        model_offsets = {model: i*0.25 for i, model in enumerate(sorted(df['Model'].unique()))}
        
        for model in sorted(df['Model'].unique()):
            model_data = df[df['Model'] == model]
            positions = [task_positions[task] + model_offsets[model] for task in model_data['Task']]
            
            ax.bar(positions, model_data['Stability'], 
                  yerr=model_data['Std'], width=0.2, 
                  label=model, color=self.model_colors.get(model, 'gray'),
                  alpha=0.8, capsize=3)
        
        ax.set_xlabel('Task Difficulty')
        ax.set_ylabel('Trajectory Stability Score')
        ax.set_title('Overall Trajectory Stability Comparison')
        ax.set_xticks([i + 0.25 for i in range(len(self.task_orders))])
        ax.set_xticklabels([t.capitalize() for t in self.task_orders])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='VLA Explosion Threshold')
        
    def _plot_kinematic_performance(self, ax):
        """Plot velocity and acceleration performance."""
        metrics = ['traj_velocity_stability_mean', 'traj_acceleration_stability_mean']
        metric_labels = ['Velocity Smoothness', 'Acceleration Smoothness']
        
        data_for_plot = []
        
        for task_type in self.task_orders:
            if task_type not in self.data:
                continue
                
            for model, stats in self.data[task_type].items():
                for metric, label in zip(metrics, metric_labels):
                    if metric in stats:
                        data_for_plot.append({
                            'Task': task_type.capitalize(),
                            'Model': model,
                            'Metric': label,
                            'Score': stats[metric]
                        })
        
        if not data_for_plot:
            ax.text(0.5, 0.5, 'No kinematic data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        df = pd.DataFrame(data_for_plot)
        
        # Create grouped bar plot
        sns.barplot(data=df, x='Task', y='Score', hue='Model', ax=ax)
        ax.set_title('Kinematic Performance (Velocity & Acceleration)')
        ax.set_ylabel('Smoothness Score')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)
        
    def _plot_gripper_performance(self, ax):
        """Plot gripper control performance."""
        data_matrix = []
        
        for task_type in self.task_orders:
            if task_type not in self.data:
                continue
                
            for model, stats in self.data[task_type].items():
                if 'gripper_overall_stability_mean' in stats:
                    data_matrix.append([
                        task_type.capitalize(),
                        model,
                        stats['gripper_overall_stability_mean'],
                        stats.get('gripper_overall_stability_std', 0)
                    ])
                    
        if not data_matrix:
            ax.text(0.5, 0.5, 'No gripper data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        df = pd.DataFrame(data_matrix, columns=['Task', 'Model', 'Stability', 'Std'])
        
        sns.barplot(data=df, x='Task', y='Stability', hue='Model', ax=ax)
        ax.set_title('Gripper Control Stability')
        ax.set_ylabel('Gripper Stability Score')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Erratic Control Threshold')
        
    def _plot_gripper_coordination(self, ax):
        """Plot gripper coordination performance."""
        data_matrix = []
        
        for task_type in self.task_orders:
            if task_type not in self.data:
                continue
                
            for model, stats in self.data[task_type].items():
                if 'gripper_coordination_mean' in stats:
                    data_matrix.append([
                        task_type.capitalize(),
                        model,
                        stats['gripper_coordination_mean']
                    ])
                    
        if not data_matrix:
            ax.text(0.5, 0.5, 'No coordination data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        df = pd.DataFrame(data_matrix, columns=['Task', 'Model', 'Coordination'])
        
        sns.barplot(data=df, x='Task', y='Coordination', hue='Model', ax=ax)
        ax.set_title('Gripper-Arm Coordination Performance')
        ax.set_ylabel('Coordination Score')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)
        
    def _plot_problem_detection(self, ax):
        """Plot problem detection rates."""
        data_matrix = []
        
        for task_type in self.task_orders:
            if task_type not in self.data:
                continue
                
            for model, stats in self.data[task_type].items():
                vla_rate = stats.get('vla_explosion_rate', 0) * 100
                gripper_rate = stats.get('erratic_gripper_rate', 0) * 100
                
                data_matrix.extend([
                    [task_type.capitalize(), model, 'VLA Explosions', vla_rate],
                    [task_type.capitalize(), model, 'Erratic Gripper', gripper_rate]
                ])
                
        if not data_matrix:
            ax.text(0.5, 0.5, 'No problem detection data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        df = pd.DataFrame(data_matrix, columns=['Task', 'Model', 'Problem', 'Rate'])
        
        sns.barplot(data=df, x='Task', y='Rate', hue='Model', ax=ax)
        ax.set_title('Problem Detection Rates')
        ax.set_ylabel('Problem Rate (%)')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
    def _plot_gripper_usage(self, ax):
        """Plot gripper usage patterns."""
        data_matrix = []
        
        for task_type in self.task_orders:
            if task_type not in self.data:
                continue
                
            for model, stats in self.data[task_type].items():
                avg_changes = stats.get('avg_gripper_changes', 0)
                expected_changes = stats.get('avg_expected_gripper_changes', 8)
                overuse_ratio = avg_changes / expected_changes if expected_changes > 0 else 0
                
                data_matrix.append([
                    task_type.capitalize(),
                    model,
                    avg_changes,
                    expected_changes,
                    overuse_ratio
                ])
                
        if not data_matrix:
            ax.text(0.5, 0.5, 'No gripper usage data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        df = pd.DataFrame(data_matrix, columns=['Task', 'Model', 'Actual', 'Expected', 'Ratio'])
        
        sns.barplot(data=df, x='Task', y='Ratio', hue='Model', ax=ax)
        ax.set_title('Gripper Overuse Ratio (Actual/Expected Changes)')
        ax.set_ylabel('Overuse Ratio')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Expected Usage')
        
    def _create_detailed_plots(self):
        """Create detailed individual plots for each metric."""
        print("Creating detailed individual plots...")
        
        # 1. Detailed trajectory stability heatmap
        self._create_stability_heatmap()
        
        # 2. Gripper performance radar chart
        self._create_gripper_radar_chart()
        
        # 3. Task difficulty progression
        self._create_difficulty_progression()
        
    def _create_stability_heatmap(self):
        """Create heatmap of stability metrics across models and tasks."""
        metrics = [
            'traj_overall_stability_mean',
            'traj_velocity_stability_mean', 
            'traj_acceleration_stability_mean',
            'traj_jerk_stability_mean'
        ]
        
        metric_labels = [
            'Overall Trajectory',
            'Velocity Smoothness',
            'Acceleration Smoothness', 
            'Jerk Smoothness'
        ]
        
        # Prepare data matrix
        data_matrix = []
        index_labels = []
        
        for task_type in self.task_orders:
            if task_type not in self.data:
                continue
                
            for model in sorted(self.data[task_type].keys()):
                stats = self.data[task_type][model]
                row = []
                for metric in metrics:
                    row.append(stats.get(metric, 0))
                data_matrix.append(row)
                index_labels.append(f"{task_type.capitalize()}-{model}")
        
        if not data_matrix:
            print("No data available for stability heatmap")
            return
            
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(data_matrix, 
                   xticklabels=metric_labels,
                   yticklabels=index_labels,
                   annot=True, fmt='.3f', cmap='RdYlBu_r',
                   cbar_kws={'label': 'Stability Score'})
        
        plt.title('Trajectory Stability Metrics Heatmap')
        plt.xlabel('Stability Metrics')
        plt.ylabel('Model-Task Combinations')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'stability_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_gripper_radar_chart(self):
        """Create radar chart for gripper performance metrics."""
        from math import pi
        
        metrics = ['gripper_smoothness_mean', 'gripper_frequency_mean', 'gripper_coordination_mean']
        metric_labels = ['Smoothness', 'Frequency', 'Coordination']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
        
        for i, task_type in enumerate(self.task_orders):
            if task_type not in self.data:
                continue
                
            ax = axes[i]
            
            # Number of variables
            N = len(metric_labels)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            for model, stats in self.data[task_type].items():
                values = []
                for metric in metrics:
                    values.append(stats.get(metric, 0))
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=model, color=self.model_colors.get(model, 'gray'))
                ax.fill(angles, values, alpha=0.25, color=self.model_colors.get(model, 'gray'))
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1)
            ax.set_title(f'{task_type.capitalize()} Tasks', size=16, weight='bold')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'gripper_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_difficulty_progression(self):
        """Create line plot showing performance progression across task difficulties."""
        metrics_to_plot = [
            ('traj_overall_stability_mean', 'Trajectory Stability'),
            ('gripper_overall_stability_mean', 'Gripper Stability')
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[i]
            
            for model in ['CogACT', 'GR00T', 'PI0']:
                x_values = []
                y_values = []
                y_errors = []
                
                for j, task_type in enumerate(self.task_orders):
                    if task_type in self.data and model in self.data[task_type]:
                        stats = self.data[task_type][model]
                        if metric in stats:
                            x_values.append(j)
                            y_values.append(stats[metric])
                            y_errors.append(stats.get(metric.replace('_mean', '_std'), 0))
                
                if x_values:
                    ax.errorbar(x_values, y_values, yerr=y_errors, 
                              marker='o', linewidth=2, markersize=8,
                              label=model, color=self.model_colors.get(model, 'gray'),
                              capsize=5, capthick=2)
            
            ax.set_xlabel('Task Difficulty')
            ax.set_ylabel('Stability Score')
            ax.set_title(f'{title} Progression')
            ax.set_xticks(range(len(self.task_orders)))
            ax.set_xticklabels([t.capitalize() for t in self.task_orders])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add threshold lines
            if 'traj' in metric:
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='VLA Explosion Threshold')
            else:
                ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Erratic Control Threshold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'difficulty_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def export_data_to_csv(self):
        """Export all visualization data to CSV files."""
        if not self.data:
            self.load_all_data()
            
        print("Exporting visualization data to CSV files...")
        
        # 1. Main performance metrics
        self._export_main_metrics_csv()
        
        # 2. Detailed trajectory stability metrics
        self._export_trajectory_metrics_csv()
        
        # 3. Gripper performance metrics
        self._export_gripper_metrics_csv()
        
        # 4. Problem detection rates
        self._export_problem_detection_csv()
        
        # 5. Task difficulty progression
        self._export_difficulty_progression_csv()
        
        print("✅ All CSV files exported successfully!")
        
    def _export_main_metrics_csv(self):
        """Export main performance metrics to CSV."""
        data_rows = []
        
        for task_type in self.task_orders:
            if task_type not in self.data:
                continue
                
            for model, stats in self.data[task_type].items():
                data_rows.append({
                    'Task_Type': task_type,
                    'Model': model,
                    'Total_Trajectories': stats.get('total_trajectories', 0),
                    'Total_Actions': stats.get('total_actions', 0),
                    'Avg_Actions_Per_Trajectory': stats.get('avg_actions_per_trajectory', 0),
                    'Trajectory_Stability_Mean': stats.get('traj_overall_stability_mean', 0),
                    'Trajectory_Stability_Std': stats.get('traj_overall_stability_std', 0),
                    'Gripper_Stability_Mean': stats.get('gripper_overall_stability_mean', 0),
                    'Gripper_Stability_Std': stats.get('gripper_overall_stability_std', 0),
                    'VLA_Explosion_Rate': stats.get('vla_explosion_rate', 0),
                    'Erratic_Gripper_Rate': stats.get('erratic_gripper_rate', 0),
                    'Gripper_Overuse_Ratio': stats.get('gripper_overuse_ratio', 0)
                })
        
        df = pd.DataFrame(data_rows)
        df.to_csv(self.results_dir / 'main_performance_metrics.csv', index=False)
        print("  ✓ main_performance_metrics.csv")
        
    def _export_trajectory_metrics_csv(self):
        """Export detailed trajectory stability metrics to CSV."""
        data_rows = []
        
        trajectory_metrics = [
            'traj_overall_stability_mean', 'traj_overall_stability_std',
            'traj_velocity_stability_mean', 'traj_velocity_stability_std',
            'traj_acceleration_stability_mean', 'traj_acceleration_stability_std',
            'traj_jerk_stability_mean', 'traj_jerk_stability_std',
            'traj_position_stability_mean', 'traj_position_stability_std'
        ]
        
        for task_type in self.task_orders:
            if task_type not in self.data:
                continue
                
            for model, stats in self.data[task_type].items():
                row = {
                    'Task_Type': task_type,
                    'Model': model
                }
                
                for metric in trajectory_metrics:
                    row[metric.replace('traj_', '').replace('_mean', '_Mean').replace('_std', '_Std')] = stats.get(metric, 0)
                
                data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        df.to_csv(self.results_dir / 'trajectory_stability_metrics.csv', index=False)
        print("  ✓ trajectory_stability_metrics.csv")
        
    def _export_gripper_metrics_csv(self):
        """Export gripper performance metrics to CSV."""
        data_rows = []
        
        gripper_metrics = [
            'gripper_overall_stability_mean', 'gripper_overall_stability_std',
            'gripper_smoothness_mean', 'gripper_smoothness_std',
            'gripper_frequency_mean', 'gripper_frequency_std',
            'gripper_coordination_mean', 'gripper_coordination_std',
            'avg_gripper_changes', 'avg_expected_gripper_changes'
        ]
        
        for task_type in self.task_orders:
            if task_type not in self.data:
                continue
                
            for model, stats in self.data[task_type].items():
                row = {
                    'Task_Type': task_type,
                    'Model': model
                }
                
                for metric in gripper_metrics:
                    clean_name = metric.replace('gripper_', '').replace('avg_', 'Avg_')
                    clean_name = clean_name.replace('_mean', '_Mean').replace('_std', '_Std')
                    row[clean_name] = stats.get(metric, 0)
                
                # Add calculated metrics
                row['Gripper_Overuse_Ratio'] = stats.get('gripper_overuse_ratio', 0)
                
                data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        df.to_csv(self.results_dir / 'gripper_performance_metrics.csv', index=False)
        print("  ✓ gripper_performance_metrics.csv")
        
    def _export_problem_detection_csv(self):
        """Export problem detection rates to CSV."""
        data_rows = []
        
        for task_type in self.task_orders:
            if task_type not in self.data:
                continue
                
            for model, stats in self.data[task_type].items():
                # VLA Explosions
                data_rows.append({
                    'Task_Type': task_type,
                    'Model': model,
                    'Problem_Type': 'VLA_Explosions',
                    'Rate': stats.get('vla_explosion_rate', 0),
                    'Rate_Percentage': stats.get('vla_explosion_rate', 0) * 100
                })
                
                # Erratic Gripper
                data_rows.append({
                    'Task_Type': task_type,
                    'Model': model,
                    'Problem_Type': 'Erratic_Gripper',
                    'Rate': stats.get('erratic_gripper_rate', 0),
                    'Rate_Percentage': stats.get('erratic_gripper_rate', 0) * 100
                })
        
        df = pd.DataFrame(data_rows)
        df.to_csv(self.results_dir / 'problem_detection_rates.csv', index=False)
        print("  ✓ problem_detection_rates.csv")
        
    def _export_difficulty_progression_csv(self):
        """Export task difficulty progression data to CSV."""
        data_rows = []
        
        key_metrics = [
            'traj_overall_stability_mean', 'traj_overall_stability_std',
            'gripper_overall_stability_mean', 'gripper_overall_stability_std'
        ]
        
        for model in ['CogACT', 'GR00T', 'PI0']:
            for i, task_type in enumerate(self.task_orders):
                if task_type in self.data and model in self.data[task_type]:
                    stats = self.data[task_type][model]
                    
                    data_rows.append({
                        'Model': model,
                        'Task_Type': task_type,
                        'Task_Order': i,
                        'Trajectory_Stability_Mean': stats.get('traj_overall_stability_mean', 0),
                        'Trajectory_Stability_Std': stats.get('traj_overall_stability_std', 0),
                        'Gripper_Stability_Mean': stats.get('gripper_overall_stability_mean', 0),
                        'Gripper_Stability_Std': stats.get('gripper_overall_stability_std', 0)
                    })
        
        df = pd.DataFrame(data_rows)
        df.to_csv(self.results_dir / 'difficulty_progression.csv', index=False)
        print("  ✓ difficulty_progression.csv")
        
    def print_summary_report(self):
        """Print comprehensive summary report."""
        if not self.data:
            self.load_all_data()
            
        print("\n" + "="*80)
        print("COIN BENCHMARK MODEL PERFORMANCE ANALYSIS REPORT")
        print("="*80)
        
        for task_type in self.task_orders:
            if task_type not in self.data:
                continue
                
            print(f"\n📊 {task_type.upper()} TASKS:")
            print("-" * 50)
            
            for model, stats in self.data[task_type].items():
                print(f"\n🤖 {model}:")
                print(f"  Trajectories Analyzed: {stats.get('total_trajectories', 'N/A')}")
                print(f"  Trajectory Stability: {stats.get('traj_overall_stability_mean', 0):.3f} ± {stats.get('traj_overall_stability_std', 0):.3f}")
                print(f"  Gripper Stability: {stats.get('gripper_overall_stability_mean', 0):.3f} ± {stats.get('gripper_overall_stability_std', 0):.3f}")
                print(f"  VLA Explosion Rate: {stats.get('vla_explosion_rate', 0)*100:.1f}%")
                print(f"  Erratic Gripper Rate: {stats.get('erratic_gripper_rate', 0)*100:.1f}%")
                print(f"  Gripper Overuse Ratio: {stats.get('gripper_overuse_ratio', 0):.1f}x")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - Visualizations saved to analysis_results/")
        print("="*80)

def main():
    """Main execution function."""
    results_dir = "/Users/lr-2002/project/reasoning_manipulation/rebuttal/analysis_results"
    
    analyzer = ModelPerformanceAnalyzer(results_dir)
    
    # Load data and create visualizations
    analyzer.load_all_data()
    analyzer.create_comprehensive_analysis()
    
    # Export data to CSV files
    analyzer.export_data_to_csv()
    
    analyzer.print_summary_report()
    
    print("\n✅ Model performance visualization analysis complete!")
    print(f"📊 Check {results_dir}/ for generated plots:")
    print("  - comprehensive_model_analysis.png")
    print("  - stability_heatmap.png") 
    print("  - gripper_radar_chart.png")
    print("  - difficulty_progression.png")
    print(f"\n📋 Check {results_dir}/ for exported CSV data:")
    print("  - main_performance_metrics.csv")
    print("  - trajectory_stability_metrics.csv")
    print("  - gripper_performance_metrics.csv")
    print("  - problem_detection_rates.csv")
    print("  - difficulty_progression.csv")

if __name__ == "__main__":
    main()
