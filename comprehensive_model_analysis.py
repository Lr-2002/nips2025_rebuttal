#!/usr/bin/env python3
"""
Simplified Model Analysis for COIN Benchmark

This script analyzes all trajectories focusing on trajectory stability and gripper performance.
Optimized for batch processing of different models.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

from data_loader import RebuttalDataLoader
from metric import TrajectoryStabilityCalculator, GripperStabilityCalculator

class ComprehensiveModelAnalyzer:
    """
    Simplified analyzer focusing on trajectory stability and gripper performance.
    """
    
    def __init__(self, data_root: str):
        """Initialize the analyzer with data root directory."""
        self.data_root = Path(data_root)
        self.results = {}
        self.summary_stats = {}
        
        # Initialize calculators - only trajectory and gripper
        self.traj_calculator = TrajectoryStabilityCalculator(dt=1/50)  # 50Hz control
        self.gripper_calculator = GripperStabilityCalculator()
        
        # Task categories for analysis
        self.task_categories = {
            'manipulation': ['Pick', 'Put', 'Move', 'Insert', 'Rotate'],
            'navigation': ['Find', 'Seek'],
            'interaction': ['Open', 'Close', 'Balance', 'Clean'],
            'complex': ['Merge', 'Finish', 'Hanobi']
        }
    
    def discover_trajectories(self) -> List[Dict[str, Any]]:
        """Discover all trajectory files in the data directory."""
        trajectories = []
        
        # Navigate through the nested directory structure
        for exp_dir in self.data_root.iterdir():
            if not exp_dir.is_dir():
                continue
                
            for traj_dir in exp_dir.iterdir():
                if not traj_dir.is_dir():
                    continue
                
                # Look for trajectory files
                pkl_files = list(traj_dir.glob("*.pkl"))
                chat_files = list(traj_dir.glob("*_chat.json"))
                video_files = list(traj_dir.glob("*.mp4"))
                
                if pkl_files and chat_files:
                    traj_info = {
                        'trajectory_id': traj_dir.name,
                        'experiment_id': exp_dir.name,
                        'pkl_file': pkl_files[0],
                        'chat_file': chat_files[0],
                        'video_file': video_files[0] if video_files else None,
                        'task_name': self._extract_task_name(traj_dir.name)
                    }
                    trajectories.append(traj_info)
        
        print(f"Discovered {len(trajectories)} trajectories")
        return trajectories
    
    def _extract_task_name(self, traj_dir_name: str) -> str:
        """Extract task name from trajectory directory name."""
        parts = traj_dir_name.split('_')
        for i, part in enumerate(parts):
            if part.startswith('Tabletop-'):
                return '_'.join(parts[i:]).replace('_gr00t_gemini-2.0-flash', '')
        return traj_dir_name
    
    def _categorize_task(self, task_name: str) -> str:
        """Categorize task based on name."""
        task_upper = task_name.upper()
        for category, keywords in self.task_categories.items():
            if any(keyword.upper() in task_upper for keyword in keywords):
                return category
        return 'other'
    
    def load_trajectory_data(self, traj_info: Dict[str, Any]) -> Dict[str, Any]:
        """Load data for a single trajectory."""
        try:
            # Load pickle data (actions)
            with open(traj_info['pkl_file'], 'rb') as f:
                pkl_data = pickle.load(f)
            
            # Load chat data
            with open(traj_info['chat_file'], 'r') as f:
                chat_data = json.load(f)
            
            # Extract actions
            actions = self._extract_actions_from_pkl(pkl_data)
            
            return {
                'trajectory_id': traj_info['trajectory_id'],
                'task_name': traj_info['task_name'],
                'task_category': self._categorize_task(traj_info['task_name']),
                'actions': actions,
                'chat_data': chat_data,
                'pkl_data': pkl_data,
                'has_video': traj_info['video_file'] is not None
            }
            
        except Exception as e:
            print(f"Error loading {traj_info['trajectory_id']}: {e}")
            return None
    
    def _extract_actions_from_pkl(self, pkl_data: Any) -> np.ndarray:
        """Extract actions from pickle data."""
        # Handle direct numpy array (most common case)
        if isinstance(pkl_data, np.ndarray):
            return pkl_data
        
        # Handle dictionary format
        if isinstance(pkl_data, dict):
            if 'actions' in pkl_data:
                actions = pkl_data['actions']
            elif 'trajectory' in pkl_data:
                traj = pkl_data['trajectory']
                if isinstance(traj, dict) and 'actions' in traj:
                    actions = traj['actions']
                elif isinstance(traj, list):
                    actions = traj
                else:
                    actions = []
            else:
                actions = []
        elif isinstance(pkl_data, list):
            actions = pkl_data
        else:
            actions = []
        
        # Convert to numpy array
        if isinstance(actions, list) and actions:
            try:
                return np.array(actions)
            except:
                return np.array([])
        elif isinstance(actions, np.ndarray):
            return actions
        else:
            return np.array([])
    
    def analyze_single_trajectory(self, traj_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single trajectory focusing on trajectory stability and gripper performance."""
        if traj_data is None:
            return None
            
        try:
            actions = traj_data['actions']
            
            if len(actions) == 0:
                print(f"Warning: No actions found for {traj_data['trajectory_id']}")
                return None
            
            # Calculate trajectory stability metrics
            traj_metrics = self.traj_calculator.calculate_stability_metrics(actions)
            
            # Calculate gripper stability metrics
            gripper_metrics = self.gripper_calculator.calculate_gripper_stability(actions)
            
            # Compile results - only trajectory and gripper metrics
            result = {
                # Basic info
                'trajectory_id': traj_data['trajectory_id'],
                'task_name': traj_data['task_name'],
                'task_category': traj_data['task_category'],
                'total_actions': len(actions),
                
                # Trajectory stability metrics
                'traj_overall_stability': traj_metrics.overall_stability_score,
                'traj_velocity_stability': traj_metrics.velocity_smoothness,
                'traj_acceleration_stability': traj_metrics.acceleration_smoothness,
                'traj_jerk_stability': traj_metrics.jerk_smoothness,
                'traj_position_stability': traj_metrics.position_stability,
                
                # Gripper stability metrics
                'gripper_overall_stability': gripper_metrics.overall_stability_score,
                'gripper_smoothness': gripper_metrics.smoothness_score,
                'gripper_frequency': gripper_metrics.frequency_score,
                'gripper_coordination': gripper_metrics.coordination_score,
                'gripper_changes': gripper_metrics.total_gripper_changes,
                'gripper_expected_changes': gripper_metrics.expected_changes,
                
                # Problem detection flags
                'has_vla_explosion': traj_metrics.overall_stability_score < 0.5,
                'has_erratic_gripper': gripper_metrics.overall_stability_score < 0.6,
            }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing {traj_data['trajectory_id']}: {e}")
            return None
    
    def run_comprehensive_analysis(self) -> pd.DataFrame:
        """Run comprehensive analysis on all trajectories."""
        print("=== Comprehensive VLA Model Analysis ===")
        
        # Discover trajectories
        trajectories = self.discover_trajectories()
        
        # Analyze each trajectory
        results = []
        failed_count = 0
        
        for i, traj_info in enumerate(trajectories):
            print(f"Analyzing {i+1}/{len(trajectories)}: {traj_info['trajectory_id']}")
            
            # Load trajectory data
            traj_data = self.load_trajectory_data(traj_info)
            
            # Analyze trajectory
            result = self.analyze_single_trajectory(traj_data)
            
            if result is not None:
                results.append(result)
            else:
                failed_count += 1
        
        print(f"\nAnalysis complete: {len(results)} successful, {failed_count} failed")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        self.results_df = df
        
        return df
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics focused on trajectory and gripper metrics."""
        summary = {}
        
        # Basic statistics
        summary['total_trajectories'] = len(df)
        summary['total_actions'] = df['total_actions'].sum()
        summary['avg_actions_per_trajectory'] = df['total_actions'].mean()
        
        # Task category distribution
        summary['task_categories'] = df['task_category'].value_counts().to_dict()
        
        # Trajectory stability metrics summary
        traj_metrics = [
            'traj_overall_stability', 'traj_velocity_stability', 'traj_acceleration_stability',
            'traj_jerk_stability', 'traj_position_stability'
        ]
        
        # Gripper stability metrics summary
        gripper_metrics = [
            'gripper_overall_stability', 'gripper_smoothness', 'gripper_frequency', 'gripper_coordination'
        ]
        
        # Calculate statistics for all metrics
        all_metrics = traj_metrics + gripper_metrics
        for metric in all_metrics:
            summary[f'{metric}_mean'] = df[metric].mean()
            summary[f'{metric}_std'] = df[metric].std()
            summary[f'{metric}_min'] = df[metric].min()
            summary[f'{metric}_max'] = df[metric].max()
        
        # Problem detection summary
        summary['vla_explosions'] = df['has_vla_explosion'].sum()
        summary['erratic_gripper_count'] = df['has_erratic_gripper'].sum()
        summary['vla_explosion_rate'] = df['has_vla_explosion'].mean()
        summary['erratic_gripper_rate'] = df['has_erratic_gripper'].mean()
        
        # Problem rates by category
        for category in df['task_category'].unique():
            cat_df = df[df['task_category'] == category]
            summary[f'{category}_vla_explosion_rate'] = cat_df['has_vla_explosion'].mean()
            summary[f'{category}_erratic_gripper_rate'] = cat_df['has_erratic_gripper'].mean()
        
        # Gripper behavior analysis
        summary['avg_gripper_changes'] = df['gripper_changes'].mean()
        summary['avg_expected_gripper_changes'] = df['gripper_expected_changes'].mean()
        summary['gripper_overuse_ratio'] = (df['gripper_changes'] / df['gripper_expected_changes']).mean()
        
        self.summary_stats = summary
        return summary
    
    def print_analysis_report(self, df: pd.DataFrame, summary: Dict[str, Any]):
        """Print focused analysis report on trajectory stability and gripper performance."""
        print("\n" + "="*80)
        print("VLA MODEL TRAJECTORY & GRIPPER ANALYSIS REPORT")
        print("="*80)
        
        # Overall Statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total Trajectories: {summary['total_trajectories']}")
        print(f"  Total Actions: {summary['total_actions']:,}")
        print(f"  Average Actions per Trajectory: {summary['avg_actions_per_trajectory']:.1f}")
        
        # Task Distribution
        print(f"\n📋 TASK CATEGORY DISTRIBUTION:")
        for category, count in summary['task_categories'].items():
            percentage = (count / summary['total_trajectories']) * 100
            print(f"  {category.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Stability Metrics Overview
        print(f"\n📈 STABILITY METRICS OVERVIEW:")
        print(f"  Trajectory Stability: {summary['traj_overall_stability_mean']:.3f} ± {summary['traj_overall_stability_std']:.3f}")
        print(f"  Gripper Stability: {summary['gripper_overall_stability_mean']:.3f} ± {summary['gripper_overall_stability_std']:.3f}")
        
        # Problem Detection
        print(f"\n⚠️  PROBLEM DETECTION SUMMARY:")
        print(f"  VLA Action Explosions: {summary['vla_explosions']} ({summary['vla_explosion_rate']:.1%})")
        print(f"  Erratic Gripper Control: {summary['erratic_gripper_count']} ({summary['erratic_gripper_rate']:.1%})")
        
        # Category-specific Analysis
        print(f"\n📊 PROBLEM RATES BY TASK CATEGORY:")
        for category in df['task_category'].unique():
            vla_cat_rate = summary.get(f'{category}_vla_explosion_rate', 0)
            gripper_cat_rate = summary.get(f'{category}_erratic_gripper_rate', 0)
            
            print(f"  {category.capitalize()}:")
            print(f"    VLA Explosions: {vla_cat_rate:.1%}")
            print(f"    Erratic Gripper: {gripper_cat_rate:.1%}")
        
        # Gripper Behavior Analysis
        print(f"\n🤖 GRIPPER BEHAVIOR ANALYSIS:")
        print(f"  Average Gripper Changes: {summary['avg_gripper_changes']:.1f}")
        print(f"  Expected Gripper Changes: {summary['avg_expected_gripper_changes']:.1f}")
        print(f"  Gripper Overuse Ratio: {summary['gripper_overuse_ratio']:.2f}x")
        
        # Top Problematic Tasks
        print(f"\n🔍 TOP PROBLEMATIC TRAJECTORIES:")
        
        # Worst trajectory stability
        worst_traj = df.loc[df['traj_overall_stability'].idxmin()]
        print(f"  Worst Trajectory Stability: {worst_traj['trajectory_id']}")
        print(f"    Task: {worst_traj['task_name']}")
        print(f"    Score: {worst_traj['traj_overall_stability']:.3f}")
        
        # Worst gripper control
        worst_gripper = df.loc[df['gripper_overall_stability'].idxmin()]
        print(f"  Worst Gripper Control: {worst_gripper['trajectory_id']}")
        print(f"    Task: {worst_gripper['task_name']}")
        print(f"    Score: {worst_gripper['gripper_overall_stability']:.3f}")
        print(f"    Changes: {worst_gripper['gripper_changes']}/{worst_gripper['gripper_expected_changes']}")
        
        # Best performing trajectories
        print(f"\n✅ TOP PERFORMING TRAJECTORIES:")
        best_overall = df.loc[(df['traj_overall_stability'] + df['gripper_overall_stability']).idxmax()]
        print(f"  Best Overall: {best_overall['trajectory_id']}")
        print(f"    Task: {best_overall['task_name']}")
        print(f"    Trajectory Stability: {best_overall['traj_overall_stability']:.3f}")
        print(f"    Gripper Stability: {best_overall['gripper_overall_stability']:.3f}")
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
    
    def save_results(self, df: pd.DataFrame, output_dir: str = "analysis_results"):
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results
        df.to_csv(output_path / "comprehensive_analysis_results.csv", index=False)
        
        # Save summary statistics
        with open(output_path / "summary_statistics.json", 'w') as f:
            json.dump(self.summary_stats, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_path}/")
        print(f"  - comprehensive_analysis_results.csv")
        print(f"  - summary_statistics.json")

def main():
    """Main analysis function."""
    # Initialize analyzer
    data_root = "/Users/lr-2002/project/reasoning_manipulation/rebuttal/remote_data"
    analyzer = ComprehensiveModelAnalyzer(data_root)
    
    # Run comprehensive analysis
    df = analyzer.run_comprehensive_analysis()
    
    if len(df) == 0:
        print("No trajectories were successfully analyzed!")
        return
    
    # Generate summary statistics
    summary = analyzer.generate_summary_statistics(df)
    
    # Print analysis report
    analyzer.print_analysis_report(df, summary)
    
    # Save results
    analyzer.save_results(df)
    
    return df, summary

if __name__ == "__main__":
    results_df, summary_stats = main()
