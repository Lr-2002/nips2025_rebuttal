import os
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class TrajectoryData:
    """Data structure for a single trajectory"""
    task_name: str
    trajectory_id: str
    trajectory_info: Dict[str, Any]
    actions: List[Any]
    video_path: str
    chat_data: Dict[str, Any]
    vqa_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class RebuttalDataLoader:
    """Data loader for rebuttal trajectory data"""
    
    def __init__(self, data_root: str = "/Users/lr-2002/project/reasoning_manipulation/rebuttal/remote_data"):
        self.data_root = Path(data_root)
        self.trajectories = {}
        self._load_all_trajectories()
    
    def _parse_folder_name(self, folder_name: str) -> Dict[str, str]:
        """Parse folder name to extract task information"""
        # Example: 20250511_013942_Tabletop-Seek-Holder-InCabinet-v1_gr00t_gemini-2.0-flash
        parts = folder_name.split('_')
        
        if len(parts) >= 4:
            date = parts[0]
            time = parts[1]
            task_name = parts[2]
            model_info = '_'.join(parts[3:])
            
            return {
                'date': date,
                'time': time,
                'task_name': task_name,
                'model_info': model_info,
                'trajectory_id': f"{date}_{time}"
            }
        else:
            return {
                'date': 'unknown',
                'time': 'unknown', 
                'task_name': folder_name,
                'model_info': 'unknown',
                'trajectory_id': folder_name
            }
    
    def _load_trajectory_files(self, traj_folder: Path) -> Dict[str, Any]:
        """Load all files for a single trajectory"""
        files = {}
        
        # Find trajectory files
        for file_path in traj_folder.iterdir():
            if file_path.is_file():
                if file_path.suffix == '.pkl':
                    # Load pickle file (actions/trajectory info)
                    try:
                        with open(file_path, 'rb') as f:
                            files['trajectory_pkl'] = pickle.load(f)
                        files['trajectory_pkl_path'] = str(file_path)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        files['trajectory_pkl'] = None
                
                elif file_path.suffix == '.mp4':
                    # Video file
                    files['video_path'] = str(file_path)
                
                elif file_path.name.endswith('_chat.json'):
                    # Chat data
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            files['chat_data'] = json.load(f)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        files['chat_data'] = None
                
                elif file_path.name.endswith('_vqa.json'):
                    # VQA data
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            files['vqa_data'] = json.load(f)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        files['vqa_data'] = None
        
        # Load metadata if exists
        metadata_folder = traj_folder / 'metadata'
        if metadata_folder.exists():
            for meta_file in metadata_folder.iterdir():
                if meta_file.suffix == '.json':
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            files['metadata'] = json.load(f)
                        break
                    except Exception as e:
                        print(f"Error loading metadata {meta_file}: {e}")
        
        return files
    
    def _extract_actions(self, trajectory_pkl: Any) -> List[Any]:
        """Extract actions from trajectory pickle data"""
        if trajectory_pkl is None:
            return []
        
        # Handle different possible structures
        if isinstance(trajectory_pkl, dict):
            if 'actions' in trajectory_pkl:
                return trajectory_pkl['actions']
            elif 'trajectory' in trajectory_pkl:
                traj = trajectory_pkl['trajectory']
                if isinstance(traj, dict) and 'actions' in traj:
                    return traj['actions']
                elif isinstance(traj, list):
                    return traj
        elif isinstance(trajectory_pkl, list):
            return trajectory_pkl
        
        # Handle numpy arrays
        try:
            import numpy as np
            if isinstance(trajectory_pkl, np.ndarray):
                return trajectory_pkl.tolist()
        except ImportError:
            pass
        
        # If it's any other iterable, try to convert to list
        try:
            if hasattr(trajectory_pkl, '__iter__') and not isinstance(trajectory_pkl, (str, bytes)):
                return list(trajectory_pkl)
        except:
            pass
        
        return []
    
    def _extract_trajectory_info(self, trajectory_pkl: Any) -> Dict[str, Any]:
        """Extract trajectory information from pickle data"""
        if trajectory_pkl is None:
            return {}
        
        if isinstance(trajectory_pkl, dict):
            # Return the whole dict but exclude actions to avoid duplication
            info = trajectory_pkl.copy()
            if 'actions' in info:
                del info['actions']
            return info
        
        # Handle numpy arrays and other data types
        info = {'raw_data_type': str(type(trajectory_pkl))}
        
        try:
            import numpy as np
            if isinstance(trajectory_pkl, np.ndarray):
                info.update({
                    'shape': trajectory_pkl.shape,
                    'dtype': str(trajectory_pkl.dtype),
                    'size': trajectory_pkl.size
                })
        except ImportError:
            pass
        
        # Add length info if available
        if hasattr(trajectory_pkl, '__len__'):
            info['length'] = len(trajectory_pkl)
        
        return info
    
    def _load_all_trajectories(self):
        """Load all trajectories from the data root"""
        if not self.data_root.exists():
            print(f"Data root {self.data_root} does not exist")
            return
        
        for folder in self.data_root.iterdir():
            if folder.is_dir():
                # Parse folder name
                folder_info = self._parse_folder_name(folder.name)
                
                # Load trajectory files
                files = self._load_trajectory_files(folder)
                
                # Extract actions and trajectory info
                actions = self._extract_actions(files.get('trajectory_pkl'))
                trajectory_info = self._extract_trajectory_info(files.get('trajectory_pkl'))
                
                # Create TrajectoryData object
                traj_data = TrajectoryData(
                    task_name=folder_info['task_name'],
                    trajectory_id=folder_info['trajectory_id'],
                    trajectory_info=trajectory_info,
                    actions=actions,
                    video_path=files.get('video_path', ''),
                    chat_data=files.get('chat_data', {}),
                    vqa_data=files.get('vqa_data'),
                    metadata=files.get('metadata')
                )
                
                # Store in trajectories dict
                self.trajectories[folder_info['trajectory_id']] = traj_data
                
                print(f"Loaded trajectory: {folder_info['trajectory_id']} - {folder_info['task_name']}")
    
    def get_trajectory(self, trajectory_id: str) -> Optional[TrajectoryData]:
        """Get a specific trajectory by ID"""
        return self.trajectories.get(trajectory_id)
    
    def get_trajectories_by_task(self, task_name: str) -> List[TrajectoryData]:
        """Get all trajectories for a specific task"""
        return [traj for traj in self.trajectories.values() if traj.task_name == task_name]
    
    def get_all_trajectories(self) -> Dict[str, TrajectoryData]:
        """Get all loaded trajectories"""
        return self.trajectories
    
    def get_task_names(self) -> List[str]:
        """Get all unique task names"""
        return list(set(traj.task_name for traj in self.trajectories.values()))
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get a summary of all loaded trajectories"""
        summary = {
            'total_trajectories': len(self.trajectories),
            'task_names': self.get_task_names(),
            'trajectories_by_task': {}
        }
        
        for task_name in summary['task_names']:
            task_trajs = self.get_trajectories_by_task(task_name)
            summary['trajectories_by_task'][task_name] = {
                'count': len(task_trajs),
                'trajectory_ids': [traj.trajectory_id for traj in task_trajs]
            }
        
        return summary
    
    def print_summary(self):
        """Print a summary of loaded data"""
        summary = self.get_trajectory_summary()
        print(f"\n=== Rebuttal Data Summary ===")
        print(f"Total trajectories: {summary['total_trajectories']}")
        print(f"Unique tasks: {len(summary['task_names'])}")
        print("\nTasks:")
        for task_name, info in summary['trajectories_by_task'].items():
            print(f"  - {task_name}: {info['count']} trajectories")
            for traj_id in info['trajectory_ids']:
                traj = self.get_trajectory(traj_id)
                print(f"    * {traj_id} - Actions: {len(traj.actions)}, Video: {bool(traj.video_path)}")

# Example usage
if __name__ == "__main__":
    # Load all trajectories
    loader = RebuttalDataLoader()
    
    # Print summary
    loader.print_summary()
    
    # Example: Get a specific trajectory
    trajectories = loader.get_all_trajectories()
    if trajectories:
        first_traj_id = list(trajectories.keys())[0]
        first_traj = loader.get_trajectory(first_traj_id)
        print(f"\n=== Example Trajectory: {first_traj_id} ===")
        print(f"Task: {first_traj.task_name}")
        print(f"Actions: {len(first_traj.actions)}")
        print(f"Video: {first_traj.video_path}")
        print(f"Chat data steps: {len(first_traj.chat_data.get('data', []))}")
        if first_traj.vqa_data:
            print(f"VQA data available: Yes")
        if first_traj.metadata:
            print(f"Metadata available: Yes")
