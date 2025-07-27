#!/usr/bin/env python3
"""
Debug script to understand the data format in remote_data
"""

import pickle
import json
import numpy as np
from pathlib import Path

def debug_single_trajectory():
    """Debug a single trajectory to understand data format."""
    
    # Pick the first trajectory
    data_root = Path("/Users/lr-2002/project/reasoning_manipulation/rebuttal/remote_data")
    traj_dir = data_root / "20250509_gr00t_120000_gemini_10_400_no_history_image" / "20250510_002539_Tabletop-Balance-Pivot-WithBalls-v1_gr00t_gemini-2.0-flash"
    
    pkl_file = traj_dir / "traj_20250510_002539.pkl"
    chat_file = traj_dir / "traj_20250510_002539_chat.json"
    
    print("=== Debugging Data Format ===")
    print(f"PKL file: {pkl_file}")
    print(f"Chat file: {chat_file}")
    
    # Load and inspect pickle data
    try:
        with open(pkl_file, 'rb') as f:
            pkl_data = pickle.load(f)
        
        print(f"\n--- PKL Data Structure ---")
        print(f"Type: {type(pkl_data)}")
        
        if isinstance(pkl_data, dict):
            print(f"Keys: {list(pkl_data.keys())}")
            for key, value in pkl_data.items():
                print(f"  {key}: {type(value)}")
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    print(f"    Length: {len(value)}")
                    if hasattr(value[0], 'shape'):
                        print(f"    First element shape: {value[0].shape}")
                    else:
                        print(f"    First element type: {type(value[0])}")
                        if isinstance(value[0], (list, tuple)):
                            print(f"    First element length: {len(value[0])}")
        elif isinstance(pkl_data, list):
            print(f"List length: {len(pkl_data)}")
            if len(pkl_data) > 0:
                print(f"First element type: {type(pkl_data[0])}")
                if hasattr(pkl_data[0], 'shape'):
                    print(f"First element shape: {pkl_data[0].shape}")
        
    except Exception as e:
        print(f"Error loading PKL: {e}")
    
    # Load and inspect chat data
    try:
        with open(chat_file, 'r') as f:
            chat_data = json.load(f)
        
        print(f"\n--- Chat Data Structure ---")
        print(f"Type: {type(chat_data)}")
        
        if isinstance(chat_data, dict):
            print(f"Keys: {list(chat_data.keys())}")
            if 'data' in chat_data:
                print(f"Data length: {len(chat_data['data'])}")
                if len(chat_data['data']) > 0:
                    first_step = chat_data['data'][0]
                    print(f"First step keys: {list(first_step.keys())}")
                    if 'response' in first_step:
                        response = first_step['response']
                        print(f"Response keys: {list(response.keys())}")
                        if 'subtasks' in response:
                            print(f"Subtasks: {response['subtasks']}")
        
    except Exception as e:
        print(f"Error loading Chat: {e}")

if __name__ == "__main__":
    debug_single_trajectory()
