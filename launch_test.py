#!/usr/bin/env python3
"""
Launch script for test.py that ensures the virtual environment is activated.
This script will automatically activate the pytorch_env virtual environment
and run the original test.py script.
"""

import subprocess
import sys
import os


def run_with_virtual_env():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the virtual environment
    venv_path = os.path.join(script_dir, 'pytorch_env')
    
    # Check if the virtual environment exists
    if not os.path.exists(venv_path):
        print(f"Error: Virtual environment not found at {venv_path}")
        print("Please run: python -m venv pytorch_env && source pytorch_env/bin/activate && pip install matplotlib torch torchvision")
        sys.exit(1)
    
    # Activate the virtual environment and run test.py
    python_executable = os.path.join(venv_path, 'bin', 'python')
    
    if not os.path.exists(python_executable):
        print(f"Error: Python executable not found at {python_executable}")
        sys.exit(1)
    
    # Prepare the command to run test.py with the virtual environment's Python
    cmd = [python_executable, os.path.join(script_dir, 'test.py')]
    
    # Add any additional arguments passed to this script
    cmd.extend(sys.argv[1:])
    
    # Run the command
    print(f"Running test.py with virtual environment Python: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running test.py: {e}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    sys.exit(run_with_virtual_env())