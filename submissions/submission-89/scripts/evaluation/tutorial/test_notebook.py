#!/usr/bin/env python
"""
Test script to execute the GMM evaluation tutorial notebook and check for errors.
"""

import nbformat
from nbclient import NotebookClient
import sys
from pathlib import Path

def execute_notebook(notebook_path, timeout=600):
    """Execute a Jupyter notebook and return any errors."""
    print(f"Executing notebook: {notebook_path}")
    print("=" * 70)
    
    # Load the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Create a notebook client
    client = NotebookClient(
        nb, 
        timeout=timeout,
        kernel_name='python3',
        allow_errors=True  # Continue execution even if cells have errors
    )
    
    # Execute the notebook
    try:
        client.execute()
        print("✓ Notebook executed successfully!")
    except Exception as e:
        print(f"✗ Error during notebook execution: {e}")
        return False
    
    # Check for errors in cells
    errors = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            if hasattr(cell, 'outputs'):
                for output in cell.outputs:
                    if output.output_type == 'error':
                        errors.append({
                            'cell': i + 1,
                            'error_type': output.ename,
                            'error_value': output.evalue,
                            'traceback': output.traceback
                        })
    
    # Report results
    if errors:
        print(f"\n✗ Found {len(errors)} errors during execution:")
        print("-" * 70)
        for error in errors:
            print(f"\nError in cell {error['cell']}:")
            print(f"  Type: {error['error_type']}")
            print(f"  Message: {error['error_value']}")
            if len(error['traceback']) > 0:
                print("  Traceback (last line):")
                print(f"    {error['traceback'][-1].strip()}")
        return False
    else:
        print("\n✓ No errors found in notebook execution!")
        return True

if __name__ == "__main__":
    notebook_path = Path(__file__).parent / "gmm_evaluation_tutorial_new.ipynb"
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        sys.exit(1)
    
    success = execute_notebook(notebook_path)
    sys.exit(0 if success else 1)