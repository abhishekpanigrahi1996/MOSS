"""
Script to:
1. Check for syntax errors in the evaluation_refactored_plotly_new.ipynb notebook.
2. Validate imports from src modules.
3. Ensure code follows refactoring guidelines.
"""
import json
import ast
import sys
import re
import os
from pathlib import Path

# Add necessary paths
sys.path.insert(0, '/mount/Storage/gmm-v4')
sys.path.insert(0, '/mount/Storage/gmm-v4/scripts/evaluation')

# Path to the notebook
notebook_path = '/mount/Storage/gmm-v4/scripts/evaluation/evaluation_refactored_plotly_new.ipynb'

def check_syntax(code):
    """Check for syntax errors in the given code string."""
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        return False, str(e)

def check_imports_and_references(code_cells):
    """Check if imports and references to src modules are correctly defined."""
    # Join all code cells into a single string for analysis
    all_code = '\n'.join([''.join(cell) for cell in code_cells])
    
    # Check for imports from src modules
    src_imports = re.findall(r'from\s+src\.(\w+)\s+import', all_code)
    src_modules = set(src_imports)
    
    print(f"\nFound imports from the following src modules: {', '.join(src_modules)}")
    
    # Check if these modules exist
    for module in src_modules:
        module_path = f'/mount/Storage/gmm-v4/scripts/evaluation/src/{module}.py'
        if os.path.exists(module_path):
            print(f"✓ Module {module}.py exists")
        else:
            print(f"✗ Module {module}.py does not exist!")
    
    # Check for function references
    for module in src_modules:
        pattern = fr'{module}\.\w+\('
        functions = re.findall(pattern, all_code)
        if functions:
            print(f"✓ Found {len(functions)} function calls to {module}")
        else:
            print(f"? Module {module} is imported but might not be used")
    
    return src_modules

def check_code_cells_length(code_cells):
    """Check if code cells follow the length guidelines."""
    long_cells = []
    for i, code in enumerate(code_cells):
        lines = code.count('\n') + 1
        if lines > 15:  # More than 15 lines
            long_cells.append((i, lines))
    
    if long_cells:
        print("\nFound code cells longer than recommended (15 lines):")
        for cell_idx, line_count in long_cells:
            print(f"- Cell {cell_idx+1}: {line_count} lines")
        print("Consider moving these to src/*.py files")
    else:
        print("\n✓ All code cells follow the length guideline (<= 15 lines)")
    
    return long_cells

def check_dataframe_usage(code_cells):
    """Check if code uses DataFrames for data aggregation and plotting."""
    all_code = '\n'.join([''.join(cell) for cell in code_cells])
    
    # Check for DataFrame creation and usage with plotting
    df_creation = re.findall(r'(pd\.DataFrame|DataFrame)\(', all_code)
    plot_calls = re.findall(r'plot_\w+\(\s*(\w+)', all_code)
    
    print(f"\nFound {len(df_creation)} DataFrame creation calls")
    print(f"Found {len(plot_calls)} plotting function calls")
    
    # Check if plotting function parameters look like DataFrames
    df_names = re.findall(r'(\w+)_df', all_code)
    df_params = [param for param in plot_calls if param.endswith('df') or param in df_names]
    
    if df_params:
        print(f"✓ Found {len(df_params)} plot calls that use DataFrame parameters")
    else:
        print("? No clear evidence of plotting functions using DataFrames")

def check_plotting_style(code_cells):
    """Check if the notebook uses a consistent plotting style."""
    all_code = '\n'.join([''.join(cell) for cell in code_cells])
    
    # Check for template usage
    template_setting = re.search(r'set_gmm_template\(\)', all_code)
    if template_setting:
        print("\n✓ Using set_gmm_template() for consistent styling")
    else:
        print("\n? No evidence of using set_gmm_template() for styling")
    
    # Check for direct plotly calls vs. helper functions
    direct_calls = re.findall(r'(?:go\.Figure|px\.\w+)\(', all_code)
    helper_calls = re.findall(r'plot_\w+\(', all_code)
    
    if direct_calls and not helper_calls:
        print("✗ Using direct Plotly calls without helper functions")
    elif helper_calls and not direct_calls:
        print("✓ Using only helper functions for plotting")
    else:
        print(f"? Mixed usage: {len(helper_calls)} helper calls and {len(direct_calls)} direct Plotly calls")

# Load the notebook
print(f"Checking notebook: {notebook_path}")
try:
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
except FileNotFoundError:
    print(f"Error: Notebook file not found at {notebook_path}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Notebook file is not valid JSON")
    sys.exit(1)

# Extract code cells
code_cells = []
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        code_cells.append(''.join(cell['source']))

# Check each code cell for syntax errors
print(f"Checking {len(code_cells)} code cells for syntax errors...")
all_cells_valid = True

for i, code in enumerate(code_cells):
    # Skip empty cells
    if not code.strip():
        print(f"Cell {i+1}: Empty - Skipped")
        continue
    
    # Check syntax
    result = check_syntax(code)
    if result is True:
        print(f"Cell {i+1}: Syntax OK")
    else:
        print(f"Cell {i+1}: Syntax Error - {result[1]}")
        all_cells_valid = False

if all_cells_valid:
    print("\n✓ All code cells have valid syntax!")
else:
    print("\n✗ Some cells have syntax errors! See details above.")

# Check imports and references
src_modules = check_imports_and_references(code_cells)

# Check code cell length
long_cells = check_code_cells_length(code_cells)

# Check DataFrame usage
check_dataframe_usage(code_cells)

# Check plotting style
check_plotting_style(code_cells)

# Overall assessment
print("\n" + "="*50)
print("REFACTORING ASSESSMENT:")
if all_cells_valid and src_modules and not long_cells:
    print("✓ The notebook follows the refactoring guidelines well!")
else:
    print("⚠ The notebook needs some adjustments to fully meet the guidelines.")
    
    if not all_cells_valid:
        print("  - Fix syntax errors in cells")
    if not src_modules:
        print("  - Move helper code to src/*.py modules")
    if long_cells:
        print("  - Reduce length of long code cells")
print("="*50)