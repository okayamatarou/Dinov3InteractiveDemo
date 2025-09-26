import ast
import sys

def validate_syntax():
    """Validate Python syntax"""
    try:
        with open('dinov3_interactive_demo.py', 'r') as f:
            code = f.read()
        ast.parse(code)
        print('✓ Syntax check passed')
        return True
    except SyntaxError as e:
        print(f'✗ Syntax error: {e}')
        return False

def check_imports():
    """Check if core dependencies are available"""
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        from PIL import Image
        print('✓ Core dependencies available')
        return True
    except ImportError as e:
        print(f'✗ Import error: {e}')
        return False

if __name__ == "__main__":
    print("Validating DINOv3 demo code...")
    
    syntax_ok = validate_syntax()
    imports_ok = check_imports()
    
    if syntax_ok and imports_ok:
        print('✓ Basic validation passed')
        sys.exit(0)
    else:
        print('✗ Validation failed')
        sys.exit(1)
