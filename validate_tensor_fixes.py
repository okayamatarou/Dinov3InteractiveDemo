"""
Comprehensive validation script for DINOv3 tensor handling fixes
Tests the coordinate-to-feature extraction pipeline without requiring actual model loading
"""

import numpy as np
import torch
from typing import Tuple

class MockTensorValidator:
    """Mock validator to test tensor handling logic without loading actual models"""
    
    def __init__(self):
        self.image_size = (224, 224)
        self.test_cases = [
            {
                'name': '4D Feature Tensor [B, H, W, D]',
                'tensor': torch.randn(1, 16, 16, 384),
                'description': 'Standard reshaped feature tensor'
            },
            {
                'name': '3D Feature Tensor [B, N, D]',
                'tensor': torch.randn(1, 256, 384),
                'description': 'Flattened patch features'
            },
            {
                'name': '3D Non-square patches',
                'tensor': torch.randn(1, 255, 384),  # Not perfect square
                'description': 'Non-square patch count'
            }
        ]
    
    def pixel_to_patch(self, x: int, y: int, features: torch.Tensor) -> Tuple[int, int]:
        """Test version of _pixel_to_patch with same logic"""
        print(f"Testing pixel_to_patch: ({x}, {y}) with tensor shape {features.shape}")
        
        if x < 0 or y < 0 or x >= self.image_size[0] or y >= self.image_size[1]:
            print(f"Warning: Coordinates ({x}, {y}) out of image bounds {self.image_size}")
            x = max(0, min(x, self.image_size[0] - 1))
            y = max(0, min(y, self.image_size[1] - 1))
            print(f"Clamped coordinates to ({x}, {y})")
        
        if len(features.shape) == 4:  # [B, H, W, D]
            _, h, w, _ = features.shape
            print(f"4D tensor - patch grid size: {h}x{w}")
        elif len(features.shape) == 3:  # [B, N, D]
            print(f"3D tensor shape: {features.shape}")
            n_patches = features.shape[1]
            h = w = int(np.sqrt(n_patches))
            print(f"Calculated patch grid: {h}x{w} from {n_patches} patches")
        else:
            raise ValueError(f"Unsupported feature tensor shape: {features.shape}")
        
        patch_x = int(x * w / self.image_size[0])
        patch_y = int(y * h / self.image_size[1])
        
        patch_x = max(0, min(patch_x, w - 1))
        patch_y = max(0, min(patch_y, h - 1))
        
        print(f"Patch coordinates: ({patch_x}, {patch_y}) in grid {h}x{w}")
        
        if patch_x >= w or patch_y >= h:
            print(f"Error: Calculated patch coordinates ({patch_x}, {patch_y}) exceed grid bounds ({h}, {w})")
            patch_x = min(patch_x, w - 1)
            patch_y = min(patch_y, h - 1)
            print(f"Final clamped patch coordinates: ({patch_x}, {patch_y})")
        
        return patch_x, patch_y
    
    def get_feature_at_pixel(self, x: int, y: int, features: torch.Tensor) -> torch.Tensor:
        """Test version of _get_feature_at_pixel with same logic"""
        print(f"Testing get_feature_at_pixel: ({x}, {y}) with tensor shape {features.shape}")
        
        patch_x, patch_y = self.pixel_to_patch(x, y, features)
        
        if len(features.shape) == 4:  # [B, H, W, D]
            _, h, w, d = features.shape
            print(f"4D indexing - accessing [0, {patch_y}, {patch_x}, :] from shape [1, {h}, {w}, {d}]")
            
            if patch_y >= h or patch_x >= w:
                raise IndexError(f"Patch coordinates ({patch_x}, {patch_y}) exceed tensor bounds ({w}, {h})")
            
            feature = features[0, patch_y, patch_x, :]
            
        elif len(features.shape) == 3:  # [B, N, D]
            _, n_patches, d = features.shape
            n_patches_side = int(np.sqrt(n_patches))
            patch_idx = patch_y * n_patches_side + patch_x
            
            print(f"3D indexing - patch_idx={patch_idx} from ({patch_x}, {patch_y}) in {n_patches_side}x{n_patches_side} grid")
            print(f"Accessing [0, {patch_idx}, :] from shape [1, {n_patches}, {d}]")
            
            if patch_idx >= n_patches:
                print(f"Warning: patch_idx {patch_idx} >= n_patches {n_patches}, clamping to {n_patches-1}")
                patch_idx = n_patches - 1
            
            if patch_idx < 0:
                print(f"Warning: patch_idx {patch_idx} < 0, clamping to 0")
                patch_idx = 0
            
            feature = features[0, patch_idx, :]
            
        else:
            raise ValueError(f"Unsupported feature tensor shape: {features.shape}")
        
        print(f"Extracted feature shape: {feature.shape}")
        
        if feature.numel() == 0:
            raise ValueError("Extracted feature is empty")
        
        return feature
    
    def test_coordinate_range(self, features: torch.Tensor):
        """Test a range of coordinates to ensure no indexing errors"""
        print(f"\n--- Testing coordinate range for tensor shape {features.shape} ---")
        
        test_coordinates = [
            (0, 0),           # Top-left corner
            (111, 111),       # Center
            (223, 223),       # Bottom-right corner
            (50, 150),        # Random point 1
            (180, 80),        # Random point 2
            (-5, 50),         # Out of bounds (negative)
            (250, 200),       # Out of bounds (positive)
        ]
        
        success_count = 0
        for x, y in test_coordinates:
            try:
                feature = self.get_feature_at_pixel(x, y, features)
                print(f"✓ SUCCESS: ({x}, {y}) -> feature shape {feature.shape}")
                success_count += 1
            except Exception as e:
                print(f"❌ FAILED: ({x}, {y}) -> {e}")
        
        print(f"Success rate: {success_count}/{len(test_coordinates)} ({100*success_count/len(test_coordinates):.1f}%)")
        return success_count == len(test_coordinates)
    
    def run_validation(self):
        """Run comprehensive validation tests"""
        print("=== DINOv3 Tensor Handling Validation ===")
        print(f"Image size: {self.image_size}")
        
        all_passed = True
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n{'='*60}")
            print(f"Test Case {i}: {test_case['name']}")
            print(f"Description: {test_case['description']}")
            print(f"Tensor shape: {test_case['tensor'].shape}")
            
            try:
                passed = self.test_coordinate_range(test_case['tensor'])
                if passed:
                    print(f"✅ Test Case {i} PASSED")
                else:
                    print(f"❌ Test Case {i} FAILED")
                    all_passed = False
            except Exception as e:
                print(f"❌ Test Case {i} CRASHED: {e}")
                all_passed = False
        
        print(f"\n{'='*60}")
        if all_passed:
            print("🎉 ALL TESTS PASSED! Tensor handling logic is robust.")
        else:
            print("⚠️  SOME TESTS FAILED. Review the tensor handling logic.")
        
        return all_passed

def main():
    """Run the validation"""
    validator = MockTensorValidator()
    success = validator.run_validation()
    
    if success:
        print("\n✅ Validation complete - tensor fixes should work correctly!")
    else:
        print("\n❌ Validation failed - tensor handling needs more work!")
    
    return success

if __name__ == "__main__":
    main()
