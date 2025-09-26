"""
Local test script to validate tensor handling logic in the DINOv3 demo
"""

import torch
import numpy as np
from PIL import Image
import io

def test_tensor_shapes():
    """Test different tensor shape scenarios that might occur"""
    print("=== Testing Tensor Shape Handling ===")
    
    test_cases = [
        {
            'name': 'DINOv2 with CLS token',
            'shape': (1, 257, 384),  # 16x16 patches + 1 CLS token
            'description': '[B, N_tokens, D] where first token is CLS'
        },
        {
            'name': 'Feature map format',
            'shape': (1, 384, 16, 16),
            'description': '[B, C, H, W] feature maps'
        },
        {
            'name': 'Processed patches',
            'shape': (1, 256, 384),  # 16x16 patches, no CLS
            'description': '[B, N_patches, D] without CLS token'
        }
    ]
    
    for case in test_cases:
        print(f"\n--- Testing {case['name']} ---")
        print(f"Input shape: {case['shape']}")
        print(f"Description: {case['description']}")
        
        features = torch.randn(case['shape'])
        
        try:
            processed_features = process_features(features)
            print(f"✓ Processed shape: {processed_features.shape}")
            
            test_pixel_coords = [(50, 50), (100, 150), (200, 200)]
            for x, y in test_pixel_coords:
                try:
                    patch_x, patch_y = pixel_to_patch_test(x, y, processed_features, (224, 224))
                    feature = get_feature_at_pixel_test(x, y, processed_features, (224, 224))
                    print(f"  ✓ Pixel ({x}, {y}) -> Patch ({patch_x}, {patch_y}) -> Feature shape: {feature.shape}")
                except Exception as e:
                    print(f"  ❌ Pixel ({x}, {y}) failed: {e}")
                    
        except Exception as e:
            print(f"❌ Processing failed: {e}")

def process_features(features):
    """Simulate the _extract_features processing logic"""
    if len(features.shape) == 3 and features.shape[1] > 1:
        if features.shape[1] == 257:  # 16x16 + 1 CLS
            patch_features = features[:, 1:]  # Remove CLS token
        else:
            patch_features = features
    elif len(features.shape) == 4:  # [B, C, H, W]
        patch_features = features.permute(0, 2, 3, 1)  # [B, H, W, C]
        return patch_features
    else:
        patch_features = features
    
    n_patches = patch_features.shape[1]
    n_patches_side = int(np.sqrt(n_patches))
    
    if n_patches_side * n_patches_side != n_patches:
        print(f"Warning: Non-square patch grid. n_patches={n_patches}")
        n_patches_side = int(np.sqrt(n_patches))
        actual_patches = n_patches_side * n_patches_side
        patch_features = patch_features[:, :actual_patches, :]
    
    patch_features = patch_features.reshape(1, n_patches_side, n_patches_side, -1)
    return patch_features

def pixel_to_patch_test(x, y, features, image_size):
    """Test pixel to patch conversion"""
    if len(features.shape) == 4:  # [B, H, W, D]
        _, h, w, _ = features.shape
    elif len(features.shape) == 3:  # [B, N, D]
        n_patches = features.shape[1]
        h = w = int(np.sqrt(n_patches))
    else:
        raise ValueError(f"Unsupported feature tensor shape: {features.shape}")
    
    patch_x = int(x * w / image_size[0])
    patch_y = int(y * h / image_size[1])
    
    patch_x = max(0, min(patch_x, w - 1))
    patch_y = max(0, min(patch_y, h - 1))
    
    return patch_x, patch_y

def get_feature_at_pixel_test(x, y, features, image_size):
    """Test feature extraction at pixel"""
    patch_x, patch_y = pixel_to_patch_test(x, y, features, image_size)
    
    if len(features.shape) == 4:  # [B, H, W, D]
        feature = features[0, patch_y, patch_x, :]
    elif len(features.shape) == 3:  # [B, N, D]
        n_patches_side = int(np.sqrt(features.shape[1]))
        patch_idx = patch_y * n_patches_side + patch_x
        if patch_idx >= features.shape[1]:
            patch_idx = features.shape[1] - 1
        feature = features[0, patch_idx, :]
    else:
        raise ValueError(f"Unsupported feature tensor shape: {features.shape}")
    
    return feature

if __name__ == "__main__":
    test_tensor_shapes()
    print("\n=== Test Complete ===")
