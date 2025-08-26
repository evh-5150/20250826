#!/usr/bin/env python3
"""
Debug script to check normalization/denormalization and intermediate values.
"""

import torch
import numpy as np
import pydicom
from utils import load_dicom_image, save_16bit_dicom_image
from diffusion_model import SimpleUnet
import torch.nn.functional as F
import matplotlib.pyplot as plt


def debug_inference(model_path: str, input_path: str, output_dir: str = "debug_output"):
    """Debug inference process step by step."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load original image
    print("Loading original image...")
    i_lr, original_range, original_dicom = load_dicom_image(input_path, device)
    print(f"Original range: {original_range}")
    print(f"Original shape: {i_lr.shape}")
    print(f"Original min/max: {i_lr.min().item():.3f}/{i_lr.max().item():.3f}")
    
    # Save original normalized
    orig_norm = ((i_lr + 1.0) / 2.0).squeeze().cpu().numpy()
    plt.imsave(f"{output_dir}/01_original_normalized.png", orig_norm, cmap='gray')
    print(f"Normalized original saved to {output_dir}/01_original_normalized.png")
    
    # Load model
    print("Loading model...")
    model = SimpleUnet(dropout_rate=0.1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Simulate inference (simplified)
    print("Running simplified inference...")
    with torch.no_grad():
        # Create a simple test: just run one forward pass
        test_input = torch.randn(1, 1, 256, 256, device=device)
        test_t = torch.tensor([50], device=device)
        test_condition = F.interpolate(i_lr, size=(256, 256), mode='nearest-exact')
        
        output = model(test_input, test_t, test_condition)
        print(f"Model output shape: {output.shape}")
        print(f"Model output range: {output.min().item():.3f} to {output.max().item():.3f}")
        
        # Save test output
        test_out = output.squeeze().cpu().numpy()
        plt.imsave(f"{output_dir}/02_model_output.png", test_out, cmap='gray')
        print(f"Model output saved to {output_dir}/02_model_output.png")
    
    # Check denormalization process
    print("Testing denormalization...")
    min_val, max_val = original_range
    print(f"Denorm params: min_val={min_val}, max_val={max_val}")
    
    # Test with a simple tensor
    test_tensor = torch.tensor([[-1.0, 0.0, 1.0]], device=device)
    denorm = ((test_tensor + 1.0) / 2.0) * (max_val - min_val) + min_val
    print(f"Test denorm: {test_tensor} -> {denorm}")
    
    # Test with actual output
    denorm_output = ((output + 1.0) / 2.0) * (max_val - min_val) + min_val
    print(f"Denorm output range: {denorm_output.min().item():.1f} to {denorm_output.max().item():.1f}")
    
    # Save denormalized
    denorm_np = denorm_output.squeeze().cpu().numpy()
    plt.imsave(f"{output_dir}/03_denormalized.png", denorm_np, cmap='gray')
    print(f"Denormalized output saved to {output_dir}/03_denormalized.png")
    
    # Check uint16 conversion
    uint16_output = np.clip(denorm_np, 0, 65535).astype(np.uint16)
    print(f"uint16 range: {uint16_output.min()} to {uint16_output.max()}")
    print(f"uint16 unique values: {len(np.unique(uint16_output))}")
    
    # Save uint16
    plt.imsave(f"{output_dir}/04_uint16.png", uint16_output, cmap='gray')
    print(f"uint16 output saved to {output_dir}/04_uint16.png")
    
    print(f"Debug outputs saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python debug_inference.py <model_path> <input_dicom>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_path = sys.argv[2]
    debug_inference(model_path, input_path)