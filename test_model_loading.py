#!/usr/bin/env python3
"""
Simple test script to verify model loading
"""

import os
import sys
import torch

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_model_loading():
    """Test model loading"""
    print("Testing model loading...")

    try:
        from Model.model import BoneAgeFullModel
        from Utils.utils import load_model

        # Check if model file exists
        model_path = "Model/best_bonenet.pth"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False

        print(f"✅ Model file found: {model_path}")

        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Device: {device}")

        # Load model
        model = load_model(model_path, device)

        if model is None:
            print("❌ Model is None after loading")
            return False

        print("✅ Model loaded successfully")
        print(f"   Model type: {type(model)}")
        print(f"   Model device: {next(model.parameters()).device}")

        # Test a simple forward pass
        try:
            # Create dummy input
            dummy_image = torch.randn(1, 3, 480, 480).to(device)
            dummy_gender = torch.tensor([1.0]).to(device)  # Male

            with torch.no_grad():
                output = model(dummy_image, dummy_gender)

            print(f"✅ Forward pass successful")
            print(f"   Output shape: {output.shape}")
            print(f"   Output value: {output.item():.2f}")

            # Test with female gender
            dummy_gender_female = torch.tensor([0.0]).to(device)  # Female
            output_female = model(dummy_image, dummy_gender_female)
            print(f"   Female output: {output_female.item():.2f}")

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def main():
    """Run the test"""
    print("HandOsteoNet Model Loading Test")
    print("=" * 40)

    if test_model_loading():
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
