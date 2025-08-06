#!/usr/bin/env python3
"""
Test script for HandOsteoNet deployment
This script tests the key components to ensure they work in deployment environment
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from Model.model import BoneAgeFullModel
        print("✅ Model imported successfully")
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    try:
        from Preprocessing.preprocessor import ImagePreprocessor, DataManager
        print("✅ Preprocessor imported successfully")
    except ImportError as e:
        print(f"❌ Preprocessor import failed: {e}")
        return False
    
    return True

def test_device_detection():
    """Test device detection"""
    print("\nTesting device detection...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Device detected: {device}")
        return True
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print("\nTesting model loading...")
    
    try:
        from Model.model import BoneAgeFullModel
        from Utils.utils import load_model
        
        model_path = "Model/best_bonenet.pth"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_path, device)
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_preprocessor():
    """Test preprocessor functionality"""
    print("\nTesting preprocessor...")
    
    try:
        from Preprocessing.preprocessor import ImagePreprocessor
        
        preprocessor = ImagePreprocessor()
        print("✅ Preprocessor initialized successfully")
        
        # Test with a dummy image
        dummy_image = np.random.randint(0, 255, (480, 480), dtype=np.uint8)
        dummy_pil = Image.fromarray(dummy_image)
        
        # Save temporarily
        temp_path = "temp_test.png"
        dummy_pil.save(temp_path)
        
        try:
            # Test preprocessing
            tensor = preprocessor.preprocess_image(temp_path)
            print("✅ Image preprocessing successful")
            print(f"   Tensor shape: {tensor.shape}")
            return True
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"❌ Preprocessor test failed: {e}")
        return False

def test_data_manager():
    """Test data manager functionality"""
    print("\nTesting data manager...")
    
    try:
        from Preprocessing.preprocessor import DataManager
        
        data_manager = DataManager()
        print("✅ Data manager initialized successfully")
        
        # Test ID generation
        next_id = data_manager.get_next_id()
        print(f"   Generated ID: {next_id}")
        
        # Test cloud deployment detection
        is_cloud = data_manager.is_cloud_deployment
        print(f"   Cloud deployment: {is_cloud}")
        
        return True
    except Exception as e:
        print(f"❌ Data manager test failed: {e}")
        return False

def test_session_state():
    """Test session state initialization"""
    print("\nTesting session state...")
    
    try:
        # Simulate session state initialization
        session_state = {}
        
        # Initialize session state
        if "model_loaded" not in session_state:
            session_state["model_loaded"] = False
        if "model" not in session_state:
            session_state["model"] = None
        if "device" not in session_state:
            session_state["device"] = None
        if "current_page" not in session_state:
            session_state["current_page"] = "Home"
        if "preprocessor" not in session_state:
            session_state["preprocessor"] = None
        if "data_manager" not in session_state:
            session_state["data_manager"] = None
        
        print("✅ Session state initialized successfully")
        print(f"   Current page: {session_state['current_page']}")
        return True
    except Exception as e:
        print(f"❌ Session state test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("HandOsteoNet Deployment Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_device_detection,
        test_model_loading,
        test_preprocessor,
        test_data_manager,
        test_session_state,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Deployment should work correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 