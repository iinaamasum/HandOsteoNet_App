#!/usr/bin/env python3
"""
Test script to verify that opencv-python-headless works with the existing code
"""

import cv2
import numpy as np
from PIL import Image
import os

def test_opencv_functions():
    """Test the OpenCV functions used in the application"""
    print("Testing OpenCV functions...")
    
    # Test 1: Basic OpenCV import
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test 2: Create a test image
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # Test 3: Test cv2.resize with INTER_LANCZOS4
    try:
        resized = cv2.resize(test_image, (200, 200), interpolation=cv2.INTER_LANCZOS4)
        print("✓ cv2.resize with INTER_LANCZOS4 works")
    except Exception as e:
        print(f"✗ cv2.resize with INTER_LANCZOS4 failed: {e}")
    
    # Test 4: Test cv2.resize with INTER_LINEAR
    try:
        resized = cv2.resize(test_image, (200, 200), interpolation=cv2.INTER_LINEAR)
        print("✓ cv2.resize with INTER_LINEAR works")
    except Exception as e:
        print(f"✗ cv2.resize with INTER_LINEAR failed: {e}")
    
    # Test 5: Test cv2.createCLAHE
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(test_image)
        print("✓ cv2.createCLAHE works")
    except Exception as e:
        print(f"✗ cv2.createCLAHE failed: {e}")
    
    # Test 6: Test cv2.imread (create a temporary file)
    try:
        # Create a temporary image file
        temp_path = "temp_test_image.png"
        Image.fromarray(test_image).save(temp_path)
        
        # Test reading with cv2.imread
        gray = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            print("✓ cv2.imread works")
        else:
            print("✗ cv2.imread failed")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"✗ cv2.imread test failed: {e}")
    
    print("OpenCV test completed!")

if __name__ == "__main__":
    test_opencv_functions() 