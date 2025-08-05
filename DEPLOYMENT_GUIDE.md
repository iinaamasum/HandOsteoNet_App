# HandOsteoNet Deployment Guide

## Issue Resolution

### Problem
The application was failing to deploy on Streamlit Cloud with the following error:
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

### Root Cause
The error occurred because `opencv-python` requires OpenGL libraries (`libGL.so.1`) which are not available in the Streamlit Cloud environment.

### Solution
Replace `opencv-python` with `opencv-python-headless` in `requirements.txt`:

```txt
# Before (causing deployment issues)
opencv-python>=4.8.0

# After (fixed for cloud deployment)
opencv-python-headless>=4.8.0
```

## Key Differences

### opencv-python vs opencv-python-headless

| Feature | opencv-python | opencv-python-headless |
|---------|---------------|------------------------|
| GUI Support | ✅ Full GUI support | ❌ No GUI support |
| OpenGL Dependencies | ✅ Requires libGL.so.1 | ❌ No OpenGL dependencies |
| Cloud Deployment | ❌ Fails on cloud platforms | ✅ Works on cloud platforms |
| Image Processing | ✅ All image processing features | ✅ All image processing features |
| Video I/O | ✅ Full video support | ✅ Full video support |

## Functions Used in This Application

The following OpenCV functions are used in this application and are all compatible with `opencv-python-headless`:

1. **Image Reading**: `cv2.imread()` - Used in `Preprocessing/preprocessor.py`
2. **Image Resizing**: `cv2.resize()` - Used in both preprocessing and GradCAM
3. **Contrast Enhancement**: `cv2.createCLAHE()` - Used in preprocessing
4. **Interpolation Methods**: `cv2.INTER_LANCZOS4`, `cv2.INTER_LINEAR` - Used in resizing

## Testing

Run the test script to verify OpenCV functionality:
```bash
python test_opencv.py
```

## Deployment Checklist

Before deploying to Streamlit Cloud:

1. ✅ Use `opencv-python-headless` instead of `opencv-python`
2. ✅ Ensure all model files are included in the repository
3. ✅ Test all OpenCV functions locally
4. ✅ Verify model loading works correctly
5. ✅ Check that all dependencies are in `requirements.txt`

## Common Issues and Solutions

### Issue: Model file not found
**Solution**: Ensure `Model/best_bonenet.pth` is included in the repository

### Issue: Memory errors on deployment
**Solution**: 
- Use CPU-only PyTorch if GPU is not needed
- Optimize model loading to use less memory

### Issue: Import errors for other libraries
**Solution**: Check that all dependencies are listed in `requirements.txt` with appropriate versions

## Streamlit Cloud Configuration

The application is configured for Streamlit Cloud with:
- Main file: `app.py`
- Python version: 3.13.5 (auto-detected)
- Dependencies: Listed in `requirements.txt`

## Monitoring Deployment

After deployment, monitor the logs for:
- Successful model loading
- No OpenCV import errors
- Proper image processing functionality
- User interaction success

## Contact

For deployment issues, contact: masum.cse19@gmail.com 