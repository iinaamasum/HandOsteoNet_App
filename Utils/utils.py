import torch
import numpy as np
import math
import os

def convert_months_to_years_months(months):
    """
    Convert months to years and months format
    """
    years = int(months // 12)
    remaining_months = int(months % 12)
    
    if years == 0:
        return f"{remaining_months} months"
    elif remaining_months == 0:
        return f"{years} years"
    else:
        return f"{years} years, {remaining_months} months"

def calculate_metrics(predicted_age, actual_age):
    """
    Calculate evaluation metrics
    """
    error = abs(predicted_age - actual_age)
    deviation = predicted_age - actual_age
    percent_error = (error / actual_age) * 100 if actual_age > 0 else 0
    
    return {
        'error': error,
        'deviation': deviation,
        'percent_error': percent_error
    }

def load_model(model_path, device):
    """
    Load the trained model with improved error handling
    """
    try:
        from Model.model import BoneAgeFullModel
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        # Create model instance
        model = BoneAgeFullModel()
        
        # Load state dict with proper error handling
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # Move to device and set to eval mode
        model.to(device)
        model.eval()
        
        # Verify model is properly loaded
        if model is None:
            raise RuntimeError("Model failed to initialize")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

def get_target_layer(model):
    """
    Get the target layer for GradCAM (last conv layer in stage4)
    """
    if model is None:
        raise ValueError("Model is None, cannot get target layer")
    return model.seg_model.s4

def format_prediction(predicted_months):
    """
    Format prediction for display
    """
    years_months = convert_months_to_years_months(predicted_months)
    return {
        'months': round(predicted_months, 1),
        'years_months': years_months
    }

def validate_inputs(image, gender, actual_age=None):
    """
    Validate user inputs
    """
    errors = []
    
    if image is None:
        errors.append("⚠️ Please upload an x-ray image")
    
    if gender is None or gender == "":
        errors.append("⚠️ Please select gender")
    
    if actual_age is not None:
        if actual_age is None or actual_age == "":
            errors.append("⚠️ Please enter the actual bone age")
        else:
            try:
                actual_age = float(actual_age)
                if actual_age < 0 or actual_age > 300:  # Reasonable range for bone age
                    errors.append("⚠️ Actual bone age should be between 0 and 300 months")
            except (ValueError, TypeError):
                errors.append("⚠️ Actual bone age should be a valid number")
    
    return errors 