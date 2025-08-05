import torch
import numpy as np
import math

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
    Load the trained model
    """
    from Model.model import BoneAgeFullModel
    
    model = BoneAgeFullModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def get_target_layer(model):
    """
    Get the target layer for GradCAM (last conv layer in stage4)
    """
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