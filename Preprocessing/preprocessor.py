import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import uuid
from datetime import datetime

class ImagePreprocessor:
    def __init__(self):
        # ImageNet normalization
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        # Transform for evaluation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ])
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single x-ray image for model input
        """
        # Read image
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize to 480x480
        gray = cv2.resize(gray, (480, 480), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply CLAHE for contrast enhancement
        clahe_4x4 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        clahe_8x8 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
        
        # Stack channels: [gray, clahe_4x4, clahe_8x8]
        image = np.stack([gray, clahe_4x4, clahe_8x8], axis=-1)
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        return image_tensor
    
    def preprocess_from_upload(self, uploaded_file):
        """
        Preprocess image from Streamlit uploaded file
        """
        # Save uploaded file temporarily
        temp_path = f"temp_{uuid.uuid4()}.png"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Preprocess the image
            image_tensor = self.preprocess_image(temp_path)
            return image_tensor
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

class DataManager:
    def __init__(self, hmc_data_dir="HMC_data"):
        self.hmc_data_dir = hmc_data_dir
        self.new_xray_dir = os.path.join(hmc_data_dir, "new_xray")
        self.csv_dir = os.path.join(hmc_data_dir, "CSV")
        self.csv_path = os.path.join(self.csv_dir, "hmc_data.csv")
        
        # Create directories if they don't exist
        os.makedirs(self.new_xray_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=['id', 'male', 'boneage'])
            df.to_csv(self.csv_path, index=False)
    
    def get_next_id(self):
        """Get the next available ID for new x-ray"""
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            if len(df) > 0:
                # Extract numeric part from existing IDs and find max
                existing_ids = df['id'].tolist()
                numeric_ids = []
                for id_str in existing_ids:
                    if id_str.startswith('HMC_'):
                        try:
                            numeric_id = int(id_str.split('_')[1])
                            numeric_ids.append(numeric_id)
                        except (ValueError, IndexError):
                            continue
                
                if numeric_ids:
                    next_id = max(numeric_ids) + 1
                else:
                    next_id = 1
            else:
                next_id = 1
        else:
            next_id = 1
        
        return f"HMC_{next_id}"
    
    def save_xray_data(self, image_tensor, gender, bone_age=None):
        """
        Save x-ray image and metadata
        gender: True for Male, False for Female
        bone_age: actual bone age in months (optional, for testing mode)
        """
        # Generate unique ID
        xray_id = self.get_next_id()
        
        # Save image
        image_path = os.path.join(self.new_xray_dir, f"{xray_id}.png")
        
        # Convert tensor to PIL Image and save
        inv_norm = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        img_tensor = inv_norm(image_tensor)
        img_np = img_tensor.detach().permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        
        # Use the grayscale channel (first channel)
        gray_img = (img_np[:, :, 0] * 255).astype(np.uint8)
        Image.fromarray(gray_img).save(image_path)
        
        # Update CSV
        df = pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame(columns=['id', 'male', 'boneage'])
        
        new_row = {
            'id': xray_id,
            'male': gender,  # True for Male, False for Female
            'boneage': bone_age if bone_age is not None else None
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.csv_path, index=False)
        
        return xray_id, image_path
    
    def get_all_data(self):
        """Get all saved data"""
        if os.path.exists(self.csv_path):
            return pd.read_csv(self.csv_path)
        return pd.DataFrame(columns=['id', 'male', 'boneage']) 