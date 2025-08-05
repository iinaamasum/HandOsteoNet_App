import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, gender_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor, gender_tensor)

        # Backward pass for regression output
        output.sum().backward()  # Sum over batch for batched inputs

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam_map = torch.sum(weights * self.activations, dim=1)
        grad_cam_map = F.relu(grad_cam_map)

        # Normalize GradCAM map per image in the batch
        grad_cam_map_min = grad_cam_map.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        grad_cam_map_max = grad_cam_map.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        grad_cam_map = (grad_cam_map - grad_cam_map_min) / (grad_cam_map_max - grad_cam_map_min + 1e-8)

        return grad_cam_map.cpu().numpy()

def inv_normalize():
    return transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

def save_gradcam_image(img_tensor, cam, filename):
    try:
        # Inverse normalize the image
        inv_norm = inv_normalize()
        img_tensor = inv_norm(img_tensor)
        img_np = img_tensor.detach().permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        # Extract original grayscale channel (channel 0)
        orig_img = img_np[:, :, 0]  # Shape: (H, W)

        # Resize CAM to match image dimensions (480x480)
        cam_resized = cv2.resize(cam, (480, 480), interpolation=cv2.INTER_LINEAR)
        cam_resized = np.clip(cam_resized, 0, 1)

        # Create heatmap overlay
        heatmap = plt.get_cmap('jet')(cam_resized)[:, :, :3]  # Shape: (H, W, 3)
        overlayed = np.clip(heatmap * 0.4 + img_np * 0.6, 0, 1)

        # Convert to RGB and save only the overlay (not merged)
        overlayed_rgb = (overlayed * 255).astype(np.uint8)
        Image.fromarray(overlayed_rgb).save(filename)
        return True
    except Exception as e:
        print(f"Error in save_gradcam_image: {str(e)}")
        return False

def generate_gradcam(model, img_tensor, gender_tensor, target_layer):
    """
    Generate GradCAM for a single image
    """
    try:
        gradcam = GradCAM(model, target_layer)
        
        # Add batch dimension if needed
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if gender_tensor.dim() == 1:
            gender_tensor = gender_tensor.unsqueeze(0)
        
        # Ensure tensors are on the same device as model
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        gender_tensor = gender_tensor.to(device)
        
        # Enable gradient computation
        img_tensor.requires_grad_(True)
        
        # Generate GradCAM
        cam = gradcam(img_tensor, gender_tensor)
        
        # Clean up
        gradcam.remove_hooks()
        
        return cam[0]  # Return single image CAM
    except Exception as e:
        print(f"Error in generate_gradcam: {str(e)}")
        # Return a default CAM if there's an error
        return np.zeros((480, 480)) 