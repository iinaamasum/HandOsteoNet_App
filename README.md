# HandOsteoNet - Advanced Bone Age Assessment System

## Overview

HandOsteoNet is a professional medical AI application for bone age assessment from hand x-ray images. Developed by the Qatar University Research Team led by Amith Khandakar, this system provides accurate bone age predictions with explainable AI visualization using GradCAM.

## Features

### 🔬 Model Evaluation
- Upload hand x-ray images (PNG, JPG, JPEG)
- Select patient gender (Male/Female)
- Get bone age predictions in months and years
- View GradCAM analysis for model interpretability
- Professional medical AI interface

### 📊 Testing & Save HMC Data
- Upload x-ray images with patient information
- Provide actual bone age for validation
- Calculate evaluation metrics (error, deviation, percent error)
- Save data to HMC database with unique IDs
- Generate comprehensive analysis reports

## Architecture

```
HandOsteoNet/
├── Model/                 # Neural network components
│   ├── attention.py      # Attention mechanisms (CBAM, Self-Attention)
│   ├── seg.py           # Segmentation model
│   ├── bone_age_model.py # Main bone age prediction model
│   └── model.py         # Integrated full model
├── GradCam/             # GradCAM implementation
│   └── gradcam.py       # GradCAM generation and visualization
├── Preprocessing/        # Image preprocessing
│   └── preprocessor.py  # Image preprocessing and data management
├── Utils/               # Utility functions
│   └── utils.py         # Helper functions and metrics
├── HMC_data/           # Data storage
│   ├── new_xray/       # Saved x-ray images
│   └── CSV/            # Patient data CSV files
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Model Architecture

The HandOsteoNet model consists of:

1. **Segmentation Model**: RegNet-based encoder-decoder for bone region segmentation
2. **Bone Age Model**: Custom CNN with attention mechanisms for age prediction
3. **Attention Mechanisms**: CBAM (Convolutional Block Attention Module) and Self-Attention
4. **Integration**: Combines segmentation features with original image for final prediction

## Installation

1. **Clone or navigate to the HandOsteoNet directory**
   ```bash
   cd HandOsteoNet
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**
   - The model expects `best_bonenet.pth` in `../QU_Final_Attention_SEG_v2/`
   - This should be the trained model from your notebook

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Application Flow

1. **Model Evaluation Tab**:
   - Upload hand x-ray image
   - Select gender (Male/Female)
   - Click "Evaluate Model"
   - View prediction results and GradCAM analysis

2. **Testing & Save HMC Data Tab**:
   - Upload hand x-ray image
   - Select gender and enter actual bone age
   - Click "Test and Save Data"
   - View prediction, metrics, and save to database

### Data Management

- **New X-rays**: Saved in `HMC_data/new_xray/` with unique IDs (HMC_1, HMC_2, etc.)
- **Patient Data**: Stored in `HMC_data/CSV/hmc_data.csv` with columns:
  - `id`: Unique identifier (HMC_X)
  - `male`: Boolean (True for Male, False for Female)
  - `boneage`: Actual bone age in months (for testing mode)

## Technical Details

### Image Preprocessing
- Resize to 480x480 pixels
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Create 3-channel input: [gray, clahe_4x4, clahe_8x8]
- Normalize using ImageNet statistics

### Model Input/Output
- **Input**: 3-channel image (480x480) + gender (0/1)
- **Output**: Bone age in months
- **Segmentation**: 1-channel mask (480x480)

### GradCAM Visualization
- Target layer: Last convolutional layer in segmentation model
- Generates heatmap overlay on original image
- Shows model attention regions

## Performance Metrics

The model achieves:
- **MAE**: ~4.95 months
- **RMSE**: ~6.67 months
- **MAPE**: ~5.05%
- **R²**: ~0.974

## Clinical Integration

This application is designed for clinical use with:
- Professional medical AI interface
- Comprehensive error handling
- Data validation and sanitization
- Audit trail with unique patient IDs
- Explainable AI with GradCAM visualization

## Development Team

- **Lead Researcher**: Amith Khandakar
- **Institution**: Qatar University
- **Application**: HandOsteoNet v1.0

## License

For clinical use and research purposes. © 2024 Qatar University Research Team.

## Support

For technical support or questions about the model architecture, please refer to the original notebook in `../QU_Final_Attention_SEG_v2/qu_final_attention_seg_v2.ipynb`. 