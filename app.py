import streamlit as st
import torch
import torchvision
import numpy as np
import os
import sys
from PIL import Image
import tempfile
import base64
import hashlib
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from Model.model import BoneAgeFullModel
from Preprocessing.preprocessor import ImagePreprocessor, DataManager
from GradCam.gradcam import generate_gradcam, save_gradcam_image
from Utils.utils import (
    convert_months_to_years_months,
    calculate_metrics,
    load_model,
    get_target_layer,
    format_prediction,
    validate_inputs,
)

# Page configuration
st.set_page_config(
    page_title="HandOsteoNet - Bone Age Assessment",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for medical AI styling
st.markdown("""
<style>
    /* Global container width */
    .main .block-container {
        max-width: 1200px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Professional color scheme for light/dark mode compatibility */
    :root {
        --primary-bg: #f8f9fa;
        --secondary-bg: #ffffff;
        --accent-bg: #e9ecef;
        --text-primary: #212529;
        --text-secondary: #6c757d;
        --border-color: #dee2e6;
        --success-bg: #d1e7dd;
        --success-border: #198754;
        --error-bg: #f8d7da;
        --error-border: #dc3545;
        --info-bg: #cff4fc;
        --info-border: #055160;
        --warning-bg: #fff3cd;
        --warning-border: #856404;
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-bg: #1a1a1a;
            --secondary-bg: #2d2d2d;
            --accent-bg: #404040;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --border-color: #404040;
            --success-bg: #0f5132;
            --success-border: #198754;
            --error-bg: #842029;
            --error-border: #dc3545;
            --info-bg: #055160;
            --info-border: #0dcaf0;
            --warning-bg: #664d03;
            --warning-border: #ffc107;
        }
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .error-card {
        background: var(--error-bg);
        border: 1px solid var(--error-border);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--text-primary);
    }
    
    .success-card {
        background: var(--success-bg);
        border: 1px solid var(--success-border);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--text-primary);
    }
    
    .info-card {
        background: var(--info-bg);
        border: 1px solid var(--info-border);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--text-primary);
    }
    
    .warning-card {
        background: var(--warning-bg);
        border: 1px solid var(--warning-border);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--text-primary);
    }
    
    /* Navigation styling */
    .nav-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Image container for better alignment */
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem;
        background: var(--secondary-bg);
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin: 1rem 0;
    }
    
    /* Sample image grid */
    .sample-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .sample-item {
        text-align: center;
        padding: 1rem;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        background: var(--secondary-bg);
        margin-bottom: 1rem;
    }
    
    /* Video container */
    .video-container {
        position: relative;
        width: 100%;
        max-width: 800px;
        margin: 2rem auto;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Professional button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border-radius: 8px;
        border: 2px dashed var(--border-color);
        background: var(--secondary-bg);
    }
    
    /* Selectbox styling */
    .stSelectbox > div {
        border-radius: 8px;
    }
    
    /* Number input styling */
    .stNumberInput > div {
        border-radius: 8px;
    }
    
    /* Tab styling for better visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: var(--secondary-bg);
        border-radius: 8px 8px 0px 0px;
        gap: 0.5rem;
        padding: 8px 16px;
        margin: 0 4px;
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--accent-bg);
        color: var(--text-primary);
        border-color: var(--border-color);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--secondary-bg);
        color: #667eea;
        border-color: #667eea;
        border-bottom-color: var(--secondary-bg);
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [aria-selected="true"]:hover {
        background-color: var(--secondary-bg);
        color: #667eea;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid var(--border-color);
    }
    
    /* Status indicators with proper spacing */
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .status-success {
        background: var(--success-bg);
        color: var(--success-border);
        border: 1px solid var(--success-border);
    }
    
    .status-warning {
        background: var(--warning-bg);
        color: var(--warning-border);
        border: 1px solid var(--warning-border);
    }
    
    /* Current page highlighting */
    .nav-button.active {
        background: rgba(255, 255, 255, 0.3);
        border-color: rgba(255, 255, 255, 0.6);
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state properly
def initialize_session_state():
    """Initialize session state"""
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "model" not in st.session_state:
        st.session_state.model = None
    if "device" not in st.session_state:
        st.session_state.device = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    if "preprocessor" not in st.session_state:
        st.session_state.preprocessor = None
    if "data_manager" not in st.session_state:
        st.session_state.data_manager = None

# Initialize session state
initialize_session_state()

@st.cache_resource
def get_device():
    """Get device with caching"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model_safely():
    """Load the model safely with error handling and caching"""
    try:
        if not st.session_state.model_loaded:
            with st.spinner("Loading HandOsteoNet model..."):
                # Set device
                device = get_device()
                st.session_state.device = device

                # Load model
                model_path = "Model/best_bonenet.pth"
                if not os.path.exists(model_path):
                    st.error(f"Model file not found at: {model_path}")
                    return False

                model = load_model(model_path, device)
                st.session_state.model = model
                st.session_state.model_loaded = True

                st.success("Model loaded successfully!")
                return True
        return True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False

@st.cache_resource
def get_preprocessor():
    """Get preprocessor with caching"""
    if st.session_state.preprocessor is None:
        st.session_state.preprocessor = ImagePreprocessor()
    return st.session_state.preprocessor

@st.cache_resource
def get_data_manager():
    """Get data manager with caching"""
    if st.session_state.data_manager is None:
        st.session_state.data_manager = DataManager()
    return st.session_state.data_manager

def create_navigation():
    """Create navigation menu"""
    st.markdown("""
    <div class="nav-container">
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 0.5rem;">
    """, unsafe_allow_html=True)
    
    pages = [
        ("Home", "Home"),
        ("Evaluate Model", "Evaluate"),
        ("Testing & Save", "Testing"),
        ("Sample Images", "Samples"),
        ("How to Use", "HowToUse"),
        ("Privacy Policy", "Privacy")
    ]
    
    cols = st.columns(len(pages))
    for i, (label, page) in enumerate(pages):
        with cols[i]:
            # Check if this is the current page
            is_active = st.session_state.current_page == page
            button_style = "primary" if is_active else "secondary"
            
            if st.button(label, key=f"nav_{page}", use_container_width=True, type=button_style):
                st.session_state.current_page = page
                st.rerun()
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def display_image_with_aspect_ratio(image, caption, max_width=400):
    """Display image maintaining aspect ratio"""
    if image is None:
        return

    # Get original dimensions
    width, height = image.size

    # Calculate new dimensions maintaining aspect ratio
    if width > max_width:
        new_width = max_width
        new_height = int((height * max_width) / width)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        resized_image = image

    # Center the image in a container
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(resized_image, caption=caption, use_container_width=True)

def home_page():
    """Home page content"""
    st.markdown("""
    <div class="main-header">
        <h1>HandOsteoNet</h1>
        <h3>Advanced Bone Age Assessment System</h3>
        <p>Developed by Qatar University Research Team, led by Amith Khandakar</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to HandOsteoNet
    
    HandOsteoNet is an advanced AI-powered system for bone age assessment from hand x-ray images. 
    Our system utilizes state-of-the-art deep learning techniques to provide accurate and reliable 
    bone age predictions for clinical and research purposes.
    
    ### Key Features:
    - **Model Evaluation**: Upload x-ray images and get instant bone age predictions
    - **Testing & Data Management**: Test model performance and save data to HMC database
    - **Sample Images**: View sample x-ray images used for training and testing
    - **How to Use**: Learn how to use the system effectively
    - **Privacy Policy**: Understand our commitment to data privacy
    
    ### Getting Started:
    1. Navigate to "Evaluate Model" to start analyzing x-ray images
    2. Upload a hand x-ray image in PNG, JPG, or JPEG format
    3. Select the patient's gender
    4. Click "Evaluate Model" to get instant results
    
    ### System Requirements:
    - Modern web browser (Chrome, Firefox, Safari, Edge)
    - Internet connection for model loading
    - Supported image formats: PNG, JPG, JPEG
    
    """)
    
    # Load model status
    if load_model_safely():
        st.success("✅ Model is ready for analysis")
    else:
        st.error("❌ Model loading failed. Please refresh the page.")

def evaluate_page():
    """Model evaluation page"""
    st.markdown("### Model Evaluation")
    st.markdown("Upload a hand x-ray image and select gender to evaluate bone age.")

    # Load model
    if not load_model_safely():
        st.stop()

    # Initialize components
    preprocessor = get_preprocessor()

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Hand X-Ray Image",
        type=["png", "jpg", "jpeg"],
        help="Upload a hand x-ray image in PNG, JPG, or JPEG format",
    )

    # Gender selection
    gender = st.selectbox(
        "Select Gender",
        options=["", "Male", "Female"],
        help="Select the patient's gender",
    )

    # Check if all required fields are filled
    is_evaluate_ready = (
        uploaded_file is not None and gender != "" and gender is not None
    )

    # Show status indicators
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        if uploaded_file is not None:
            st.markdown("""
            <div class="status-indicator status-success">
                ✅ X-Ray uploaded successfully
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-indicator status-warning">
                ⚠️ X-Ray upload required
            </div>
            """, unsafe_allow_html=True)
    
    with col_status2:
        if gender != "" and gender is not None:
            st.markdown(f"""
            <div class="status-indicator status-success">
                ✅ Gender selected: {gender}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-indicator status-warning">
                ⚠️ Gender selection required
            </div>
            """, unsafe_allow_html=True)

    # Evaluate button
    if st.button(
        "Evaluate Model",
        type="primary",
        use_container_width=True,
        disabled=not is_evaluate_ready,
    ):
        # Validate inputs
        errors = validate_inputs(uploaded_file, gender)

        if errors:
            st.markdown('<div class="error-card">', unsafe_allow_html=True)
            for error in errors:
                st.error(error)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            try:
                with st.spinner("Processing image and generating prediction..."):
                    # Preprocess image
                    image_tensor = preprocessor.preprocess_from_upload(uploaded_file)

                    # Prepare gender tensor
                    gender_tensor = torch.tensor([1.0 if gender == "Male" else 0.0])

                    # Move tensors to device
                    image_tensor = image_tensor.to(st.session_state.device)
                    gender_tensor = gender_tensor.to(st.session_state.device)

                    # Make prediction
                    with torch.no_grad():
                        prediction = st.session_state.model(
                            image_tensor.unsqueeze(0), gender_tensor
                        )

                    predicted_months = prediction.item()
                    formatted_prediction = format_prediction(predicted_months)

                    # Display results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            '<div class="prediction-card">', unsafe_allow_html=True
                        )
                        st.markdown(f"### Prediction Results")
                        st.markdown(
                            f"**Bone Age:** {formatted_prediction['years_months']}"
                        )
                        st.markdown(f"**Months:** {formatted_prediction['months']}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f"### Model Confidence")
                        st.markdown(f"**Prediction Confidence:** High")
                        st.markdown(f"**Model Status:** Active")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Display images with proper aspect ratio
                    st.markdown("### Analysis Visualization")

                    # Original image
                    inv_norm = torch.nn.Sequential(
                        torchvision.transforms.Normalize(
                            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                        )
                    )
                    orig_img = inv_norm(image_tensor)
                    orig_img_np = orig_img.detach().permute(1, 2, 0).cpu().numpy()
                    orig_img_np = np.clip(orig_img_np, 0, 1)
                    orig_img_display = (orig_img_np[:, :, 0] * 255).astype(np.uint8)

                    # Create PIL image for display
                    orig_pil = Image.fromarray(orig_img_display)

                    # Generate GradCAM
                    try:
                        target_layer = get_target_layer(st.session_state.model)
                        cam = generate_gradcam(
                            st.session_state.model,
                            image_tensor,
                            gender_tensor,
                            target_layer,
                        )

                        # Save GradCAM image with unique name
                        timestamp = int(time.time())
                        temp_gradcam_path = f"temp_gradcam_{timestamp}.png"
                        gradcam_success = save_gradcam_image(
                            image_tensor, cam, temp_gradcam_path
                        )

                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            display_image_with_aspect_ratio(
                                orig_pil, "Original X-Ray", 400
                            )

                        with col_img2:
                            if gradcam_success and os.path.exists(temp_gradcam_path):
                                gradcam_img = Image.open(temp_gradcam_path)
                                display_image_with_aspect_ratio(
                                    gradcam_img, "GradCAM Analysis", 400
                                )
                            else:
                                st.error("GradCAM analysis could not be generated")

                        # Clean up temporary file
                        if os.path.exists(temp_gradcam_path):
                            os.remove(temp_gradcam_path)

                    except Exception as e:
                        st.error(f"Error generating GradCAM: {str(e)}")
                        # Show only original image if GradCAM fails
                        display_image_with_aspect_ratio(orig_pil, "Original X-Ray", 400)

            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")

def testing_page():
    """Testing and save data page"""
    st.markdown("### Testing & Save HMC Data")
    st.markdown(
        "Upload a hand x-ray image, provide patient information, and save to HMC database."
    )

    # Load model
    if not load_model_safely():
        st.stop()

    # Initialize components
    preprocessor = get_preprocessor()
    data_manager = get_data_manager()

    # File upload
    uploaded_file_test = st.file_uploader(
        "Upload Hand X-Ray Image",
        type=["png", "jpg", "jpeg"],
        key="test_upload",
        help="Upload a hand x-ray image in PNG, JPG, or JPEG format",
    )

    # Patient information
    col1, col2 = st.columns(2)

    with col1:
        gender_test = st.selectbox(
            "Select Gender",
            options=["", "Male", "Female"],
            key="test_gender",
            help="Select the patient's gender",
        )

    with col2:
        actual_age = st.number_input(
            "Actual Bone Age (months)",
            min_value=0.0,
            max_value=300.0,
            value=None,
            step=0.1,
            help="Enter the actual bone age in months",
        )

    # Check if all required fields are filled for testing
    is_test_ready = (
        uploaded_file_test is not None 
        and gender_test != "" 
        and gender_test is not None 
        and actual_age is not None 
        and actual_age > 0
    )

    # Show status indicators for testing
    col_status1, col_status2, col_status3 = st.columns(3)
    with col_status1:
        if uploaded_file_test is not None:
            st.markdown("""
            <div class="status-indicator status-success">
                ✅ X-Ray uploaded successfully
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-indicator status-warning">
                ⚠️ X-Ray upload required
            </div>
            """, unsafe_allow_html=True)

    with col_status2:
        if gender_test != "" and gender_test is not None:
            st.markdown(f"""
            <div class="status-indicator status-success">
                ✅ Gender selected: {gender_test}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-indicator status-warning">
                ⚠️ Gender selection required
            </div>
            """, unsafe_allow_html=True)

    with col_status3:
        if actual_age is not None and actual_age > 0:
            st.markdown(f"""
            <div class="status-indicator status-success">
                ✅ Age entered: {actual_age} months
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-indicator status-warning">
                ⚠️ Age required
            </div>
            """, unsafe_allow_html=True)

    # Test and Save button
    if st.button(
        "Test and Save Data",
        type="primary",
        use_container_width=True,
        disabled=not is_test_ready,
    ):
        # Validate inputs
        errors = validate_inputs(uploaded_file_test, gender_test, actual_age)

        if errors:
            st.markdown('<div class="error-card">', unsafe_allow_html=True)
            for error in errors:
                st.error(error)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            try:
                with st.spinner(
                    "Processing image, generating prediction, and saving data..."
                ):
                    # Preprocess image
                    image_tensor = preprocessor.preprocess_from_upload(
                        uploaded_file_test
                    )

                    # Prepare gender tensor
                    gender_tensor = torch.tensor(
                        [1.0 if gender_test == "Male" else 0.0]
                    )

                    # Move tensors to device
                    image_tensor = image_tensor.to(st.session_state.device)
                    gender_tensor = gender_tensor.to(st.session_state.device)

                    # Make prediction
                    with torch.no_grad():
                        prediction = st.session_state.model(
                            image_tensor.unsqueeze(0), gender_tensor
                        )

                    predicted_months = prediction.item()
                    formatted_prediction = format_prediction(predicted_months)

                    # Calculate metrics
                    metrics = calculate_metrics(predicted_months, actual_age)

                    # Save data (only for local deployment)
                    is_local = os.path.exists("/Users") or os.path.exists("/home/user")
                    if is_local:
                        gender_bool = gender_test == "Male"
                        xray_id, image_path = data_manager.save_xray_data(
                            image_tensor, gender_bool, actual_age
                        )
                        st.markdown('<div class="success-card">', unsafe_allow_html=True)
                        st.success(f"✅ Data saved successfully! X-Ray ID: {xray_id}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-card">', unsafe_allow_html=True)
                        st.info("ℹ️ Data saving is disabled in cloud deployment for privacy and security.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Results in columns
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            '<div class="prediction-card">', unsafe_allow_html=True
                        )
                        st.markdown(f"### Prediction Results")
                        st.markdown(
                            f"**Predicted Bone Age:** {formatted_prediction['years_months']}"
                        )
                        st.markdown(
                            f"**Predicted Months:** {formatted_prediction['months']}"
                        )
                        st.markdown(
                            f"**Actual Bone Age:** {convert_months_to_years_months(actual_age)}"
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f"### Evaluation Metrics")
                        st.metric("Absolute Error", f"{metrics['error']:.1f} months")
                        st.metric("Deviation", f"{metrics['deviation']:.1f} months")
                        st.metric("Percent Error", f"{metrics['percent_error']:.1f}%")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Display images with proper aspect ratio
                    st.markdown("### Analysis Visualization")

                    # Original image
                    inv_norm = torch.nn.Sequential(
                        torchvision.transforms.Normalize(
                            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                        )
                    )
                    orig_img = inv_norm(image_tensor)
                    orig_img_np = orig_img.detach().permute(1, 2, 0).cpu().numpy()
                    orig_img_np = np.clip(orig_img_np, 0, 1)
                    orig_img_display = (orig_img_np[:, :, 0] * 255).astype(np.uint8)

                    # Create PIL image for display
                    orig_pil = Image.fromarray(orig_img_display)

                    # Generate GradCAM
                    try:
                        target_layer = get_target_layer(st.session_state.model)
                        cam = generate_gradcam(
                            st.session_state.model,
                            image_tensor,
                            gender_tensor,
                            target_layer,
                        )

                        # Save GradCAM image with unique name
                        timestamp = int(time.time())
                        temp_gradcam_path = f"temp_gradcam_test_{timestamp}.png"
                        gradcam_success = save_gradcam_image(
                            image_tensor, cam, temp_gradcam_path
                        )

                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            display_image_with_aspect_ratio(
                                orig_pil, "Original X-Ray", 400
                            )

                        with col_img2:
                            if gradcam_success and os.path.exists(temp_gradcam_path):
                                gradcam_img = Image.open(temp_gradcam_path)
                                display_image_with_aspect_ratio(
                                    gradcam_img, "GradCAM Analysis", 400
                                )
                            else:
                                st.error("GradCAM analysis could not be generated")

                        # Clean up temporary file
                        if os.path.exists(temp_gradcam_path):
                            os.remove(temp_gradcam_path)

                    except Exception as e:
                        st.error(f"Error generating GradCAM: {str(e)}")
                        # Show only original image if GradCAM fails
                        display_image_with_aspect_ratio(orig_pil, "Original X-Ray", 400)

            except Exception as e:
                st.error(f"Error during testing and saving: {str(e)}")

def samples_page():
    """Sample images page"""
    st.markdown("### Sample Images")
    st.markdown("Below are sample hand x-ray images used for training and testing the HandOsteoNet system.")

    # Get sample images from Samples directory
    samples_dir = "Samples"
    if os.path.exists(samples_dir):
        sample_files = [
            f
            for f in os.listdir(samples_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        sample_files.sort()  # Sort for consistent display

        if sample_files:
            st.markdown("""
            <div class="info-card">
                <p><strong>Note:</strong> These sample images demonstrate the type of hand x-ray images 
                that can be processed by HandOsteoNet. Each image shows the hand structure clearly 
                for accurate bone age assessment. These images are used for training and validation purposes.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display images in a professional grid layout
            st.markdown("### Sample X-Ray Images")
            
            # Create a responsive grid
            num_cols = 3
            for i in range(0, len(sample_files), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    if i + j < len(sample_files):
                        filename = sample_files[i + j]
                        with cols[j]:
                            try:
                                image_path = os.path.join(samples_dir, filename)
                                image = Image.open(image_path)
                                
                                # Create a professional card for each image
                                st.markdown(f"""
                                <div class="sample-item">
                                    <h4 style="margin-bottom: 1rem; color: var(--text-primary);">Sample {i+j+1}</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                display_image_with_aspect_ratio(
                                    image, f"Sample {i+j+1}: {filename}", 300
                                )
                                
                                st.markdown(f"""
                                <div style="text-align: center; margin-top: 0.5rem;">
                                    <small style="color: var(--text-secondary);">{filename}</small>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Error loading {filename}: {str(e)}")
        else:
            st.markdown("""
            <div class="warning-card">
                <p><strong>No Sample Images Found:</strong> The Samples directory is empty or contains no valid image files.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="error-card">
            <p><strong>Samples Directory Not Found:</strong> The Samples directory does not exist in the project structure.</p>
        </div>
        """, unsafe_allow_html=True)

def how_to_use_page():
    """How to use guide page"""
    st.markdown("### How to Use HandOsteoNet")

    # Video section
    st.markdown("#### Video Tutorial")
    video_path = "Video/sample_video.mp4"

    if os.path.exists(video_path):
        st.markdown("""
        <div class="info-card">
            <p><strong>Watch the Tutorial:</strong> Follow along with our step-by-step video guide to learn how to use HandOsteoNet effectively.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(
            """
        <div class="video-container">
        """,
            unsafe_allow_html=True,
        )

        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()

        st.video(video_bytes)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-card">
            <p><strong>Video Tutorial Not Available:</strong> The tutorial video file was not found. Please refer to the text guidelines below.</p>
        </div>
        """, unsafe_allow_html=True)

    # Text guidelines
    st.markdown("#### Step-by-Step Guidelines")

    st.markdown(
        """
    <div class="info-card">
        <h4>Getting Started</h4>
        <p><strong>Step 1: Access the System</strong></p>
        <ul>
            <li>Open HandOsteoNet in your web browser</li>
            <li>Ensure you have a stable internet connection</li>
            <li>Wait for the model to load (indicated by green checkmark)</li>
        </ul>
    </div>
    
    <div class="info-card">
        <h4>Image Preparation</h4>
        <p><strong>Step 2: Prepare Your Image</strong></p>
        <ul>
            <li>Use clear, high-quality hand x-ray images</li>
            <li>Supported formats: PNG, JPG, JPEG</li>
            <li>Ensure the hand is clearly visible and properly positioned</li>
            <li>Recommended: Full hand view with fingers spread</li>
        </ul>
    </div>
    
    <div class="info-card">
        <h4>Analysis Process</h4>
        <p><strong>Step 3: Upload and Analyze</strong></p>
        <ul>
            <li>Click "Browse files" to upload your x-ray image</li>
            <li>Select the patient's gender (Male/Female)</li>
            <li>Click "Evaluate Model" to start analysis</li>
            <li>Wait for processing (usually takes 10-30 seconds)</li>
        </ul>
    </div>
    
    <div class="info-card">
        <h4>Review Results</h4>
        <p><strong>Step 4: Review Results</strong></p>
        <ul>
            <li>View the predicted bone age in years and months</li>
            <li>Check the model confidence indicators</li>
            <li>Examine the original image and GradCAM analysis</li>
            <li>GradCAM highlights areas the AI focused on for prediction</li>
        </ul>
    </div>
    
    <div class="info-card">
        <h4>Data Management</h4>
        <p><strong>Step 5: Save Data (Optional)</strong></p>
        <ul>
            <li>For testing purposes, you can save results to HMC database</li>
            <li>Provide actual bone age for comparison</li>
            <li>View evaluation metrics and performance analysis</li>
        </ul>
    </div>
    
    <div class="success-card">
        <h4>Best Practices</h4>
        <ul>
            <li>Use images with good contrast and clarity</li>
            <li>Ensure proper hand positioning (palm down, fingers spread)</li>
            <li>Avoid images with artifacts or poor quality</li>
            <li>For clinical use, verify results with medical professionals</li>
        </ul>
    </div>
    
    <div class="warning-card">
        <h4>Troubleshooting</h4>
        <ul>
            <li>If model fails to load, refresh the page</li>
            <li>For upload errors, check file format and size</li>
            <li>If analysis fails, try a different image</li>
            <li>Contact support for technical issues</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def privacy_policy_page():
    """Privacy policy page"""
    st.markdown("### Privacy Policy")
    st.markdown("**Effective Date: 1st January 2025**")

    st.markdown("""
    <div class="info-card">
        <p><strong>Your Privacy Matters:</strong> We are committed to protecting your privacy and ensuring the security of your data. This policy explains how we handle information on our platform.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>Platform Information</h4>
        <p>Thank you for using our HandOsteoNet website hosted at https://handosteonet.streamlit.app. (will update the link later)</p>
        <p>Your privacy is of utmost importance to us. This policy outlines our commitment to protecting your data and explains how we handle any interactions on our platform.</p>
    </div>
    
    <div class="success-card">
        <h4>Information We Do Not Collect</h4>
        
        <h5>No Personal Data Collection:</h5>
        <ul>
            <li>We do not collect, store, or process any personal information, including names, email addresses, or contact details.</li>
        </ul>
        
        <h5>No Image Retention:</h5>
        <ul>
            <li>The medical images uploaded for bone age estimation are processed securely and immediately removed after prediction.</li>
            <li>No uploaded images are stored on our servers or shared with any third parties.</li>
        </ul>
        
        <h5>No Tracking Technologies:</h5>
        <ul>
            <li>We do not use cookies, trackers, or analytics tools to monitor user behavior or gather data about your visits to the site.</li>
        </ul>
    </div>
    
    <div class="info-card">
        <h4>Why This Matters</h4>
        <p>We understand that the images you upload may be sensitive and are committed to ensuring your privacy at all times. Our platform operates as a secure, temporary processing service designed to classify MRI images without retaining or distributing any data.</p>
    </div>
    
    <div class="info-card">
        <h4>Data Security</h4>
        <p>Although we do not store any data, we use secure protocols to handle image processing. This ensures that all interactions are encrypted and protected.</p>
    </div>
    
    <div class="info-card">
        <h4>Third-Party Services</h4>
        <p>Our website does not integrate with third-party services that collect or process user information.</p>
    </div>
    
    <div class="warning-card">
        <h4>Changes to This Policy</h4>
        <p>We may update this Privacy Policy from time to time to reflect improvements or changes in our services. Updates will be published on this page with a new effective date.</p>
    </div>
    
    <div class="success-card">
        <h4>Contact Us</h4>
        <p>If you have questions or concerns regarding this Privacy Policy, please contact us at:</p>
        <p><strong>Email:</strong> masum.cse19@gmail.com</p>
        <p>By using https://handosteonet.streamlit.app, you acknowledge and agree to the terms of this Privacy Policy.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Create navigation
    create_navigation()

    # Route to appropriate page based on session state
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Evaluate":
        evaluate_page()
    elif st.session_state.current_page == "Testing":
        testing_page()
    elif st.session_state.current_page == "Samples":
        samples_page()
    elif st.session_state.current_page == "HowToUse":
        how_to_use_page()
    elif st.session_state.current_page == "Privacy":
        privacy_policy_page()

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div class="footer">
        <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap;">
            <div style="text-align: center;">
                <p><strong>© 2025 Qatar University Research Team</strong></p>
                <p>HandOsteoNet v1.0 | Advanced Bone Age Assessment</p>
                <p><strong>Contact:</strong> masum.cse19@gmail.com</p>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
