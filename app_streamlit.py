"""
Streamlit Application for Object Detection
Simplified version for easy deployment on Streamlit Cloud
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="Object Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Object Detection & Image Processing")
st.markdown("Upload an image to see edge detection results (simplified version for Streamlit deployment)")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            # Convert BGR to RGB for display
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(rgb_image, use_column_width=True)
        
        with col2:
            st.subheader("Edge Detection Result")
            # Apply edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            rgb_edges = cv2.cvtColor(edges_colored, cv2.COLOR_BGR2RGB)
            st.image(rgb_edges, use_column_width=True)
        
        st.success("✅ Image processed successfully!")
        st.info("Note: Full YOLO object detection model not available on Streamlit due to size limits. This shows edge detection as a demonstration.")
    else:
        st.error("❌ Error: Could not read the image file. Please try a different image.")
else:
    st.info("👆 Upload an image to get started")

# Add footer
st.markdown("---")
st.markdown("**Note:** This is a simplified version using edge detection. The full YOLO model requires larger deployment resources.")
