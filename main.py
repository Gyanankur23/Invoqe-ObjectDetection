"""
FastAPI Application for Object Detection & Tracking
REST API for computer vision tasks
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
import io
from PIL import Image
import uvicorn

from detection import ObjectDetector
from tracking import DetectionAndTracking

# Initialize FastAPI app
app = FastAPI(
    title="Object Detection & Tracking API",
    description="Computer Vision API for object detection and tracking using YOLO",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
detector = None
tracking_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global detector, tracking_system
    try:
        print("Loading YOLO model...")
        detector = ObjectDetector('yolov8n.pt')
        tracking_system = DetectionAndTracking('yolov8n.pt')
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")

# Request/Response models
class DetectionResponse(BaseModel):
    status: str
    object_count: int
    detections: list

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        message="Object Detection & Tracking API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    return HealthResponse(
        status="healthy",
        message="Models are ready"
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5
):
    """Detect objects in uploaded image"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect objects
        annotated_image, detections = detector.detect_frame(image, conf_threshold)
        
        # Convert to PIL Image
        rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Return image and detection info
        return StreamingResponse(
            img_byte_arr,
            media_type="image/jpeg",
            headers={
                "X-Object-Count": str(len(detections)),
                "X-Detections": str(detections)
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/info")
async def detect_objects_info(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5
):
    """Detect objects and return JSON info without image"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect objects
        _, detections = detector.detect_frame(image, conf_threshold)
        
        return DetectionResponse(
            status="success",
            object_count=len(detections),
            detections=detections
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classes")
async def get_classes():
    """Get list of detectable classes"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    return {
        "classes": detector.class_names,
        "count": len(detector.class_names)
    }

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
