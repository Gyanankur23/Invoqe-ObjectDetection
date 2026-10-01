"""
FastAPI Application for Object Detection & Tracking
REST API for computer vision tasks
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
import io
from PIL import Image
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Object Detection & Tracking API",
    description="Computer Vision API for image processing",
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

# Request/Response models
class DetectionResponse(BaseModel):
    status: str
    object_count: int
    detections: list

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/")
async def root():
    """Root endpoint - return HTML frontend"""
    return HTMLResponse(content=open('web_interface.html', 'r', encoding='utf-8').read())

@app.get("/api", response_model=HealthResponse)
async def api_root():
    """API root endpoint"""
    return HealthResponse(
        status="healthy",
        message="Object Detection API is running (simplified version for Vercel)"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is ready"
    )

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5
):
    """Process uploaded image (simplified version)"""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Simple image processing (edge detection as placeholder)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Convert to PIL Image
        rgb_image = cv2.cvtColor(edges_colored, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Return processed image
        return StreamingResponse(
            img_byte_arr,
            media_type="image/jpeg",
            headers={
                "X-Object-Count": "0",
                "X-Detections": "[]"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/info")
async def detect_objects_info(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5
):
    """Process image and return JSON info (simplified version)"""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Return mock detection info
        return {
            "status": "success",
            "object_count": 0,
            "detections": []
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classes")
async def get_classes():
    """Get list of detectable classes (simplified version)"""
    return {
        "classes": ["Note: Full YOLO model not available on Vercel due to size limits"],
        "count": 1
    }

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
