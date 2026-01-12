# backend/app_fixed.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import uuid
import os

app = FastAPI()

# Fix CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "app": "FairFrame",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/analyze")
async def analyze_media(file: UploadFile = File(...), media_type: str = "video"):
    """Simple test endpoint"""
    return {
        "status": "success",
        "job_id": str(uuid.uuid4()),
        "filename": file.filename,
        "media_type": media_type,
        "message": "File received successfully",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint for frontend"""
    return {
        "message": "API is working!",
        "data": [1, 2, 3, 4, 5],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting FairFrame API...")
    print("📡 http://localhost:8000")
    print("📚 http://localhost:8000/docs")
    print("🔄 CORS: Enabled for all origins")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)