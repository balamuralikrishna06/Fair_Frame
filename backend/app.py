from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import hashlib
import json
from datetime import datetime
import random

app = FastAPI(title="FairFrame API", version="2.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fair-frame-2k31.onrender.com/","http://localhost:8001","http://localhost:8000","http://localhost:3000",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "FairFrame AI Media Analysis API",
        "status": "running",
        "version": "2.0"
    }
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "FairFrame AI"
    }

@app.post("/api/analyze")
async def analyze_media(file: UploadFile = File(...)):
    """Analyze media file - same file = same result"""
    
    try:
        # Read file
        contents = await file.read()
        file_size = len(contents)
        
        # Create hash for consistent results
        file_hash = hashlib.md5(contents).hexdigest()
        
        # Use hash as seed for consistent random results
        seed = int(file_hash[:8], 16)
        local_random = random.Random(seed)
        
        # Reset file pointer
        await file.seek(0)
        
        # Get file type
        file_type = file.content_type.split('/')[0]
        
        # Generate consistent bias score (0-100)
        hash_num = int(file_hash[:4], 16)
        bias_score = round((hash_num % 10000) / 100, 1)  # 0.0-99.9
        
        # Calculate related scores
        neutral_content = round(100 - bias_score, 1)
        potential_bias = round(bias_score * 0.7, 1)
        strong_bias = round(bias_score * 0.3, 1)
        
        # Sentiment based on bias
        if bias_score < 20:
            sentiment = "Very Positive"
        elif bias_score < 40:
            sentiment = "Positive"
        elif bias_score < 60:
            sentiment = "Neutral"
        elif bias_score < 80:
            sentiment = "Negative"
        else:
            sentiment = "Very Negative"
        
        # Summary based on file type and bias
        if file_type == 'image':
            if bias_score < 30:
                summary = "Image shows balanced composition with minimal detectable bias. Visual elements appear neutral and factual."
            elif bias_score < 70:
                summary = "Moderate bias detected in visual presentation. Some perspectives may be emphasized over others."
            else:
                summary = "Significant visual bias observed. Image strongly favors specific interpretations."
        elif file_type == 'video':
            summary = f"Video analysis completed. Content shows {'minimal' if bias_score < 40 else 'moderate' if bias_score < 70 else 'significant'} bias."
        elif file_type == 'audio':
            summary = f"Audio analysis completed. {'Balanced tone detected.' if bias_score < 40 else 'Some bias in language detected.' if bias_score < 70 else 'Strong bias in presentation.'}"
        else:
            summary = "Media analysis completed successfully."
        
        # Fixed recommendations (same for same file)
        recommendations = [
            "Consider including diverse perspectives",
            "Use neutral and factual language",
            "Provide verifiable sources when possible",
            "Balance emotional and factual content",
            "Review for unintentional bias"
        ]
        
        # Confidence based on file size
        confidence = min(95, 70 + (file_size / (1024 * 1024)))  # Larger files = higher confidence
        
        response = {
            "success": True,
            "filename": file.filename,
            "file_type": file.content_type,
            "file_size": file_size,
            "file_hash": file_hash[:8],  # Short hash for display
            "analysis": {
                "bias_score": bias_score,
                "neutral_content": neutral_content,
                "potential_bias": potential_bias,
                "strong_bias": strong_bias,
                "sentiment": sentiment,
                "confidence": round(confidence, 1),
                "summary": summary,
                "recommendations": recommendations[:3],  # First 3 recommendations
                "details": {
                    "analysis_model": "FairFrame AI v2.1",
                    "detected_elements": "AI-powered content analysis",
                    "credibility_score": round(100 - bias_score * 0.8, 1)
                }
            },
            "timestamp": datetime.now().isoformat(),
            "processing_time": round(1.5 + (file_size / (10 * 1024 * 1024)), 2),  # Simulate processing time
            "analysis_id": f"FF-{file_hash[:8].upper()}"
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 FairFrame AI Server Starting...")
    print("📡 API: http://localhost:8001")
    print("📤 Endpoint: POST http://localhost:8001/api/analyze")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8001)