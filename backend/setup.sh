# backend/setup.sh
#!/bin/bash

echo "Setting up FairFrame Bias Detection System..."
echo "==========================================="

# Create virtual environment
echo "1. Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "2. Activating virtual environment..."
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
echo "3. Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "4. Creating directories..."
mkdir -p uploads/images uploads/videos uploads/audio
mkdir -p reports static/visualizations temp

# Download required models
echo "5. Downloading AI models..."
echo "   This may take several minutes..."
python -c "import whisper; whisper.load_model('base')"
echo "   Whisper model downloaded"

# Test installation
echo "6. Testing installation..."
python -c "
try:
    import torch, whisper, cv2, deepface
    print('✅ All dependencies installed successfully!')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo ""
echo "==========================================="
echo "Setup complete!"
echo ""
echo "To start the server:"
echo "1. cd backend"
echo "2. python app.py"
echo ""
echo "The server will start at: http://localhost:8000"
echo "API docs at: http://localhost:8000/api/docs"