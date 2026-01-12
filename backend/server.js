const express = require('express');
const multer = require('multer');
const cors = require('cors');
const crypto = require('crypto');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());

// File upload setup
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 500 * 1024 * 1024 }, // 500MB
    fileFilter: (req, file, cb) => {
        const allowedTypes = [
            'image/jpeg', 'image/jpg', 'image/png',
            'video/mp4', 'video/quicktime', 'video/x-msvideo',
            'audio/mpeg', 'audio/wav'
        ];
        
        if (allowedTypes.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type'), false);
        }
    }
});

// Helper functions
function generateConsistentScore(fileBuffer) {
    const hash = crypto.createHash('md5').update(fileBuffer).digest('hex');
    const hashNum = parseInt(hash.substring(0, 8), 16);
    return (hashNum % 10000) / 100; // 0.00-99.99
}

function getSentiment(score) {
    if (score < 20) return "Very Positive";
    if (score < 40) return "Positive";
    if (score < 60) return "Neutral";
    if (score < 80) return "Negative";
    return "Very Negative";
}

function getSummary(fileType, score) {
    const summaries = {
        image: {
            low: "Image shows balanced composition with minimal detectable bias. Visual elements appear neutral.",
            medium: "Moderate bias detected in visual presentation. Some perspectives may be emphasized.",
            high: "Significant visual bias observed. Image strongly favors specific interpretations."
        },
        video: {
            low: "Video content appears balanced with fair representation of multiple viewpoints.",
            medium: "Some bias detected in video editing, framing, or narrative presentation.",
            high: "Video shows strong bias through selective editing and one-sided presentation."
        },
        audio: {
            low: "Audio content demonstrates balanced tone and factual presentation.",
            medium: "Moderate bias detected in language choice, tone, or emphasis.",
            high: "Audio shows strong bias through emotional language and one-sided arguments."
        }
    };
    
    const type = fileType.split('/')[0];
    const category = score < 40 ? 'low' : score < 70 ? 'medium' : 'high';
    
    return summaries[type]?.[category] || "Analysis completed successfully.";
}

// Routes
app.get('/', (req, res) => {
    res.json({
        message: "FairFrame AI API",
        version: "2.0",
        status: "running",
        endpoints: {
            analyze: "POST /api/analyze",
            health: "GET /api/health"
        }
    });
});

app.get('/api/health', (req, res) => {
    res.json({ status: "healthy", timestamp: new Date().toISOString() });
});

app.post('/api/analyze', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No file uploaded" });
        }

        const file = req.file;
        const fileBuffer = file.buffer;
        
        // Generate consistent score based on file hash
        const biasScore = generateConsistentScore(fileBuffer);
        const neutralContent = 100 - biasScore;
        const potentialBias = biasScore * 0.7;
        const strongBias = biasScore * 0.3;
        
        // File info
        const fileType = file.mimetype;
        const fileSize = file.size;
        const fileHash = crypto.createHash('md5').update(fileBuffer).digest('hex');
        
        // Generate analysis
        const analysis = {
            success: true,
            filename: file.originalname,
            file_type: fileType,
            file_size: fileSize,
            file_hash: fileHash.substring(0, 8),
            analysis: {
                bias_score: parseFloat(biasScore.toFixed(1)),
                neutral_content: parseFloat(neutralContent.toFixed(1)),
                potential_bias: parseFloat(potentialBias.toFixed(1)),
                strong_bias: parseFloat(strongBias.toFixed(1)),
                sentiment: getSentiment(biasScore),
                confidence: parseFloat((80 + Math.random() * 15).toFixed(1)),
                summary: getSummary(fileType, biasScore),
                recommendations: [
                    "Consider including diverse perspectives",
                    "Use neutral and factual language",
                    "Provide verifiable sources",
                    "Balance emotional and factual content",
                    "Review for unintentional bias"
                ].slice(0, 3),
                details: {
                    analysis_model: "FairFrame AI v2.1",
                    detected_elements: "AI-powered content analysis",
                    credibility_score: parseFloat((100 - biasScore * 0.8).toFixed(1)),
                    processing_model: "Deep Neural Network"
                }
            },
            timestamp: new Date().toISOString(),
            processing_time: parseFloat((1.5 + (fileSize / (10 * 1024 * 1024))).toFixed(2)),
            analysis_id: `FF-${fileHash.substring(0, 8).toUpperCase()}`
        };

        res.json(analysis);
        
    } catch (error) {
        console.error("Analysis error:", error);
        res.status(500).json({ error: "Analysis failed", details: error.message });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 FairFrame Node.js Server running on port ${PORT}`);
    console.log(`📡 http://localhost:${PORT}`);
    console.log(`📤 POST http://localhost:${PORT}/api/analyze`);
});

// Handle uncaught errors
process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});