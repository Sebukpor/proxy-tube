/**
 * TB Screening AI - Proxy Server
 * Secure intermediary between frontend and private Hugging Face Space
 * Handles: HF Bearer Token (for private space) + FastAPI X-API-Key (for app auth)
 * Deploy to Render.com
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const multer = require('multer');
const FormData = require('form-data');
const fetch = require('node-fetch');
const winston = require('winston');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

// Logger Configuration
const logger = winston.createLogger({
    level: process.env.NODE_ENV === 'production' ? 'info' : 'debug',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    transports: [
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.colorize(),
                winston.format.simple()
            )
        })
    ]
});

// Configuration
const CONFIG = {
    PORT: process.env.PORT || 10000,
    HF_SPACE_URL: process.env.HF_SPACE_URL,
    HF_TOKEN: process.env.HF_TOKEN,           // Hugging Face Bearer token
    API_KEY: process.env.API_KEY,             // Your FastAPI app key
    RATE_LIMIT_WINDOW: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
    RATE_LIMIT_MAX: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,
    ALLOWED_ORIGINS: process.env.ALLOWED_ORIGINS ? process.env.ALLOWED_ORIGINS.split(',') : ['*']
};

// Validate required environment variables
if (!CONFIG.HF_SPACE_URL || !CONFIG.HF_TOKEN || !CONFIG.API_KEY) {
    logger.error('Missing required environment variables:');
    if (!CONFIG.HF_SPACE_URL) logger.error('  - HF_SPACE_URL');
    if (!CONFIG.HF_TOKEN) logger.error('  - HF_TOKEN (Hugging Face read token)');
    if (!CONFIG.API_KEY) logger.error('  - API_KEY (FastAPI app key)');
    process.exit(1);
}

// Initialize Express
const app = express();

// Security Middleware
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com", "https://cdnjs.cloudflare.com"],
            fontSrc: ["'self'", "https://fonts.gstatic.com", "https://cdnjs.cloudflare.com"],
            scriptSrc: ["'self'"],
            imgSrc: ["'self'", "data:", "blob:"],
            connectSrc: ["'self'"]
        }
    },
    crossOriginEmbedderPolicy: false
}));

// CORS Configuration
const corsOptions = {
    origin: (origin, callback) => {
        if (CONFIG.ALLOWED_ORIGINS.includes('*') || CONFIG.ALLOWED_ORIGINS.includes(origin) || !origin) {
            callback(null, true);
        } else {
            callback(new Error('Not allowed by CORS'));
        }
    },
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Request-ID'],
    credentials: true,
    maxAge: 86400
};

app.use(cors(corsOptions));
app.use(compression());

// Rate Limiting
const limiter = rateLimit({
    windowMs: CONFIG.RATE_LIMIT_WINDOW,
    max: CONFIG.RATE_LIMIT_MAX,
    message: {
        error: 'Too many requests from this IP, please try again later.',
        retryAfter: Math.ceil(CONFIG.RATE_LIMIT_WINDOW / 1000)
    },
    standardHeaders: true,
    legacyHeaders: false,
    handler: (req, res) => {
        logger.warn(`Rate limit exceeded for IP: ${req.ip}`);
        res.status(429).json({
            error: 'Too many requests',
            message: 'Please slow down and try again later.'
        });
    }
});

app.use('/api/', limiter);

// Body Parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Multer Configuration
const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: {
        fileSize: 50 * 1024 * 1024, // 50MB max
        files: 20 // Max 20 files per batch
    },
    fileFilter: (req, file, cb) => {
        const allowedMimes = ['image/png', 'image/jpeg', 'image/jpg', 'application/dicom'];
        const allowedExts = ['.dcm', '.png', '.jpg', '.jpeg'];
        const ext = path.extname(file.originalname).toLowerCase();
        
        if (allowedMimes.includes(file.mimetype) || allowedExts.includes(ext)) {
            cb(null, true);
        } else {
            cb(new Error(`Invalid file type: ${file.originalname}. Only DICOM and image files are allowed.`));
        }
    }
});

// Request ID Middleware
app.use((req, res, next) => {
    req.id = uuidv4();
    res.setHeader('X-Request-ID', req.id);
    next();
});

// Logging Middleware
app.use((req, res, next) => {
    const start = Date.now();
    res.on('finish', () => {
        const duration = Date.now() - start;
        logger.info({
            requestId: req.id,
            method: req.method,
            path: req.path,
            statusCode: res.statusCode,
            duration: `${duration}ms`,
            ip: req.ip
        });
    });
    next();
});

// Health Check
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        uptime: process.uptime(),
        hf_connected: !!CONFIG.HF_TOKEN,
        api_key_configured: !!CONFIG.API_KEY
    });
});

/**
 * Make authenticated request to Hugging Face Space
 * Uses Bearer token for HF auth + X-API-Key for FastAPI app auth
 */
async function makeHFRequest(endpoint, formData, requestId) {
    const url = `${CONFIG.HF_SPACE_URL}${endpoint}`;
    
    const headers = {
        'Authorization': `Bearer ${CONFIG.HF_TOKEN}`,  // HF Private Space auth
        'X-API-Key': CONFIG.API_KEY,                   // Your FastAPI app auth
        ...formData.getHeaders()
    };

    logger.debug(`Making request to ${url}`, { requestId });

    const response = await fetch(url, {
        method: 'POST',
        headers: headers,
        body: formData
    });

    return response;
}

// Single Prediction Endpoint
app.post('/api/predict', upload.single('file'), async (req, res) => {
    const requestId = req.id;
    
    try {
        if (!req.file) {
            return res.status(400).json({
                error: 'No file provided',
                message: 'Please upload a DICOM or image file.'
            });
        }

        logger.info(`Processing single prediction for file: ${req.file.originalname}`, { requestId });

        // Prepare form data for HF Space
        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype || 'application/octet-stream'
        });

        // Forward to Hugging Face Space with both auth tokens
        const hfResponse = await makeHFRequest('/predict', formData, requestId);

        if (!hfResponse.ok) {
            const errorText = await hfResponse.text();
            logger.error(`HF Space error: ${hfResponse.status} - ${errorText}`, { requestId });
            
            if (hfResponse.status === 401 || hfResponse.status === 403) {
                return res.status(503).json({
                    error: 'Authentication failed',
                    message: 'Service authentication error. Please contact support.'
                });
            }
            
            throw new Error(`HF Space returned ${hfResponse.status}`);
        }

        const result = await hfResponse.json();
        
        // Add proxy metadata
        const enhancedResult = {
            ...result,
            proxied: true,
            proxy_request_id: requestId,
            proxy_timestamp: new Date().toISOString()
        };

        logger.info(`Prediction successful for ${req.file.originalname}`, { 
            requestId, 
            prediction: enhancedResult.result?.prediction,
            probability: enhancedResult.result?.probability
        });

        res.json(enhancedResult);

    } catch (error) {
        logger.error(`Prediction error: ${error.message}`, { requestId, stack: error.stack });
        
        res.status(500).json({
            error: 'Analysis failed',
            message: 'Unable to process image. Please try again later.',
            requestId: requestId
        });
    }
});

// Batch Prediction Endpoint
app.post('/api/batch/predict', upload.array('files', 20), async (req, res) => {
    const requestId = req.id;
    
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({
                error: 'No files provided',
                message: 'Please upload at least one file.'
            });
        }

        logger.info(`Processing batch prediction for ${req.files.length} files`, { requestId });

        // Prepare form data for HF Space
        const formData = new FormData();
        req.files.forEach(file => {
            formData.append('files', file.buffer, {
                filename: file.originalname,
                contentType: file.mimetype || 'application/octet-stream'
            });
        });

        // Forward to Hugging Face Space with both auth tokens
        const hfResponse = await makeHFRequest('/batch/predict', formData, requestId);

        if (!hfResponse.ok) {
            const errorText = await hfResponse.text();
            logger.error(`HF Space batch error: ${hfResponse.status} - ${errorText}`, { requestId });
            
            if (hfResponse.status === 401 || hfResponse.status === 403) {
                return res.status(503).json({
                    error: 'Authentication failed',
                    message: 'Service authentication error. Please contact support.'
                });
            }
            
            throw new Error(`HF Space returned ${hfResponse.status}`);
        }

        // Stream the CSV response back to client
        const contentType = hfResponse.headers.get('content-type');
        const contentDisposition = hfResponse.headers.get('content-disposition');
        
        res.setHeader('Content-Type', contentType || 'text/csv');
        if (contentDisposition) {
            res.setHeader('Content-Disposition', contentDisposition);
        }
        res.setHeader('X-Proxy-Request-ID', requestId);

        hfResponse.body.pipe(res);

        logger.info(`Batch prediction successful for ${req.files.length} files`, { requestId });

    } catch (error) {
        logger.error(`Batch prediction error: ${error.message}`, { requestId, stack: error.stack });
        
        res.status(500).json({
            error: 'Batch analysis failed',
            message: 'Unable to process images. Please try again later.',
            requestId: requestId
        });
    }
});

// Serve Static Frontend Files
const frontendPath = path.join(__dirname, '../frontend');
if (fs.existsSync(frontendPath)) {
    app.use(express.static(frontendPath));
    
    app.get('/', (req, res) => {
        res.sendFile(path.join(frontendPath, 'index.html'));
    });
    
    logger.info(`Serving frontend from: ${frontendPath}`);
} else {
    logger.warn('Frontend directory not found. API-only mode.');
    app.get('/', (req, res) => {
        res.json({
            service: 'TB Screening AI Proxy',
            status: 'running',
            endpoints: {
                health: '/health',
                predict: '/api/predict (POST)',
                batchPredict: '/api/batch/predict (POST)'
            },
            auth: {
                hf_token_configured: !!CONFIG.HF_TOKEN,
                api_key_configured: !!CONFIG.API_KEY
            }
        });
    });
}

// Error Handling Middleware
app.use((err, req, res, next) => {
    logger.error('Unhandled error:', {
        message: err.message,
        stack: err.stack,
        requestId: req.id
    });

    if (err instanceof multer.MulterError) {
        if (err.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({
                error: 'File too large',
                message: 'Maximum file size is 50MB.'
            });
        }
        if (err.code === 'LIMIT_FILE_COUNT') {
            return res.status(400).json({
                error: 'Too many files',
                message: 'Maximum 20 files allowed per batch.'
            });
        }
    }

    res.status(500).json({
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'production' ? 'Something went wrong.' : err.message,
        requestId: req.id
    });
});

// 404 Handler
app.use((req, res) => {
    res.status(404).json({
        error: 'Not found',
        message: `Endpoint ${req.method} ${req.path} does not exist.`
    });
});

// Graceful Shutdown
process.on('SIGTERM', () => {
    logger.info('SIGTERM received, shutting down gracefully');
    process.exit(0);
});

process.on('SIGINT', () => {
    logger.info('SIGINT received, shutting down gracefully');
    process.exit(0);
});

// Start Server
app.listen(CONFIG.PORT, () => {
    logger.info(`🚀 TB Screening Proxy Server running on port ${CONFIG.PORT}`);
    logger.info(`📡 HF Space URL: ${CONFIG.HF_SPACE_URL}`);
    logger.info(`🔐 HF Token configured: ${CONFIG.HF_TOKEN ? 'Yes' : 'No'}`);
    logger.info(`🔑 API Key configured: ${CONFIG.API_KEY ? 'Yes' : 'No'}`);
    logger.info(`🌐 CORS Origins: ${CONFIG.ALLOWED_ORIGINS.join(', ')}`);
});
