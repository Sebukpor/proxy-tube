/**
 * TB Screening AI - Proxy Server (API Only)
 * For Render.com deployment - Frontend hosted separately on Vercel
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
    HF_TOKEN: process.env.HF_TOKEN,
    API_KEY: process.env.API_KEY,
    RATE_LIMIT_WINDOW: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
    RATE_LIMIT_MAX: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,
    ALLOWED_ORIGINS: process.env.ALLOWED_ORIGINS ? process.env.ALLOWED_ORIGINS.split(',') : ['*']
};

// Validate required environment variables
if (!CONFIG.HF_SPACE_URL || !CONFIG.HF_TOKEN || !CONFIG.API_KEY) {
    logger.error('Missing required environment variables:');
    if (!CONFIG.HF_SPACE_URL) logger.error('  - HF_SPACE_URL');
    if (!CONFIG.HF_TOKEN) logger.error('  - HF_TOKEN');
    if (!CONFIG.API_KEY) logger.error('  - API_KEY');
    process.exit(1);
}

// Initialize Express
const app = express();

// Security Middleware
app.use(helmet());

// CORS Configuration - IMPORTANT for Vercel frontend
const corsOptions = {
    origin: (origin, callback) => {
        // Allow requests with no origin (mobile apps, curl, etc.)
        if (!origin) return callback(null, true);
        
        if (CONFIG.ALLOWED_ORIGINS.includes('*') || 
            CONFIG.ALLOWED_ORIGINS.some(allowed => origin.includes(allowed)) ||
            origin.includes('vercel.app')) {  // Auto-allow Vercel domains
            callback(null, true);
        } else {
            logger.warn(`CORS blocked origin: ${origin}`);
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
    legacyHeaders: false
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
        fileSize: 50 * 1024 * 1024,
        files: 20
    },
    fileFilter: (req, file, cb) => {
        const allowedExts = ['.dcm', '.png', '.jpg', '.jpeg'];
        const ext = path.extname(file.originalname).toLowerCase();
        
        if (allowedExts.includes(ext)) {
            cb(null, true);
        } else {
            cb(new Error(`Invalid file type: ${file.originalname}`));
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
            ip: req.ip,
            origin: req.headers.origin
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
        uptime: process.uptime()
    });
});

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        service: 'TB Screening AI Proxy',
        status: 'running',
        endpoints: {
            health: '/health',
            predict: '/api/predict (POST)',
            batchPredict: '/api/batch/predict (POST)'
        },
        cors: 'Enabled for Vercel frontend'
    });
});

// Make HF Request helper
async function makeHFRequest(endpoint, formData, requestId) {
    const url = `${CONFIG.HF_SPACE_URL}${endpoint}`;
    
    const headers = {
        'Authorization': `Bearer ${CONFIG.HF_TOKEN}`,
        'X-API-Key': CONFIG.API_KEY,
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

// Single Prediction
app.post('/api/predict', upload.single('file'), async (req, res) => {
    const requestId = req.id;
    
    try {
        if (!req.file) {
            return res.status(400).json({
                error: 'No file provided',
                message: 'Please upload a DICOM or image file.'
            });
        }

        logger.info(`Processing: ${req.file.originalname}`, { requestId });

        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype || 'application/octet-stream'
        });

        const hfResponse = await makeHFRequest('/predict', formData, requestId);

        if (!hfResponse.ok) {
            const errorText = await hfResponse.text();
            logger.error(`HF error: ${hfResponse.status} - ${errorText}`, { requestId });
            
            if (hfResponse.status === 401 || hfResponse.status === 403) {
                return res.status(503).json({
                    error: 'Authentication failed',
                    message: 'Service authentication error.'
                });
            }
            
            throw new Error(`HF returned ${hfResponse.status}`);
        }

        const result = await hfResponse.json();
        
        res.json({
            ...result,
            proxied: true,
            proxy_request_id: requestId,
            proxy_timestamp: new Date().toISOString()
        });

    } catch (error) {
        logger.error(`Error: ${error.message}`, { requestId, stack: error.stack });
        res.status(500).json({
            error: 'Analysis failed',
            message: 'Unable to process image.',
            requestId: requestId
        });
    }
});

// Batch Prediction
app.post('/api/batch/predict', upload.array('files', 20), async (req, res) => {
    const requestId = req.id;
    
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({
                error: 'No files provided',
                message: 'Please upload at least one file.'
            });
        }

        logger.info(`Batch processing: ${req.files.length} files`, { requestId });

        const formData = new FormData();
        req.files.forEach(file => {
            formData.append('files', file.buffer, {
                filename: file.originalname,
                contentType: file.mimetype || 'application/octet-stream'
            });
        });

        const hfResponse = await makeHFRequest('/batch/predict', formData, requestId);

        if (!hfResponse.ok) {
            const errorText = await hfResponse.text();
            logger.error(`HF batch error: ${hfResponse.status}`, { requestId });
            
            if (hfResponse.status === 401 || hfResponse.status === 403) {
                return res.status(503).json({
                    error: 'Authentication failed',
                    message: 'Service authentication error.'
                });
            }
            
            throw new Error(`HF returned ${hfResponse.status}`);
        }

        const contentType = hfResponse.headers.get('content-type');
        const contentDisposition = hfResponse.headers.get('content-disposition');
        
        res.setHeader('Content-Type', contentType || 'text/csv');
        if (contentDisposition) {
            res.setHeader('Content-Disposition', contentDisposition);
        }
        res.setHeader('X-Proxy-Request-ID', requestId);

        hfResponse.body.pipe(res);

    } catch (error) {
        logger.error(`Batch error: ${error.message}`, { requestId });
        res.status(500).json({
            error: 'Batch analysis failed',
            message: 'Unable to process images.',
            requestId: requestId
        });
    }
});

// Error Handling
app.use((err, req, res, next) => {
    logger.error('Error:', {
        message: err.message,
        requestId: req.id
    });

    if (err instanceof multer.MulterError) {
        if (err.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({
                error: 'File too large',
                message: 'Maximum file size is 50MB.'
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

// Start Server
app.listen(CONFIG.PORT, () => {
    logger.info(`🚀 Server running on port ${CONFIG.PORT}`);
    logger.info(`📡 HF Space: ${CONFIG.HF_SPACE_URL}`);
    logger.info(`🔐 Auth: ${CONFIG.HF_TOKEN ? 'HF Token OK' : 'Missing'}, ${CONFIG.API_KEY ? 'API Key OK' : 'Missing'}`);
});
