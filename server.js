/**
 * TB Screening AI - Proxy Server (Improved Production Version)
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

// ==========================================================
// Logger
// ==========================================================
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

// ==========================================================
// Configuration
// ==========================================================
const CONFIG = {
    PORT: parseInt(process.env.PORT) || 10000,
    HF_SPACE_URL: (process.env.HF_SPACE_URL || '').replace(/\/$/, ''),
    HF_TOKEN: process.env.HF_TOKEN,
    API_KEY: process.env.API_KEY,

    RATE_LIMIT_WINDOW: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
    RATE_LIMIT_MAX: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,

    ALLOWED_ORIGINS: process.env.ALLOWED_ORIGINS
        ? process.env.ALLOWED_ORIGINS.split(',').map(s => s.trim())
        : ['*'],

    SINGLE_TIMEOUT_MS: parseInt(process.env.SINGLE_TIMEOUT_MS) || 60000,
    BATCH_TIMEOUT_MS: parseInt(process.env.BATCH_TIMEOUT_MS) || 180000,

    MAX_BATCH_FILES: parseInt(process.env.MAX_BATCH_FILES) || 20,
    MAX_FILE_SIZE_MB: parseInt(process.env.MAX_FILE_SIZE_MB) || 50
};

// ==========================================================
// ENV VALIDATION (NO HARD CRASH)
// ==========================================================
const missing = ['HF_SPACE_URL', 'HF_TOKEN', 'API_KEY'].filter(k => !CONFIG[k]);

if (missing.length) {
    logger.error(`❌ Missing ENV variables: ${missing.join(', ')}`);
    logger.warn("⚠️ Server will still start, but requests will fail.");
}

// ==========================================================
// Express App
// ==========================================================
const app = express();

// Security
app.use(helmet({
    crossOriginResourcePolicy: { policy: 'cross-origin' }
}));

// CORS
const corsOptions = {
    origin: (origin, callback) => {
        if (!origin) return callback(null, true);

        const allowed =
            CONFIG.ALLOWED_ORIGINS.includes('*') ||
            CONFIG.ALLOWED_ORIGINS.some(o => origin.includes(o)) ||
            /\.vercel\.app$/.test(origin) ||
            /localhost(:\d+)?$/.test(origin);

        if (allowed) callback(null, true);
        else callback(new Error(`CORS blocked: ${origin}`));
    },
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Request-ID'],
    exposedHeaders: ['Content-Disposition', 'Content-Length'],
    credentials: true
};

app.use(cors(corsOptions));
app.options('*', cors(corsOptions));

app.use(compression());

// Rate limit
app.use('/api/', rateLimit({
    windowMs: CONFIG.RATE_LIMIT_WINDOW,
    max: CONFIG.RATE_LIMIT_MAX
}));

// Body parsing
app.use(express.json({ limit: '10mb' }));

// ==========================================================
// Multer
// ==========================================================
const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: CONFIG.MAX_FILE_SIZE_MB * 1024 * 1024,
        files: CONFIG.MAX_BATCH_FILES
    }
});

// ==========================================================
// Request Logging
// ==========================================================
app.use((req, res, next) => {
    req.id = uuidv4();
    const start = Date.now();

    res.on('finish', () => {
        logger.info(`${req.method} ${req.path} ${res.statusCode} - ${Date.now() - start}ms`);
    });

    next();
});

// ==========================================================
// Helper: Call HF Space with timeout + retry
// ==========================================================
async function callHFSpace(path, formData, timeoutMs, retries = 1) {
    const url = `${CONFIG.HF_SPACE_URL}${path}`;

    for (let attempt = 0; attempt <= retries; attempt++) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${CONFIG.HF_TOKEN}`,
                    'X-API-Key': CONFIG.API_KEY,
                    ...formData.getHeaders()
                },
                body: formData,
                signal: controller.signal
            });

            clearTimeout(timer);

            if (response.status === 503 && attempt < retries) {
                logger.warn("HF cold start, retrying...");
                continue;
            }

            return response;

        } catch (err) {
            clearTimeout(timer);

            if (err.name === 'AbortError') {
                if (attempt < retries) continue;
                throw new Error(`Timeout after ${timeoutMs}ms`);
            }

            throw err;
        }
    }
}

// ==========================================================
// Routes
// ==========================================================

app.get('/', (_, res) => {
    res.json({ status: 'running' });
});

app.get('/health', (_, res) => {
    res.json({ status: 'healthy' });
});

// ==========================================================
// Single Prediction
// ==========================================================
app.post('/api/predict', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }

    try {
        const formData = new FormData();
        formData.append('file', req.file.buffer, req.file.originalname);

        const hfRes = await callHFSpace('/predict', formData, CONFIG.SINGLE_TIMEOUT_MS);

        if (!hfRes.ok) {
            return res.status(502).json({ error: `HF error ${hfRes.status}` });
        }

        const data = await hfRes.json();
        res.json(data);

    } catch (err) {
        logger.error(err);
        res.status(500).json({ error: err.message });
    }
});

// ==========================================================
// Batch Prediction
// ==========================================================
app.post('/api/batch/predict', upload.array('files'), async (req, res) => {
    if (!req.files || req.files.length === 0) {
        return res.status(400).json({ error: 'No files uploaded' });
    }

    try {
        const formData = new FormData();

        for (const file of req.files) {
            formData.append('files', file.buffer, file.originalname);
        }

        const hfRes = await callHFSpace('/batch/predict', formData, CONFIG.BATCH_TIMEOUT_MS);

        if (!hfRes.ok) {
            return res.status(502).json({ error: `HF error ${hfRes.status}` });
        }

        const buffer = await hfRes.buffer();

        res.setHeader('Content-Type', 'text/csv');
        res.setHeader('Content-Disposition', 'attachment; filename="results.csv"');

        res.send(buffer);

    } catch (err) {
        logger.error(err);
        res.status(500).json({ error: err.message });
    }
});

// ==========================================================
// Error Handler
// ==========================================================
app.use((err, req, res, next) => {
    logger.error(err);
    res.status(500).json({ error: 'Internal server error' });
});

// ==========================================================
// START SERVER
// ==========================================================
app.listen(CONFIG.PORT, () => {
    logger.info(`🚀 Server running on port ${CONFIG.PORT}`);
    logger.info(`HF URL: ${CONFIG.HF_SPACE_URL || 'NOT SET'}`);
});
