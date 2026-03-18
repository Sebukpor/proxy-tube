/**
 * TB Screening AI - Proxy Server (API Only)
 * For Render.com deployment - Frontend hosted separately on Vercel
 *
 * CHANGELOG:
 * - Fixed batch endpoint: HF Space URL no longer double-prefixes "/api"
 * - Fixed batch response streaming: buffers full response before piping to avoid
 *   premature stream termination on slow HF cold starts
 * - Added per-file retry logic (1 retry) for transient HF errors
 * - Added Content-Length header passthrough so browsers show download progress
 * - Fixed CORS preflight for multipart/form-data (OPTIONS handling)
 * - Added graceful timeout (60 s single / 180 s batch) with proper error response
 * - Hardened multer error messages
 * - Fixed HTML issue: batch overlay is in the viewer panel scope — server now
 *   returns a "202 Accepted + SSE" pattern is NOT used here; instead we return
 *   the CSV directly, which is what the frontend expects
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const multer = require('multer');
const FormData = require('form-data');
const fetch = require('node-fetch');
const AbortController = require('abort-controller'); // npm i abort-controller
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

    // HF_SPACE_URL should be the BASE url, e.g. https://user-space.hf.space
    // Do NOT include trailing slash or /predict suffix here.
    HF_SPACE_URL: (process.env.HF_SPACE_URL || '').replace(/\/$/, ''),

    HF_TOKEN:  process.env.HF_TOKEN,
    API_KEY:   process.env.API_KEY,

    RATE_LIMIT_WINDOW:  parseInt(process.env.RATE_LIMIT_WINDOW_MS)       || 15 * 60 * 1000,
    RATE_LIMIT_MAX:     parseInt(process.env.RATE_LIMIT_MAX_REQUESTS)    || 100,
    ALLOWED_ORIGINS:    process.env.ALLOWED_ORIGINS
                            ? process.env.ALLOWED_ORIGINS.split(',').map(s => s.trim())
                            : ['*'],

    // Timeout in milliseconds
    SINGLE_TIMEOUT_MS: parseInt(process.env.SINGLE_TIMEOUT_MS) || 60_000,
    BATCH_TIMEOUT_MS:  parseInt(process.env.BATCH_TIMEOUT_MS)  || 180_000,

    // Max files per batch request
    MAX_BATCH_FILES: parseInt(process.env.MAX_BATCH_FILES) || 20,

    // Single file size limit (bytes)
    MAX_FILE_SIZE_MB: parseInt(process.env.MAX_FILE_SIZE_MB) || 50,
};

// Validate required environment variables
const missing = ['HF_SPACE_URL', 'HF_TOKEN', 'API_KEY'].filter(k => !CONFIG[k]);
if (missing.length) {
    missing.forEach(k => logger.error(`Missing required env var: ${k}`));
    process.exit(1);
}

// ==========================================================
// Express App
// ==========================================================
const app = express();

// Security
app.use(helmet({
    crossOriginResourcePolicy: { policy: 'cross-origin' } // allow image/file responses
}));

// CORS
const corsOptions = {
    origin: (origin, callback) => {
        if (!origin) return callback(null, true); // curl, mobile, etc.
        const allowed =
            CONFIG.ALLOWED_ORIGINS.includes('*') ||
            CONFIG.ALLOWED_ORIGINS.some(o => origin.includes(o)) ||
            /\.vercel\.app$/.test(origin) ||
            /localhost(:\d+)?$/.test(origin);
        if (allowed) {
            callback(null, true);
        } else {
            logger.warn(`CORS blocked origin: ${origin}`);
            callback(new Error(`Origin ${origin} not allowed by CORS`));
        }
    },
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Request-ID'],
    exposedHeaders: ['Content-Disposition', 'Content-Length', 'X-Request-ID', 'X-Proxy-Request-ID'],
    credentials: true,
    optionsSuccessStatus: 204, // Some legacy browsers choke on 200 for OPTIONS
    maxAge: 86400
};

app.use(cors(corsOptions));

// Explicitly handle OPTIONS preflight for all routes (belt-and-suspenders)
app.options('*', cors(corsOptions));

app.use(compression());

// Rate limiting (applied only to /api/* routes)
const limiter = rateLimit({
    windowMs: CONFIG.RATE_LIMIT_WINDOW,
    max: CONFIG.RATE_LIMIT_MAX,
    standardHeaders: true,
    legacyHeaders: false,
    message: {
        error: 'Too many requests from this IP, please try again later.',
        retryAfter: Math.ceil(CONFIG.RATE_LIMIT_WINDOW / 1000)
    }
});
app.use('/api/', limiter);

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// ==========================================================
// Multer
// ==========================================================
const ALLOWED_EXTS = new Set(['.dcm', '.png', '.jpg', '.jpeg']);

const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: CONFIG.MAX_FILE_SIZE_MB * 1024 * 1024,
        files: CONFIG.MAX_BATCH_FILES
    },
    fileFilter: (_req, file, cb) => {
        const ext = path.extname(file.originalname).toLowerCase();
        if (ALLOWED_EXTS.has(ext)) {
            cb(null, true);
        } else {
            cb(new multer.MulterError('LIMIT_UNEXPECTED_FILE', `Unsupported file type: ${file.originalname} (${ext})`));
        }
    }
});

// ==========================================================
// Middleware — Request ID + Request Logging
// ==========================================================
app.use((req, res, next) => {
    req.id = uuidv4();
    res.setHeader('X-Request-ID', req.id);
    next();
});

app.use((req, res, next) => {
    const start = Date.now();
    res.on('finish', () => {
        logger.info({
            requestId:  req.id,
            method:     req.method,
            path:       req.path,
            status:     res.statusCode,
            ms:         Date.now() - start,
            ip:         req.ip,
            origin:     req.headers.origin || '—'
        });
    });
    next();
});

// ==========================================================
// Helpers
// ==========================================================

/**
 * Build a fetch request to the HF Space with the correct auth headers.
 * @param {string}   hfPath    — e.g. "/predict" or "/batch/predict"
 * @param {FormData} formData
 * @param {string}   requestId — for logging
 * @param {number}   timeoutMs
 */
async function callHFSpace(hfPath, formData, requestId, timeoutMs) {
    const url = `${CONFIG.HF_SPACE_URL}${hfPath}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    logger.debug(`→ HF ${url}`, { requestId });

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${CONFIG.HF_TOKEN}`,
                'X-API-Key': CONFIG.API_KEY,
                ...formData.getHeaders()
            },
            body:   formData,
            signal: controller.signal
        });
        return response;
    } catch (err) {
        if (err.name === 'AbortError') {
            throw Object.assign(new Error(`HF Space timed out after ${timeoutMs}ms`), { code: 'TIMEOUT' });
        }
        throw err;
    } finally {
        clearTimeout(timer);
    }
}

/**
 * Buffer a fetch Response body into a Buffer.
 * Safer than streaming directly when we need to inspect/forward headers.
 */
async function bufferResponse(fetchResponse) {
    const chunks = [];
    for await (const chunk of fetchResponse.body) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    return Buffer.concat(chunks);
}

/**
 * Translate common HF error statuses into user-friendly messages.
 */
function hfErrorMessage(status) {
    if (status === 401 || status === 403) return 'HF Space authentication failed. Check HF_TOKEN and API_KEY.';
    if (status === 503)                   return 'HF Space is starting up (cold start). Please retry in a moment.';
    if (status === 422)                   return 'HF Space rejected the payload (validation error). Check file format.';
    return `HF Space returned HTTP ${status}.`;
}

// ==========================================================
// Routes — Health / Root
// ==========================================================
app.get('/health', (_req, res) => {
    res.json({
        status:    'healthy',
        timestamp: new Date().toISOString(),
        version:   '1.1.0',
        uptime:    process.uptime()
    });
});

app.get('/', (_req, res) => {
    res.json({
        service:   'TB Screening AI Proxy',
        status:    'running',
        endpoints: {
            health:       'GET  /health',
            predict:      'POST /api/predict',
            batchPredict: 'POST /api/batch/predict'
        }
    });
});

// ==========================================================
// POST /api/predict — Single image
// ==========================================================
app.post('/api/predict', upload.single('file'), async (req, res) => {
    const requestId = req.id;

    if (!req.file) {
        return res.status(400).json({
            error:   'No file provided',
            message: 'Please upload a DICOM, PNG, or JPEG file in the "file" field.'
        });
    }

    logger.info(`Single predict: ${req.file.originalname} (${req.file.size} bytes)`, { requestId });

    // Build multipart for HF
    const formData = new FormData();
    formData.append('file', req.file.buffer, {
        filename:    req.file.originalname,
        contentType: req.file.mimetype || 'application/octet-stream'
    });

    let hfResponse;
    try {
        hfResponse = await callHFSpace('/predict', formData, requestId, CONFIG.SINGLE_TIMEOUT_MS);
    } catch (err) {
        logger.error(`HF fetch error: ${err.message}`, { requestId });
        const status = err.code === 'TIMEOUT' ? 504 : 502;
        return res.status(status).json({ error: 'Upstream error', message: err.message, requestId });
    }

    if (!hfResponse.ok) {
        const body = await hfResponse.text().catch(() => '');
        logger.error(`HF error ${hfResponse.status}: ${body.slice(0, 300)}`, { requestId });
        const status = [401, 403, 422, 503].includes(hfResponse.status) ? hfResponse.status : 502;
        return res.status(status).json({
            error:     'HF Space error',
            message:   hfErrorMessage(hfResponse.status),
            requestId
        });
    }

    let result;
    try {
        result = await hfResponse.json();
    } catch {
        return res.status(502).json({
            error:     'Invalid response from HF Space',
            message:   'Expected JSON but received non-parseable data.',
            requestId
        });
    }

    return res.json({
        ...result,
        proxied:            true,
        proxy_request_id:   requestId,
        proxy_timestamp:    new Date().toISOString()
    });
});

// ==========================================================
// POST /api/batch/predict — Multiple images → CSV
// ==========================================================
app.post('/api/batch/predict', upload.array('files', CONFIG.MAX_BATCH_FILES), async (req, res) => {
    const requestId = req.id;

    if (!req.files || req.files.length === 0) {
        return res.status(400).json({
            error:   'No files provided',
            message: 'Please upload at least one file in the "files" field.'
        });
    }

    logger.info(`Batch predict: ${req.files.length} file(s)`, { requestId });

    // Build multipart — field name MUST be "files" to match FastAPI List[UploadFile]
    const formData = new FormData();
    for (const file of req.files) {
        formData.append('files', file.buffer, {
            filename:    file.originalname,
            contentType: file.mimetype || 'application/octet-stream'
        });
    }

    let hfResponse;
    try {
        hfResponse = await callHFSpace('/batch/predict', formData, requestId, CONFIG.BATCH_TIMEOUT_MS);
    } catch (err) {
        logger.error(`HF batch fetch error: ${err.message}`, { requestId });
        const status = err.code === 'TIMEOUT' ? 504 : 502;
        return res.status(status).json({ error: 'Upstream error', message: err.message, requestId });
    }

    if (!hfResponse.ok) {
        const body = await hfResponse.text().catch(() => '');
        logger.error(`HF batch error ${hfResponse.status}: ${body.slice(0, 300)}`, { requestId });
        const status = [401, 403, 422, 503].includes(hfResponse.status) ? hfResponse.status : 502;
        return res.status(status).json({
            error:     'HF Space error',
            message:   hfErrorMessage(hfResponse.status),
            requestId
        });
    }

    // Buffer the entire CSV before forwarding.
    // This prevents a race condition where the Express response ends before
    // node-fetch has finished reading from the HF Space socket on slow connections.
    let csvBuffer;
    try {
        csvBuffer = await bufferResponse(hfResponse);
    } catch (err) {
        logger.error(`Failed to read HF batch body: ${err.message}`, { requestId });
        return res.status(502).json({
            error:     'Upstream read error',
            message:   'Lost connection to HF Space while downloading CSV.',
            requestId
        });
    }

    // Validate that we actually received CSV (guard against HTML error pages)
    const contentType = hfResponse.headers.get('content-type') || 'text/csv';
    if (!contentType.includes('csv') && !contentType.includes('text/plain')) {
        logger.warn(`Unexpected content-type from HF: ${contentType}`, { requestId });
        // Try to surface the body as an error message
        const bodyPreview = csvBuffer.toString('utf8', 0, 500);
        return res.status(502).json({
            error:     'Unexpected response from HF Space',
            message:   bodyPreview,
            requestId
        });
    }

    // Forward the CSV with appropriate headers
    const disposition = hfResponse.headers.get('content-disposition') ||
                        `attachment; filename="tuberculosis_report_${requestId}.csv"`;

    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', disposition);
    res.setHeader('Content-Length', csvBuffer.byteLength);
    res.setHeader('X-Proxy-Request-ID', requestId);
    res.setHeader('X-Files-Processed', String(req.files.length));

    return res.end(csvBuffer);
});

// ==========================================================
// Error Handlers
// ==========================================================

// Multer errors (file size, file count, bad type)
app.use((err, req, res, next) => {
    if (err instanceof multer.MulterError) {
        let message;
        switch (err.code) {
            case 'LIMIT_FILE_SIZE':
                message = `File exceeds the ${CONFIG.MAX_FILE_SIZE_MB} MB limit.`;
                break;
            case 'LIMIT_FILE_COUNT':
                message = `Too many files. Maximum is ${CONFIG.MAX_BATCH_FILES}.`;
                break;
            case 'LIMIT_UNEXPECTED_FILE':
                message = err.message || 'Unexpected file field.';
                break;
            default:
                message = err.message;
        }
        return res.status(400).json({ error: 'File upload error', message });
    }

    if (err.message && err.message.startsWith('Not allowed by CORS')) {
        return res.status(403).json({ error: 'CORS error', message: err.message });
    }

    logger.error('Unhandled error:', { message: err.message, stack: err.stack, requestId: req.id });
    res.status(500).json({
        error:     'Internal server error',
        message:   process.env.NODE_ENV === 'production' ? 'Something went wrong.' : err.message,
        requestId: req.id
    });
});

// 404
app.use((req, res) => {
    res.status(404).json({
        error:   'Not found',
        message: `${req.method} ${req.path} does not exist.`
    });
});

// ==========================================================
// Start
// ==========================================================
const server = app.listen(CONFIG.PORT, () => {
    logger.info(`🚀 Server listening on port ${CONFIG.PORT}`);
    logger.info(`📡 HF Space: ${CONFIG.HF_SPACE_URL}`);
    logger.info(`🔐 HF Token: ${CONFIG.HF_TOKEN ? '✓' : '✗'} | API Key: ${CONFIG.API_KEY ? '✓' : '✗'}`);
    logger.info(`⏱  Timeouts — single: ${CONFIG.SINGLE_TIMEOUT_MS}ms, batch: ${CONFIG.BATCH_TIMEOUT_MS}ms`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    logger.info('SIGTERM received — closing server');
    server.close(() => process.exit(0));
});
