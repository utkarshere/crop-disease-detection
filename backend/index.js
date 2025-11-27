const express = require('express');
const cors = require('cors');
const axios = require('axios');
const multer = require('multer');
const FormData = require('form-data');
const fs = require('fs');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';

app.use(cors());
app.use(express.json());

// Configure multer for file uploads
const upload = multer({ dest: 'uploads/' });

// Health check
app.get('/', (req, res) => {
    res.send('Node.js Backend is running');
});

// Proxy to FastAPI categories endpoint
app.get('/api/categories', async (req, res) => {
    try {
        const response = await axios.get(`${FASTAPI_URL}/categories`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching categories:', error.message);
        res.status(500).json({ error: 'Failed to fetch categories from model service' });
    }
});

// Proxy to FastAPI predict endpoint
app.post('/api/predict', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }

    try {
        const formData = new FormData();
        formData.append('file', fs.createReadStream(req.file.path));

        const response = await axios.post(`${FASTAPI_URL}/predict_and_recommend_expert`, formData, {
            headers: {
                ...formData.getHeaders(),
            },
        });

        // Clean up the uploaded file
        fs.unlinkSync(req.file.path);

        res.json(response.data);
    } catch (error) {
        console.error('Error predicting disease:', error.message);
        // Clean up the uploaded file if it exists
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
        res.status(500).json({ error: 'Failed to get prediction from model service' });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
