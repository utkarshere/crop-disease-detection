import React, { useState } from 'react';
import { Container, Row, Col, Card, Button, Spinner, Alert, Form } from 'react-bootstrap';
import axios from 'axios';
import { FaCloudUploadAlt, FaCheckCircle, FaExclamationTriangle, FaRedo } from 'react-icons/fa';
import ReactMarkdown from 'react-markdown';

const Home = () => {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            if (selectedFile.size > 5 * 1024 * 1024) {
                setError("File size exceeds 5MB limit.");
                return;
            }
            setFile(selectedFile);
            setPreview(URL.createObjectURL(selectedFile));
            setError(null);
            setResult(null);
        }
    };

    const handleReset = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file) {
            setError("Please select an image first.");
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://localhost:5000/api/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setResult(response.data);
        } catch (err) {
            console.error(err);
            setError("Failed to analyze image. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <div className="hero-section">
                <Container>
                    <h1 className="display-4 fw-bold">Smart Crop Disease Detection</h1>
                    <p className="lead">Upload a leaf image to get instant diagnosis and expert treatment advice.</p>
                </Container>
            </div>

            <Container className="mb-5">
                {!result ? (
                    <Row className="justify-content-center">
                        <Col md={8} lg={6}>
                            <Card className="upload-card">
                                <Card.Body>
                                    <h3 className="mb-4">Upload Leaf Image</h3>
                                    <Form onSubmit={handleSubmit}>
                                        <div className="mb-4 text-center">
                                            {preview ? (
                                                <img src={preview} alt="Preview" className="img-fluid rounded shadow-sm" style={{ maxHeight: '300px' }} />
                                            ) : (
                                                <div className="p-5 border rounded bg-light text-muted">
                                                    <FaCloudUploadAlt size={50} className="mb-3" />
                                                    <p>Click to select or drag and drop image here</p>
                                                    <small>(Max 5MB)</small>
                                                </div>
                                            )}
                                        </div>

                                        <Form.Group controlId="formFile" className="mb-3">
                                            <Form.Control type="file" accept="image/*" onChange={handleFileChange} />
                                        </Form.Group>

                                        {error && <Alert variant="danger">{error}</Alert>}

                                        <div className="d-grid gap-2">
                                            <Button
                                                variant="primary"
                                                type="submit"
                                                className="btn-primary-custom"
                                                disabled={loading || !file}
                                            >
                                                {loading ? <Spinner animation="border" size="sm" /> : 'Analyze Image'}
                                            </Button>
                                            {preview && !loading && (
                                                <Button variant="outline-secondary" onClick={handleReset}>
                                                    Reset
                                                </Button>
                                            )}
                                        </div>
                                    </Form>
                                </Card.Body>
                            </Card>
                        </Col>
                    </Row>
                ) : (
                    <Row className="justify-content-center mt-4">
                        <Col md={12}>
                            <Card className="result-card">
                                <div className="result-header d-flex justify-content-between align-items-center">
                                    <h2 className="m-0"><FaCheckCircle className="me-2" /> Diagnosis Result</h2>
                                    <Button variant="light" size="sm" onClick={handleReset}>
                                        <FaRedo className="me-1" /> Analyze Another
                                    </Button>
                                </div>
                                <Card.Body className="p-4">
                                    <Row>
                                        <Col md={4} className="border-end">
                                            <div className="text-center mb-4">
                                                <img src={preview} alt="Analyzed Leaf" className="img-fluid rounded shadow-sm" style={{ maxHeight: '250px' }} />
                                            </div>

                                            <h4 className="text-secondary">Predicted Disease</h4>
                                            <p className="display-6 text-dark fw-bold mb-4">{result.predicted_disease}</p>

                                            <h5 className="text-secondary">Confidence</h5>
                                            <div className="progress mb-4" style={{ height: '25px' }}>
                                                <div
                                                    className="progress-bar bg-success"
                                                    role="progressbar"
                                                    style={{ width: result.confidence }}
                                                    aria-valuenow={parseFloat(result.confidence)}
                                                    aria-valuemin="0"
                                                    aria-valuemax="100"
                                                >
                                                    {result.confidence}
                                                </div>
                                            </div>

                                            <div className="d-grid">
                                                <Button variant="outline-primary" onClick={handleReset}>
                                                    <FaRedo className="me-2" /> Analyze Another Image
                                                </Button>
                                            </div>
                                        </Col>
                                        <Col md={8}>
                                            {result.is_healthy ? (
                                                <Alert variant="success" className="h-100 d-flex flex-column justify-content-center align-items-center text-center">
                                                    <FaCheckCircle size={50} className="mb-3" />
                                                    <h4 className="alert-heading">Healthy Plant!</h4>
                                                    <p className="lead">Your plant looks healthy. Keep up the good work!</p>
                                                    <hr />
                                                    <p className="mb-0">Continue monitoring for any changes.</p>
                                                </Alert>
                                            ) : (
                                                <div className="expert-advice h-100">
                                                    <h4 className="text-success mb-3 border-bottom pb-2">Expert Recommendation</h4>
                                                    {result.expert_recommendation ? (
                                                        <div className="markdown-content">
                                                            <ReactMarkdown>{result.expert_recommendation}</ReactMarkdown>
                                                        </div>
                                                    ) : (
                                                        <p className="text-muted">
                                                            {result.raw_treatment || "No specific treatment found."}
                                                        </p>
                                                    )}
                                                </div>
                                            )}
                                        </Col>
                                    </Row>
                                    {result.note && (
                                        <Alert variant="warning" className="mt-4">
                                            <FaExclamationTriangle className="me-2" />
                                            {result.note}
                                        </Alert>
                                    )}
                                </Card.Body>
                            </Card>
                        </Col>
                    </Row>
                )}
            </Container>
        </div>
    );
};

export default Home;
