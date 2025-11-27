import React from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import { FaSearch, FaDatabase, FaRobot } from 'react-icons/fa';

const HowItWorks = () => {
    return (
        <Container className="py-5">
            <h1 className="text-center mb-5 fw-bold">How It Works</h1>

            <Row className="g-4">
                <Col md={4}>
                    <Card className="h-100 border-0 shadow-sm text-center p-4">
                        <Card.Body>
                            <div className="text-primary mb-3">
                                <FaSearch size={50} />
                            </div>
                            <Card.Title>1. Image Analysis</Card.Title>
                            <Card.Text>
                                Our advanced AI model (EfficientNet) scans your uploaded leaf image to identify visual patterns associated with various crop diseases.
                            </Card.Text>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={4}>
                    <Card className="h-100 border-0 shadow-sm text-center p-4">
                        <Card.Body>
                            <div className="text-success mb-3">
                                <FaDatabase size={50} />
                            </div>
                            <Card.Title>2. Knowledge Retrieval</Card.Title>
                            <Card.Text>
                                Once a disease is identified, the system searches our extensive knowledge base (RAG) to find the most relevant treatment and prevention information.
                            </Card.Text>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={4}>
                    <Card className="h-100 border-0 shadow-sm text-center p-4">
                        <Card.Body>
                            <div className="text-warning mb-3">
                                <FaRobot size={50} />
                            </div>
                            <Card.Title>3. Expert Advice</Card.Title>
                            <Card.Text>
                                Google's Gemini AI acts as an expert agronomist, synthesizing the technical data into simple, actionable advice for farmers.
                            </Card.Text>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            <div className="mt-5 p-5 bg-light rounded-3">
                <h3>About the Technology</h3>
                <p>
                    This application combines <strong>Computer Vision</strong> for accurate disease detection with <strong>Retrieval-Augmented Generation (RAG)</strong> and <strong>Large Language Models (LLMs)</strong> to provide context-aware, expert-level recommendations.
                </p>
            </div>
        </Container>
    );
};

export default HowItWorks;
