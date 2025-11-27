import React from 'react';
import { Navbar, Container, Nav } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import { FaLeaf } from 'react-icons/fa';

const NavigationBar = () => {
    return (
        <Navbar expand="lg" className="navbar-custom" variant="dark">
            <Container>
                <Navbar.Brand as={Link} to="/">
                    <FaLeaf className="me-2" /> Crop Doctor
                </Navbar.Brand>
                <Navbar.Toggle aria-controls="basic-navbar-nav" />
                <Navbar.Collapse id="basic-navbar-nav">
                    <Nav className="ms-auto">
                        <Nav.Link as={Link} to="/">Home</Nav.Link>
                        <Nav.Link as={Link} to="/how-it-works">How it Works</Nav.Link>
                    </Nav>
                </Navbar.Collapse>
            </Container>
        </Navbar>
    );
};

export default NavigationBar;
