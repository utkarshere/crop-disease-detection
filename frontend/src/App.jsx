import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NavigationBar from './components/Navbar';
import Home from './components/Home';
import HowItWorks from './components/HowItWorks';

function App() {
  return (
    <Router>
      <div className="d-flex flex-column min-vh-100">
        <NavigationBar />
        <div className="flex-grow-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/how-it-works" element={<HowItWorks />} />
          </Routes>
        </div>
        <footer className="bg-dark text-white text-center py-3 mt-auto">
          <small>&copy; 2025 Crop Doctor. AI-Powered Crop Disease Detection.</small> <br></br>
          <small>Made by Manav Juneja & Utkarsh Dubey</small>
        </footer>
      </div>
    </Router>
  );
}

export default App;
