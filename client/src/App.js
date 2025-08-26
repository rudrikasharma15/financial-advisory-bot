import logo from './logo.svg';
import './App.css';

function App() {
  return (
    
    <div className="App">
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-logo">MyBrand</div>
          <ul className="nav-links">
            <li><a href="#home">Home</a></li>
            <li><a href="#features">Features</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#contact">Contact</a></li>
          </ul>
          <div className="nav-toggle" id="mobile-menu">&#9776;</div>
        </div>
      </nav>

      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
     <footer className="footer">
        <div className="footer-container">
          <div className="footer-brand">MyBrand</div>
          <ul className="footer-links">
            <li><a href="#privacy">Privacy Policy</a></li>
            <li><a href="#terms">Terms of Service</a></li>
            <li><a href="#support">Support</a></li>
          </ul>
          <div className="footer-socials">
            <a href="https://facebook.com" target="_blank" rel="noreferrer">🌐</a>
            <a href="https://twitter.com" target="_blank" rel="noreferrer">🐦</a>
            <a href="https://instagram.com" target="_blank" rel="noreferrer">📸</a>
          </div>
        </div>
        <p className="footer-copy">© {new Date().getFullYear()} MyBrand. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
