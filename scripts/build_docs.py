#!/usr/bin/env python3
"""
Documentation Builder
Builds the complete documentation site from HTML files
"""

import os
import shutil
from pathlib import Path

def create_main_index():
    """Create the main index.html file"""
    index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Droplet-Film Theory Development - Documentation</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
    <div class="container">
        <header class="main-header">
            <h1>Droplet-Film Theory Development</h1>
            <p class="subtitle">Scripts 2.0 - Complete Documentation</p>
        </header>
        
        <main class="main-content">
            <section class="hero-section">
                <h2>Welcome to DFT Development</h2>
                <p>This project implements physics-informed machine learning models for predicting liquid loading in gas wells using Droplet-Film Theory.</p>
            </section>
            
            <section class="docs-overview">
                <h2>Documentation Sections</h2>
                <div class="docs-grid">
                    <div class="doc-card">
                        <h3><a href="html/main_documentation.html">Project Overview</a></h3>
                        <p>Complete project overview, core components, and quick start guide.</p>
                    </div>
                    
                    <div class="doc-card">
                        <h3><a href="html/installation_guide.html">Installation Guide</a></h3>
                        <p>Step-by-step installation instructions for all platforms and dependencies.</p>
                    </div>
                    
                    <div class="doc-card">
                        <h3><a href="html/usage_examples.html">Usage Examples</a></h3>
                        <p>Practical code examples and workflows for using the DFT implementation.</p>
                    </div>
                    
                    <div class="doc-card">
                        <h3><a href="html/api_reference.html">API Reference</a></h3>
                        <p>Detailed technical specifications for all classes, methods, and parameters.</p>
                    </div>
                    
                    <div class="doc-card">
                        <h3><a href="html/troubleshooting.html">Troubleshooting</a></h3>
                        <p>Common issues, solutions, and debugging tips for the DFT implementation.</p>
                    </div>
                </div>
            </section>
            
            <section class="quick-start">
                <h2>Quick Start</h2>
                <div class="quick-start-steps">
                    <div class="step">
                        <span class="step-number">1</span>
                        <p>Clone the repository</p>
                        <code>git clone [repository-url]</code>
                    </div>
                    <div class="step">
                        <span class="step-number">2</span>
                        <p>Navigate to scripts_2.0</p>
                        <code>cd scripts_2.0</code>
                    </div>
                    <div class="step">
                        <span class="step-number">3</span>
                        <p>Install dependencies</p>
                        <code>pip install -r requirements.txt</code>
                    </div>
                    <div class="step">
                        <span class="step-number">4</span>
                        <p>Run examples</p>
                        <code>jupyter notebook</code>
                    </div>
                </div>
            </section>
        </main>
        
        <footer class="main-footer">
            <p>&copy; 2024 Droplet-Film Theory Development. All rights reserved.</p>
        </footer>
    </div>
</body>
</html>'''
    
    # Write to docs directory
    docs_dir = Path('docs')
    docs_dir.mkdir(exist_ok=True)
    
    with open(docs_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    print("Created main index.html")

def create_assets_directory():
    """Create assets directory with CSS and other resources"""
    assets_dir = Path('docs/assets')
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Create CSS file
    css_content = '''/* DFT Documentation Styles */
:root {
    --primary-color: #2c3e50;
    --secondary-color: #3498db;
    --accent-color: #e74c3c;
    --text-color: #2c3e50;
    --light-gray: #ecf0f1;
    --border-color: #bdc3c7;
    --code-bg: #f8f9fa;
    --link-color: #3498db;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    line-height: 1.6;
    color: var(--text-color);
    background-color: #fff;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Header Styles */
.main-header, .doc-header {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    padding: 3rem 0;
    text-align: center;
    margin-bottom: 2rem;
}

.main-header h1, .doc-header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    font-weight: 300;
}

.subtitle {
    font-size: 1.2rem;
    opacity: 0.9;
}

/* Navigation */
.doc-nav {
    background: rgba(255, 255, 255, 0.1);
    padding: 1rem;
    border-radius: 8px;
    margin-top: 1rem;
}

.doc-nav a {
    color: white;
    text-decoration: none;
    margin: 0 1rem;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    transition: background-color 0.3s;
}

.doc-nav a:hover {
    background-color: rgba(255, 255, 255, 0.2);
}

/* Main Content */
.main-content, .doc-content {
    margin-bottom: 3rem;
}

.hero-section {
    text-align: center;
    padding: 3rem 0;
    background: var(--light-gray);
    border-radius: 8px;
    margin-bottom: 3rem;
}

.hero-section h2 {
    font-size: 2rem;
    margin-bottom: 1rem;
    color: var(--primary-color);
}

.hero-section p {
    font-size: 1.1rem;
    max-width: 600px;
    margin: 0 auto;
    color: #666;
}

/* Documentation Grid */
.docs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin: 2rem 0;
}

.doc-card {
    background: white;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 2rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s, box-shadow 0.3s;
}

.doc-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.15);
}

.doc-card h3 {
    color: var(--secondary-color);
    margin-bottom: 1rem;
}

.doc-card a {
    color: inherit;
    text-decoration: none;
}

.doc-card a:hover {
    color: var(--accent-color);
}

/* Quick Start Steps */
.quick-start-steps {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
    margin: 2rem 0;
}

.step {
    text-align: center;
    padding: 2rem;
    background: var(--light-gray);
    border-radius: 8px;
    position: relative;
}

.step-number {
    display: inline-block;
    width: 40px;
    height: 40px;
    background: var(--secondary-color);
    color: white;
    border-radius: 50%;
    line-height: 40px;
    font-weight: bold;
    margin-bottom: 1rem;
}

.step code {
    background: var(--code-bg);
    padding: 0.5rem;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    display: block;
    margin-top: 1rem;
}

/* Documentation Sections */
.doc-section {
    margin-bottom: 3rem;
    padding: 2rem;
    background: white;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

.doc-section h2 {
    color: var(--primary-color);
    border-bottom: 2px solid var(--secondary-color);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

.doc-section p {
    margin-bottom: 1rem;
    color: #555;
}

.doc-section ul {
    margin: 1rem 0;
    padding-left: 2rem;
}

.doc-section li {
    margin-bottom: 0.5rem;
}

/* Code Styling */
code {
    background: var(--code-bg);
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
}

.code {
    background: #2d3748;
    color: #e2e8f0;
    padding: 1rem;
    border-radius: 6px;
    display: block;
    margin: 1rem 0;
    overflow-x: auto;
}

.command {
    background: #2d3748;
    color: #68d391;
    padding: 1rem;
    border-radius: 6px;
    display: block;
    margin: 1rem 0;
}

.comment {
    background: #f7fafc;
    color: #718096;
    padding: 0.5rem;
    border-radius: 4px;
    display: inline-block;
    margin: 0.5rem 0;
}

/* Footer */
.main-footer, .doc-footer {
    text-align: center;
    padding: 2rem 0;
    background: var(--light-gray);
    border-top: 1px solid var(--border-color);
    color: #666;
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        padding: 0 15px;
    }
    
    .main-header h1, .doc-header h1 {
        font-size: 2rem;
    }
    
    .docs-grid {
        grid-template-columns: 1fr;
    }
    
    .quick-start-steps {
        grid-template-columns: 1fr;
    }
    
    .doc-nav a {
        display: block;
        margin: 0.5rem 0;
    }
}'''
    
    with open(assets_dir / 'style.css', 'w', encoding='utf-8') as f:
        f.write(css_content)
    
    print("Created assets directory with CSS")

def build_documentation():
    """Build the complete documentation site"""
    print("Building documentation site...")
    
    # Create main index
    create_main_index()
    
    # Create assets
    create_assets_directory()
    
    # Create _build directory structure for GitHub Pages
    build_dir = Path('docs/_build/html')
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy HTML files to build directory
    html_dir = Path('docs/html')
    if html_dir.exists():
        for html_file in html_dir.glob('*.html'):
            shutil.copy2(html_file, build_dir)
    
    # Copy assets to build directory
    assets_dir = Path('docs/assets')
    if assets_dir.exists():
        build_assets = build_dir / 'assets'
        build_assets.mkdir(exist_ok=True)
        for asset_file in assets_dir.glob('*'):
            if asset_file.is_file():
                shutil.copy2(asset_file, build_assets)
    
    # Copy main index to build directory (this will be the root index.html for GitHub Pages)
    main_index = Path('docs/index.html')
    if main_index.exists():
        shutil.copy2(main_index, build_dir)
        print("Main index.html copied to build directory (will be root for GitHub Pages)")
    
    # Also create a copy at the root level for easier access
    root_index = Path('docs/_build/index.html')
    if main_index.exists():
        shutil.copy2(main_index, root_index)
        print("Root index.html created for GitHub Pages")
    
    print("Documentation build complete!")
    print("Files are ready in docs/_build/html/")

if __name__ == "__main__":
    build_documentation()
