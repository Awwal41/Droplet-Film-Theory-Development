#!/usr/bin/env python3
"""
SRT to HTML Converter for DFT Documentation
Converts SRT subtitle files to HTML documentation pages
"""

import os
import re
from pathlib import Path
from datetime import datetime

def clean_srt_content(content):
    """Clean and format SRT content for HTML conversion"""
    # Remove SRT timing markers (lines with --> or just numbers)
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Skip timing lines (contain --> or are just numbers)
        if '-->' in line or (line.strip().isdigit() and len(line.strip()) <= 3):
            continue
            
        # Skip subtitle index numbers
        if line.strip().isdigit() and len(line.strip()) <= 3:
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def convert_srt_to_html(srt_file, output_dir):
    """Convert a single SRT file to HTML"""
    print(f"Converting {srt_file}...")
    
    # Read SRT file
    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Clean the content
    cleaned_content = clean_srt_content(content)
    
    # Extract title from first line
    lines = cleaned_content.split('\n')
    title = lines[0].strip() if lines else "Documentation"
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - DFT Documentation</title>
    <link rel="stylesheet" href="../assets/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <nav>
                <a href="../index.html">Home</a>
                <a href="documentation.html">Documentation</a>
                <a href="installation_guide.html">Installation</a>
                <a href="usage_examples.html">Usage</a>
                <a href="api_reference.html">API Reference</a>
                <a href="troubleshooting.html">Troubleshooting</a>
            </nav>
        </header>
        
        <main>
            <div class="content">
{format_content_for_html(cleaned_content)}
            </div>
        </main>
        
        <footer>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Droplet-Film Theory Development Project</p>
        </footer>
    </div>
</body>
</html>"""
    
    # Write HTML file
    output_file = output_dir / f"{srt_file.stem.lower()}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Created {output_file}")
    return output_file

def format_content_for_html(content):
    """Format content for HTML display"""
    lines = content.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append('<br>')
            continue
            
        # Check for headers (lines that are short and don't end with punctuation)
        if len(line) < 50 and not line.endswith(('.', ':', ';', ',')):
            if line.isupper() or line.startswith(('Project', 'Core', 'Key', 'Quick', 'File', 'Documentation')):
                formatted_lines.append(f'<h2>{line}</h2>')
            else:
                formatted_lines.append(f'<h3>{line}</h3>')
        else:
            # Regular paragraph
            formatted_lines.append(f'<p>{line}</p>')
    
    return '\n'.join(formatted_lines)

def create_main_index(output_dir):
    """Create the main index.html file"""
    index_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DFT Documentation - Home</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Droplet-Film Theory Development</h1>
            <p>Physics-Informed Machine Learning for Gas Well Liquid Loading Prediction</p>
        </header>
        
        <main>
            <div class="content">
                <h2>Welcome to DFT Documentation</h2>
                <p>This documentation provides comprehensive information about the Droplet-Film Theory Development project, which implements physics-informed machine learning models for predicting liquid loading in gas wells.</p>
                
                <div class="card-grid">
                    <div class="card">
                        <h3><a href="html/documentation.html">Main Documentation</a></h3>
                        <p>Comprehensive overview of the project, its components, and implementation details.</p>
                    </div>
                    
                    <div class="card">
                        <h3><a href="html/installation_guide.html">Installation Guide</a></h3>
                        <p>Step-by-step instructions for setting up the development environment.</p>
                    </div>
                    
                    <div class="card">
                        <h3><a href="html/usage_examples.html">Usage Examples</a></h3>
                        <p>Practical examples and tutorials for using the DFT models.</p>
                    </div>
                    
                    <div class="card">
                        <h3><a href="html/api_reference.html">API Reference</a></h3>
                        <p>Detailed documentation of all functions, classes, and methods.</p>
                    </div>
                    
                    <div class="card">
                        <h3><a href="html/troubleshooting.html">Troubleshooting</a></h3>
                        <p>Common issues and solutions for using the DFT development tools.</p>
                    </div>
                </div>
                
                <h2>Quick Start</h2>
                <ol>
                    <li>Clone the repository</li>
                    <li>Navigate to the scripts_2.0 directory</li>
                    <li>Install dependencies: <code>pip install -r requirements.txt</code></li>
                    <li>Run the Jupyter notebooks to explore the models</li>
                </ol>
            </div>
        </main>
        
        <footer>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Droplet-Film Theory Development Project</p>
        </footer>
    </div>
</body>
</html>"""
    
    index_file = output_dir / "index.html"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"✅ Created {index_file}")

def create_css_file(output_dir):
    """Create the CSS stylesheet"""
    css_content = """/* DFT Documentation Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f5f5f5;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    background: white;
    min-height: 100vh;
    box-shadow: 0 0 20px rgba(0,0,0,0.1);
}

header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    text-align: center;
}

header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

header p {
    font-size: 1.2rem;
    opacity: 0.9;
}

nav {
    margin-top: 1.5rem;
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 1rem;
}

nav a {
    color: white;
    text-decoration: none;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    transition: background-color 0.3s;
}

nav a:hover {
    background-color: rgba(255,255,255,0.2);
}

main {
    padding: 2rem;
}

.content {
    max-width: 800px;
    margin: 0 auto;
}

h2 {
    color: #667eea;
    margin: 2rem 0 1rem 0;
    border-bottom: 2px solid #667eea;
    padding-bottom: 0.5rem;
}

h3 {
    color: #764ba2;
    margin: 1.5rem 0 0.5rem 0;
}

p {
    margin-bottom: 1rem;
    text-align: justify;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    transition: transform 0.3s, box-shadow 0.3s;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.card h3 {
    margin-top: 0;
}

.card h3 a {
    color: #667eea;
    text-decoration: none;
}

.card h3 a:hover {
    text-decoration: underline;
}

code {
    background: #e9ecef;
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
}

ol, ul {
    margin-left: 2rem;
    margin-bottom: 1rem;
}

li {
    margin-bottom: 0.5rem;
}

footer {
    background: #343a40;
    color: white;
    text-align: center;
    padding: 1.5rem;
    margin-top: 3rem;
}

footer p {
    margin-bottom: 0.5rem;
}

@media (max-width: 768px) {
    .container {
        margin: 0;
        box-shadow: none;
    }
    
    header {
        padding: 1rem;
    }
    
    header h1 {
        font-size: 2rem;
    }
    
    main {
        padding: 1rem;
    }
    
    nav {
        flex-direction: column;
        align-items: center;
    }
    
    .card-grid {
        grid-template-columns: 1fr;
    }
}"""
    
    # Create assets directory
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    css_file = assets_dir / "style.css"
    with open(css_file, 'w', encoding='utf-8') as f:
        f.write(css_content)
    
    print(f"✅ Created {css_file}")

def main():
    """Main conversion function"""
    print("=" * 60)
    print("DFT SRT to HTML Converter")
    print("=" * 60)
    
    # Define SRT files and their mappings
    srt_files = [
        'DOCUMENTATION.srt',
        'INSTALLATION_GUIDE.srt',
        'USAGE_EXAMPLES.srt',
        'API_REFERENCE.srt',
        'TROUBLESHOOTING.srt'
    ]
    
    # Create output directory
    output_dir = Path('docs/html')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}")
    
    # Convert each SRT file
    converted_files = []
    for srt_file in srt_files:
        srt_path = Path(srt_file)
        if srt_path.exists():
            html_file = convert_srt_to_html(srt_path, output_dir)
            converted_files.append(html_file)
        else:
            print(f"⚠️  Warning: {srt_file} not found, skipping...")
    
    # Create main index and CSS
    create_main_index(Path('docs'))
    create_css_file(Path('docs'))
    
    print("\n" + "=" * 60)
    print(f"✅ Conversion complete! Generated {len(converted_files)} HTML files")
    print("Files created:")
    for file in converted_files:
        print(f"  - {file}")
    print("  - docs/index.html")
    print("  - docs/assets/style.css")
    print("=" * 60)

if __name__ == "__main__":
    main()
