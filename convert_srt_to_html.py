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
            <p>Droplet-Film Theory Development Project</p>
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
            <div class="sidebar">
                <h3>Contents</h3>
                <ul>
                    <li><a href="#introduction">Introduction</a></li>
                    <li><a href="#installation">Installation</a></li>
                    <li><a href="#quick-start">Quick Start</a></li>
                    <li><a href="#data-format">Data Format</a></li>
                    <li><a href="#running">Running DFT Development</a></li>
                    <li><a href="#troubleshooting">Troubleshooting</a></li>
                    <li><a href="#examples">Examples</a></li>
                    <li><a href="#performance">Performance</a></li>
                    <li><a href="#how-to">How-to Guides</a></li>
                    <li><a href="#tutorials">Tutorial Scripts</a></li>
                </ul>
            </div>
            
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
    """Format content for HTML display with essay-like structure"""
    lines = content.split('\n')
    formatted_lines = []
    in_code_block = False
    in_list = False
    current_paragraph = []
    
    def flush_paragraph():
        nonlocal current_paragraph
        if current_paragraph:
            paragraph_text = ' '.join(current_paragraph)
            # Handle bold text
            paragraph_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', paragraph_text)
            # Handle links
            paragraph_text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', paragraph_text)
            formatted_lines.append(f'<p class="paragraph">{paragraph_text}</p>')
            current_paragraph = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            flush_paragraph()
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            continue
        
        # Handle code blocks
        if line.startswith('```'):
            flush_paragraph()
            if not in_code_block:
                formatted_lines.append('<pre><code>')
                in_code_block = True
            else:
                formatted_lines.append('</code></pre>')
                in_code_block = False
            continue
        
        if in_code_block:
            formatted_lines.append(f'<span class="code-line">{line}</span>')
            continue
        
        # Handle numbered sections (e.g., "1. Introduction", "2.1. Overview")
        if re.match(r'^\d+\.', line) or re.match(r'^\d+\.\d+\.', line):
            flush_paragraph()
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            if re.match(r'^\d+\.\d+\.', line):
                formatted_lines.append(f'<h3 class="subsection">{line}</h3>')
            else:
                formatted_lines.append(f'<h2 class="section">{line}</h2>')
        
        # Handle main headers (User Guide, API Reference, etc.)
        elif line in ['User Guide', 'API Reference', 'Examples and Tutorials', 'Introduction', 'Installation', 'Quick Start', 'Data Format', 'Running DFT Development', 'Troubleshooting', 'Examples', 'Performance', 'How-to Guides', 'Tutorial Scripts']:
            flush_paragraph()
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(f'<h1 class="main-header">{line}</h1>')
        
        # Handle subheaders with asterisks
        elif line.startswith('* ') and len(line) < 100:
            flush_paragraph()
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            clean_line = line[2:].strip()
            formatted_lines.append(f'<h4 class="subheader">{clean_line}</h4>')
        
        # Handle code snippets (lines starting with specific patterns)
        elif line.startswith(('python', 'bash', 'pip install', 'git clone', 'cd ', 'python -c', 'import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except:', 'with ')):
            flush_paragraph()
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(f'<div class="code-snippet">{line}</div>')
        
        # Handle bullet points - convert to flowing text
        elif line.startswith('- ') or line.startswith('* '):
            if not in_list:
                in_list = True
                formatted_lines.append('<ul class="flowing-list">')
            clean_line = line[2:].strip()
            formatted_lines.append(f'<li class="flowing-item">{clean_line}</li>')
        
        # Handle numbered lists - convert to flowing text
        elif re.match(r'^\d+\. ', line):
            if not in_list:
                in_list = True
                formatted_lines.append('<ul class="flowing-list">')
            clean_line = re.sub(r'^\d+\. ', '', line)
            formatted_lines.append(f'<li class="flowing-item">{clean_line}</li>')
        
        # Regular paragraphs - accumulate text for better flow
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            current_paragraph.append(line)
    
    # Flush any remaining paragraph
    flush_paragraph()
    if in_list:
        formatted_lines.append('</ul>')
    
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
    """Create the CSS stylesheet with LAMMPS-style design"""
    css_content = """/* DFT Documentation Styles - LAMMPS Inspired */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #ffffff;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    background: white;
    min-height: 100vh;
}

header {
    background: #2c3e50;
    color: white;
    padding: 1.5rem 2rem;
    border-bottom: 3px solid #3498db;
}

header h1 {
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
    font-weight: 300;
}

header p {
    font-size: 1rem;
    opacity: 0.9;
    margin-bottom: 0;
}

nav {
    margin-top: 1rem;
    display: flex;
    justify-content: flex-start;
    flex-wrap: wrap;
    gap: 0.5rem;
}

nav a {
    color: white;
    text-decoration: none;
    padding: 0.4rem 0.8rem;
    border-radius: 3px;
    transition: background-color 0.3s;
    font-size: 0.9rem;
}

nav a:hover {
    background-color: #3498db;
}

main {
    padding: 2rem;
    display: flex;
    gap: 2rem;
}

.sidebar {
    width: 250px;
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 5px;
    height: fit-content;
    position: sticky;
    top: 2rem;
}

.sidebar h3 {
    color: #2c3e50;
    margin-bottom: 1rem;
    font-size: 1.1rem;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
}

.sidebar ul {
    list-style: none;
    margin: 0;
    padding: 0;
}

.sidebar li {
    margin-bottom: 0.3rem;
}

.sidebar a {
    color: #555;
    text-decoration: none;
    font-size: 0.9rem;
    padding: 0.2rem 0;
    display: block;
    transition: color 0.3s;
}

.sidebar a:hover {
    color: #3498db;
}

.content {
    flex: 1;
    max-width: 800px;
}

/* LAMMPS-style headers */
.main-header {
    color: #2c3e50;
    font-size: 1.8rem;
    margin: 2rem 0 1.5rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #3498db;
    font-weight: 400;
}

.section {
    color: #2c3e50;
    font-size: 1.4rem;
    margin: 2rem 0 1rem 0;
    padding-left: 0.5rem;
    border-left: 4px solid #3498db;
    font-weight: 400;
}

.subsection {
    color: #34495e;
    font-size: 1.2rem;
    margin: 1.5rem 0 0.8rem 0;
    padding-left: 1rem;
    font-weight: 400;
}

.subheader {
    color: #34495e;
    font-size: 1rem;
    margin: 1rem 0 0.5rem 0;
    font-weight: 500;
}

.paragraph {
    margin-bottom: 1rem;
    text-align: left;
    line-height: 1.7;
}

/* Code styling */
.code-snippet {
    background: #f4f4f4;
    border: 1px solid #ddd;
    border-left: 4px solid #3498db;
    padding: 0.8rem 1rem;
    margin: 1rem 0;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    overflow-x: auto;
    border-radius: 3px;
}

pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    border-left: 4px solid #3498db;
    padding: 1rem;
    margin: 1rem 0;
    overflow-x: auto;
    border-radius: 3px;
}

code {
    background: #f4f4f4;
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
}

/* Lists */
.flowing-list {
    margin: 1rem 0;
    padding-left: 0;
    list-style: none;
}

.flowing-item {
    margin-bottom: 0.8rem;
    padding-left: 1.5rem;
    position: relative;
    line-height: 1.6;
}

.flowing-item:before {
    content: "•";
    color: #3498db;
    font-weight: bold;
    position: absolute;
    left: 0;
    top: 0;
}

.flowing-item:last-child {
    margin-bottom: 0;
}

/* Links */
a {
    color: #3498db;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
}

th, td {
    border: 1px solid #ddd;
    padding: 0.8rem;
    text-align: left;
}

th {
    background-color: #f8f9fa;
    font-weight: 600;
}

/* Cards for examples */
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 5px;
    border-left: 4px solid #3498db;
    transition: transform 0.3s, box-shadow 0.3s;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.card h3 {
    margin-top: 0;
    color: #2c3e50;
}

.card h3 a {
    color: #2c3e50;
    text-decoration: none;
}

.card h3 a:hover {
    color: #3498db;
    text-decoration: underline;
}

/* Footer */
footer {
    background: #2c3e50;
    color: white;
    text-align: center;
    padding: 1.5rem;
    margin-top: 3rem;
    border-top: 3px solid #3498db;
}

footer p {
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}

/* Responsive design */
@media (max-width: 768px) {
    main {
        flex-direction: column;
        padding: 1rem;
    }
    
    .sidebar {
        width: 100%;
        position: static;
        margin-bottom: 2rem;
    }
    
    .main-header {
        font-size: 1.5rem;
    }
    
    .section {
        font-size: 1.2rem;
    }
    
    .subsection {
        font-size: 1.1rem;
    }
    
    nav {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .card-grid {
        grid-template-columns: 1fr;
    }
}

/* Print styles */
@media print {
    .sidebar {
        display: none;
    }
    
    .container {
        max-width: none;
    }
    
    .main-header, .section, .subsection {
        page-break-after: avoid;
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
