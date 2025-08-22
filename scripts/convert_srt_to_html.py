#!/usr/bin/env python3
"""
SRT to HTML Converter for Documentation
Converts SRT subtitle files to clean HTML documentation
"""

import os
import re
import glob
from pathlib import Path

def clean_srt_content(content):
    """Clean SRT content by removing timing and numbering"""
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip empty lines, timing lines, and segment numbers
        if (line.strip() and 
            not re.match(r'^\d+$', line.strip()) and  # Skip segment numbers
            not re.match(r'^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}$', line.strip()) and  # Skip timing
            not re.match(r'^\s*$', line)):  # Skip empty lines
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def srt_to_html(srt_content, title):
    """Convert SRT content to HTML"""
    cleaned_content = clean_srt_content(srt_content)
    
    # Split into sections
    sections = cleaned_content.split('\n\n')
    
    html_parts = []
    
    for section in sections:
        if section.strip():
            lines = section.strip().split('\n')
            if lines:
                # First line is usually the section title
                section_title = lines[0].strip()
                section_content = lines[1:] if len(lines) > 1 else []
                
                # Create HTML section
                html_section = f'<section class="doc-section">\n'
                html_section += f'  <h2>{section_title}</h2>\n'
                
                if section_content:
                    # Process content lines
                    content_html = []
                    for line in section_content:
                        line = line.strip()
                        if line:
                            if line.startswith('- '):
                                # List item
                                content_html.append(f'    <li>{line[2:]}</li>')
                            elif line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. ') or line.startswith('4. ') or line.startswith('5. '):
                                # Numbered list item
                                content_html.append(f'    <li>{line[3:]}</li>')
                            elif line.startswith('□'):
                                # Checkbox item
                                content_html.append(f'    <li><input type="checkbox" disabled> {line[1:]}</li>')
                            elif line.startswith('#'):
                                # Code comment
                                content_html.append(f'    <code class="comment">{line}</code><br>')
                            elif line.startswith('import ') or line.startswith('from ') or line.startswith('def ') or line.startswith('class '):
                                # Code
                                content_html.append(f'    <code class="code">{line}</code><br>')
                            elif line.startswith('pip install') or line.startswith('python ') or line.startswith('conda '):
                                # Command
                                content_html.append(f'    <code class="command">{line}</code><br>')
                            else:
                                # Regular text
                                content_html.append(f'    <p>{line}</p>')
                    
                    if content_html:
                        # Check if we have list items
                        if any('<li>' in item for item in content_html):
                            html_section += '  <ul>\n'
                            for item in content_html:
                                if '<li>' in item:
                                    html_section += f'{item}\n'
                                else:
                                    html_section += f'  {item}\n'
                            html_section += '  </ul>\n'
                        else:
                            for item in content_html:
                                html_section += f'  {item}\n'
                
                html_section += '</section>\n'
                html_parts.append(html_section)
    
    # Combine into full HTML document
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - DFT Documentation</title>
    <link rel="stylesheet" href="../assets/style.css">
</head>
<body>
    <div class="container">
        <header class="doc-header">
            <h1>{title}</h1>
            <nav class="doc-nav">
                <a href="../index.html">Home</a>
                <a href="main_documentation.html">Overview</a>
                <a href="installation_guide.html">Installation</a>
                <a href="usage_examples.html">Usage</a>
                <a href="api_reference.html">API</a>
                <a href="troubleshooting.html">Troubleshooting</a>
            </nav>
        </header>
        
        <main class="doc-content">
{''.join(html_parts)}
        </main>
        
        <footer class="doc-footer">
            <p>Droplet-Film Theory Development - Scripts 2.0</p>
        </footer>
    </div>
</body>
</html>'''
    
    return html

def main():
    """Main conversion function"""
    # Create output directory
    output_dir = Path('docs/html')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all SRT files
    srt_files = glob.glob('*.srt')
    
    # Conversion mapping
    conversions = {
        'DOCUMENTATION.srt': 'main_documentation.html',
        'INSTALLATION_GUIDE.srt': 'installation_guide.html',
        'USAGE_EXAMPLES.srt': 'usage_examples.html',
        'API_REFERENCE.srt': 'api_reference.html',
        'TROUBLESHOOTING.srt': 'troubleshooting.html'
    }
    
    for srt_file, html_file in conversions.items():
        if os.path.exists(srt_file):
            print(f"Converting {srt_file} to {html_file}...")
            
            # Read SRT content
            with open(srt_file, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            
            # Extract title from first non-empty line
            lines = srt_content.split('\n')
            title = "Documentation"
            for line in lines:
                if line.strip() and not re.match(r'^\d+$', line.strip()) and not re.match(r'^\d{2}:\d{2}:\d{2}', line.strip()):
                    title = line.strip()
                    break
            
            # Convert to HTML
            html_content = srt_to_html(srt_content, title)
            
            # Write HTML file
            output_path = output_dir / html_file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"Created {output_path}")
        else:
            print(f"Warning: {srt_file} not found")
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
