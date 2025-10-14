#!/usr/bin/env python3
"""
Documentation Builder for DFT Project
Builds the complete documentation site from HTML files
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def build_documentation():
    """Build the complete documentation site"""
    print("=" * 60)
    print("DFT Documentation Builder")
    print("=" * 60)
    
    # Ensure docs directory exists
    docs_dir = Path('docs')
    docs_dir.mkdir(exist_ok=True)
    
    # Check if HTML files exist
    html_dir = docs_dir / 'html'
    if not html_dir.exists():
        print("❌ HTML directory not found. Run convert_srt_to_html.py first.")
        return False
    
    # Check for required files
    required_files = [
        'html/documentation.html',
        'html/installation_guide.html', 
        'html/usage_examples.html',
        'html/api_reference.html',
        'html/troubleshooting.html'
    ]
    
    missing_files = []
    for file in required_files:
        if not (docs_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing HTML files: {missing_files}")
        return False
    
    print("✅ All required HTML files found")
    
    # Create _build directory for GitHub Pages
    build_dir = docs_dir / '_build'
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    
    # Copy HTML files to _build
    html_build_dir = build_dir / 'html'
    shutil.copytree(html_dir, html_build_dir)
    
    # Copy assets
    assets_dir = docs_dir / 'assets'
    if assets_dir.exists():
        assets_build_dir = build_dir / 'assets'
        shutil.copytree(assets_dir, assets_build_dir)
    
    # Copy main index
    index_file = docs_dir / 'index.html'
    if index_file.exists():
        shutil.copy2(index_file, build_dir / 'index.html')
    
    # Create GitHub Pages specific files
    create_github_pages_files(build_dir)
    
    print("✅ Documentation build complete!")
    print(f"Build directory: {build_dir.absolute()}")
    print("\nFiles created:")
    for root, dirs, files in os.walk(build_dir):
        for file in files:
            rel_path = Path(root).relative_to(build_dir) / file
            print(f"  - {rel_path}")
    
    return True

def create_github_pages_files(build_dir):
    """Create GitHub Pages specific configuration files"""
    
    # Create .nojekyll file to disable Jekyll processing
    nojekyll_file = build_dir / '.nojekyll'
    with open(nojekyll_file, 'w') as f:
        f.write('')
    
    # Create README for GitHub Pages
    readme_content = f"""# DFT Documentation

This directory contains the built documentation for the Droplet-Film Theory Development project.

## Generated Files

- `index.html` - Main documentation homepage
- `html/` - Individual documentation pages
- `assets/` - CSS and other resources

## Last Updated

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Source

This documentation is automatically generated from SRT files in the repository root.
"""
    
    readme_file = build_dir / 'README.md'
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ Created GitHub Pages configuration files")

def main():
    """Main build function"""
    success = build_documentation()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 Documentation build successful!")
        print("\nNext steps:")
        print("1. Commit and push your changes")
        print("2. Enable GitHub Pages in repository settings")
        print("3. Set source to 'GitHub Actions'")
        print("4. Your docs will be automatically deployed!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Documentation build failed!")
        print("Please check the errors above and try again.")
        print("=" * 60)

if __name__ == "__main__":
    main()
