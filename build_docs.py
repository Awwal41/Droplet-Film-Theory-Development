#!/usr/bin/env python3
"""
Documentation Builder for DFT Project
Builds the documentation site from reStructuredText (.rst) files using Sphinx.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def build_documentation():
    """Build the complete documentation site from RST sources using Sphinx."""
    print("=" * 60)
    print("DFT Documentation Builder (RST/Sphinx)")
    print("=" * 60)

    docs_dir = Path('docs')
    source_dir = docs_dir / 'source'
    build_dir = docs_dir / '_build'

    if not source_dir.exists():
        print("ERROR: docs/source not found. Run srt_to_rst.py first to generate RST from SRT.")
        return False

    required_rst = [
        'installation_guide.rst',
        'usage_examples.rst',
        'api_reference.rst',
        'troubleshooting.rst',
    ]
    missing = [f for f in required_rst if not (source_dir / f).exists()]
    if missing:
        print(f"ERROR: Missing RST files: {missing}")
        return False

    print("OK All required RST files found")

    if build_dir.exists():
        shutil.rmtree(build_dir)

    cmd = [
        sys.executable, '-m', 'sphinx',
        '-b', 'html',
        str(source_dir),
        str(build_dir),
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("ERROR: Sphinx build failed")
        return False

    create_github_pages_files(build_dir)

    print("OK Documentation build complete!")
    print(f"Build directory: {build_dir.absolute()}")
    print("\nFiles created:")
    for root, dirs, files in os.walk(build_dir):
        for file in files:
            rel_path = Path(root).relative_to(build_dir) / file
            print(f"  - {rel_path}")

    return True


def create_github_pages_files(build_dir):
    """Create GitHub Pages specific configuration files."""
    nojekyll_file = build_dir / '.nojekyll'
    nojekyll_file.write_text('')

    readme_content = f"""# DFT Documentation

This directory contains the built documentation for the Droplet-Film Model Development project.

## Source

This documentation is built from reStructuredText (.rst) files in docs/source using Sphinx.

## Last Updated

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    (build_dir / 'README.md').write_text(readme_content, encoding='utf-8')
    print("OK Created GitHub Pages configuration files")


def main():
    """Main build function."""
    success = build_documentation()

    if success:
        print("\n" + "=" * 60)
        print("Documentation build successful!")
        print("\nNext steps:")
        print("1. Commit and push your changes")
        print("2. Enable GitHub Pages in repository settings")
        print("3. Set source to 'GitHub Actions'")
        print("4. Your docs will be automatically deployed!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Documentation build failed!")
        print("Please check the errors above and try again.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
