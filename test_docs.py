#!/usr/bin/env python3
"""
Test script for the documentation build process.
Verifies that RST sources exist and that Sphinx builds successfully.
"""

import os
import sys
from pathlib import Path


def test_documentation_build():
    """Test the RST/Sphinx documentation build process."""
    print("Testing documentation build process...")

    source_dir = Path('docs/source')
    rst_files = [
        'documentation.rst',
        'installation_guide.rst',
        'usage_examples.rst',
        'api_reference.rst',
        'troubleshooting.rst',
    ]

    if not source_dir.exists():
        print("ERROR: docs/source not found. Run srt_to_rst.py to generate RST files.")
        return False

    missing = [f for f in rst_files if not (source_dir / f).exists()]
    if missing:
        print(f"ERROR: Missing RST files: {missing}")
        return False

    print("OK All RST files found")

    if not (source_dir / 'conf.py').exists() or not (source_dir / 'index.rst').exists():
        print("ERROR: Sphinx conf.py or index.rst not found")
        return False

    print("OK Sphinx config and index found")

    if not os.path.exists('build_docs.py'):
        print("ERROR: build_docs.py not found")
        return False

    try:
        from build_docs import build_documentation
        print("OK Build script imported")
        print("Running Sphinx build...")
        success = build_documentation()
        if not success:
            return False
        build_dir = Path('docs/_build')
        if build_dir.exists() and (build_dir / 'index.html').exists():
            print("OK Generated index.html")
            return True
        print("ERROR: Build completed but index.html not found")
        return False
    except Exception as e:
        print(f"ERROR during build: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 50)
    print("DFT Documentation Build Test")
    print("=" * 50)

    success = test_documentation_build()

    print("\n" + "=" * 50)
    if success:
        print("All tests passed! Documentation build successful.")
        print("\nNext steps:")
        print("1. Commit and push your changes")
        print("2. Enable GitHub Pages in repository settings")
        print("3. Set source to 'GitHub Actions'")
    else:
        print("Some tests failed. Please check the errors above.")
        sys.exit(1)
    print("=" * 50)


if __name__ == "__main__":
    main()
