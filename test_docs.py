#!/usr/bin/env python3
"""
Test script for the documentation build process
Run this to test the SRT to HTML conversion locally
"""

import os
import sys
from pathlib import Path

def test_conversion():
    """Test the SRT to HTML conversion process"""
    print("Testing documentation build process...")
    
    # Check if SRT files exist
    srt_files = [
        'DOCUMENTATION.srt',
        'INSTALLATION_GUIDE.srt', 
        'USAGE_EXAMPLES.srt',
        'API_REFERENCE.srt',
        'TROUBLESHOOTING.srt'
    ]
    
    missing_files = []
    for srt_file in srt_files:
        if not os.path.exists(srt_file):
            missing_files.append(srt_file)
    
    if missing_files:
        print(f"❌ Missing SRT files: {missing_files}")
        return False
    
    print("✅ All SRT files found")
    
    # Check if conversion scripts exist
    if not os.path.exists('convert_srt_to_html.py') or not os.path.exists('build_docs.py'):
        print("❌ Conversion scripts not found")
        return False
    
    print("✅ Conversion scripts found")
    
    # Try to import and run conversion
    try:
        from convert_srt_to_html import main as convert_main
        from build_docs import build_documentation
        
        print("✅ Scripts imported successfully")
        
        # Run conversion
        print("Running SRT to HTML conversion...")
        convert_main()
        
        # Run build
        print("Building documentation site...")
        build_documentation()
        
        # Check if output files were created
        docs_dir = Path('docs')
        if docs_dir.exists():
            html_files = list(docs_dir.glob('html/*.html'))
            if html_files:
                print(f"✅ Generated {len(html_files)} HTML files")
                return True
            else:
                print("❌ No HTML files generated")
                return False
        else:
            print("❌ Docs directory not created")
            return False
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("DFT Documentation Build Test")
    print("=" * 50)
    
    success = test_conversion()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Documentation build successful.")
        print("\nNext steps:")
        print("1. Commit and push your changes")
        print("2. Enable GitHub Pages in repository settings")
        print("3. Set source to 'GitHub Actions'")
        print("4. Your docs will be automatically deployed!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
