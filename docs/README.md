# DFT Documentation System

This directory contains the automatically generated HTML documentation for the Droplet-Film Theory Development project.

## How It Works

1. **SRT Files**: The source documentation is written in SRT (subtitle) format for easy editing
2. **Automatic Conversion**: GitHub Actions automatically converts SRT files to HTML when you push changes
3. **GitHub Pages**: The HTML documentation is automatically deployed to GitHub Pages for easy access

## File Structure

```
docs/
├── index.html              # Main documentation homepage
├── html/                   # Individual documentation pages
│   ├── main_documentation.html
│   ├── installation_guide.html
│   ├── usage_examples.html
│   ├── api_reference.html
│   └── troubleshooting.html
├── assets/                 # CSS and other resources
│   └── style.css
└── _build/                 # Build output for GitHub Pages
    └── html/
```

## Local Development

To build the documentation locally:

```bash
# Install dependencies
pip install -r docs_requirements.txt

# Convert SRT to HTML
python convert_srt_to_html.py

# Build complete site
python build_docs.py
```

## GitHub Pages Setup

1. Go to your repository Settings
2. Navigate to Pages section
3. Set Source to "GitHub Actions"
4. The workflow will automatically deploy your docs

## Customization

- **Styling**: Edit `build_docs.py` to modify the CSS
- **Layout**: Modify the HTML templates in the conversion scripts
- **Navigation**: Update the navigation structure in the build scripts

## Adding New Documentation

1. Create a new `.srt` file in the root directory
2. Add it to the conversion mapping in `convert_srt_to_html.py`
3. Update the navigation in `build_docs.py`
4. Push your changes - GitHub Actions will handle the rest!

## Accessing Your Documentation

Once deployed, your documentation will be available at:
`https://[username].github.io/[repository-name]/`
