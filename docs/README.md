# DFT Documentation System

This directory contains the documentation source and build for the Droplet-Film Model Development project.

## How It Works

1. **RST Files**: The source documentation is written in reStructuredText (`.rst`) in `docs/source/`
2. **Build**: Sphinx builds the RST files to HTML (e.g. `python build_docs.py`)
3. **GitHub Pages**: The HTML in `docs/_build` is deployed via GitHub Actions

## File Structure

```
docs/
├── source/                 # reStructuredText (.rst) source files
│   ├── index.rst
│   ├── documentation.rst
│   ├── installation_guide.rst
│   ├── usage_examples.rst
│   ├── api_reference.rst
│   ├── troubleshooting.rst
│   └── conf.py             # Sphinx configuration
├── _build/                 # Built HTML (output of build_docs.py)
│   └── (HTML and assets)
└── README.md               # This file
```

## Local Development

To build the documentation locally:

```bash
# Install dependencies (including Sphinx)
pip install -r docs_requirements.txt

# If you have SRT files and want to regenerate RST (optional)
python srt_to_rst.py

# Build the documentation (RST -> HTML via Sphinx)
python build_docs.py
```

## Adding New Documentation

1. Create a new `.rst` file in `docs/source/`
2. Add it to the `toctree` in `docs/source/index.rst`
3. Run `python build_docs.py` and push; GitHub Actions will deploy

## Accessing Your Documentation

Once deployed, the documentation will be available at:
`https://[username].github.io/[repository-name]/`
