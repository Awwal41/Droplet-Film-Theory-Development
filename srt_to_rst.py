#!/usr/bin/env python3
"""
Convert SRT documentation files to reStructuredText (.rst) for Sphinx.
Run from project root: python srt_to_rst.py
"""

import re
from pathlib import Path


def clean_srt_content(content):
    """Remove SRT timing markers and index numbers."""
    lines = content.split('\n')
    cleaned = []
    for line in lines:
        if not line.strip():
            cleaned.append(line)
            continue
        if '-->' in line:
            continue
        if line.strip().isdigit() and len(line.strip()) <= 4:
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def srt_to_rst(content):
    """Convert SRT-style content to reStructuredText."""
    content = clean_srt_content(content)
    lines = content.split('\n')
    rst_lines = []
    in_code_block = False
    code_lang = 'python'
    i = 0
    doc_title_done = False

    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()

        if line_stripped.startswith('```'):
            if not in_code_block:
                match = re.match(r'^```(\w*)', line_stripped)
                code_lang = match.group(1) if match and match.group(1) else 'text'
                rst_lines.append('')
                rst_lines.append('.. code-block:: ' + code_lang)
                rst_lines.append('')
                in_code_block = True
            else:
                rst_lines.append('')
                in_code_block = False
            i += 1
            continue

        if in_code_block:
            rst_lines.append('   ' + line.replace('\t', '    '))
            i += 1
            continue

        if line_stripped == '---':
            rst_lines.append('')
            rst_lines.append('---')
            rst_lines.append('')
            i += 1
            continue

        if line_stripped.startswith('## '):
            header_text = line_stripped[3:].strip()
            rst_lines.append('')
            rst_lines.append(header_text)
            rst_lines.append('=' * len(header_text))
            rst_lines.append('')
            i += 1
            continue

        if line_stripped.startswith('### '):
            header_text = line_stripped[4:].strip()
            rst_lines.append('')
            rst_lines.append(header_text)
            rst_lines.append('-' * len(header_text))
            rst_lines.append('')
            i += 1
            continue

        # First non-empty line as document title (if we haven't seen ## yet)
        if not doc_title_done and line_stripped and not line_stripped.startswith('```'):
            doc_title_done = True
            rst_lines.append(line_stripped)
            rst_lines.append('=' * len(line_stripped))
            rst_lines.append('')
            i += 1
            continue

        rst_lines.append(line)
        i += 1

    return '\n'.join(rst_lines).strip() + '\n'


def main():
    project_root = Path(__file__).resolve().parent
    source_dir = project_root / 'docs' / 'source'
    source_dir.mkdir(parents=True, exist_ok=True)

    srt_files = [
        ('INSTALLATION_GUIDE.srt', 'installation_guide.rst'),
        ('USAGE_EXAMPLES.srt', 'usage_examples.rst'),
        ('API_REFERENCE.srt', 'api_reference.rst'),
        ('TROUBLESHOOTING.srt', 'troubleshooting.rst'),
    ]

    for srt_name, rst_name in srt_files:
        srt_path = project_root / srt_name
        if not srt_path.exists():
            print(f"Warning: {srt_name} not found, skipping")
            continue
        content = srt_path.read_text(encoding='utf-8')
        rst_content = srt_to_rst(content)
        out_path = source_dir / rst_name
        out_path.write_text(rst_content, encoding='utf-8')
        print(f"OK {rst_name}")

    print("Done. RST files are in docs/source/")


if __name__ == '__main__':
    main()
