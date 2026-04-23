## Setup
1. Run `uv sync`
2. To add packages: `uv add <package name>`
3. To delete a package: `uv remove <package name>`
3. To run a file: `uv run <python file>`


Overnight running command:
nohup uv run src/transcribe.py --input-list data/selected_videos.txt --output-dir output --timestamped-log > output/overnight_stdout.log 2>&1 &

Running with a selected input:
uv run src/transcribe.py --input-list data/selected_videos.txt