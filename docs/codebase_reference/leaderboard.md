# `leaderboard/` Module Reference

This directory contains the Python automation logic and static assets required to build the web-based HTML leaderboard. It maps experiment configurations and `BenchmarkMetrics.csv` files into an interactive table.

## Dependencies & Imports

```python
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse
```
- **Standard Library:** Uses `subprocess` to execute `git` commands (for tracking experiment contributors). Uses `json` and `csv` to extract data from the `results/` payload. Uses `urllib.parse` and `re` to sanitize github endpoints and URL paths.

---

## Data Flow & Architecture

1. **Target Identification:** `generate_leaderboard.py` iterates over all child directories inside `results/` using `Path.iterdir()`.
2. **Data Ingestion:**
   - For every experiment folder, it calls `read_config` (`exp_config.json`).
   - It calls `read_metrics` (looking for `metrics/BenchmarkMetrics.csv`).
3. **Attribution Mining:** The script calls `subprocess.run(["git", "log", "--follow", ...])` on the training script associated with the experiment. It parses the commit author's email/name to generate an automatic GitHub avatar link.
4. **Seed Collapsing:** It groups experiments that share identical configurations (Network, Algorithm, Task) but differ only by Random Seed. It averages their scalar metrics.
5. **Template Rendering:** A massive string replacement operation occurs on `leaderboard_template.html`. Keys like `__DATA__` (the raw JSON payload) and `__PAGE_TITLE__` (from `leaderboard_strings.json`) are injected.
6. **Output Generation:** The final `.html` and associated `.png` logo files are written to `docs/leaderboard/`.

---

## Deep Function References: `generate_leaderboard.py`

### 1. `collect_experiments(results_dir: Path) -> List[Dict]`
**Purpose:** Iterates the file system to build the raw experiment list.
**Logic:**
```python
raw_experiments = []
for exp_dir in sorted(results_dir.iterdir()):
    config = read_config(exp_dir)
    metrics = read_metrics(exp_dir)
    if not config or not metrics:
        continue
    # ... (append configuration and metrics data to dict)
```
Ignores directories missing valid configs or metrics (e.g., failed or crashed runs).

### 2. `collapse_repeated_experiments(experiments: List[Dict]) -> List[Dict]`
**Purpose:** Groups independent Monte-Carlo simulations (seeds) into single, statistically averaged leaderboard rows (folds).
**Logic:**
Uses a helper `collapse_key` to build a unique tuple signature based on algorithmic hyperparameters:
```python
def collapse_key(exp: Dict) -> Tuple[str, ...]:
    return tuple(str(exp.get(field) or "") for field in [
        "exp_type", "env_config", "task_config", "network", "algorithm", "script", "alg_config"
    ])
```
If multiple experiments evaluate to the identical `collapse_key` tuple, `average_metrics()` is called to calculate the mean of every numerical scalar in `BenchmarkMetrics.csv` across the grouped runs.

### 3. `script_contributor_info_from_git(script_file, repo_root, cache)`
**Purpose:** Credits the author of the experiment code.
**Logic:**
Runs a reverse `git log` command to trace the original file creator.
```python
result = subprocess.run(
    ["git", "log", "--follow", "--reverse", "--format=%aN%x1f%aE", "--", rel_script],
    capture_output=True, text=True, check=False
)
payload = lines[0].split("\x1f", 1)
contributor_name = payload[0].strip()
```

### 4. `build_html(payload: Dict, output_path: Path, template: str)`
**Purpose:** Hydrates the frontend UI.
**Logic:**
Constructs a giant dictionary of HTML macro replacements.
```python
replacements = {
    "__PAGE_TITLE__": strings["page_title"],
    "__DATA__": json.dumps(payload, indent=2),
    # ...
}
for token, value in replacements.items():
    html = html.replace(token, str(value))
```
The `__DATA__` payload is embedded directly as a JavaScript variable inside the HTML, allowing the frontend JS to handle all filtering, sorting, and pagination dynamically without requiring a backend web server.
