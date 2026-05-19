# `tests/` Module Reference

This directory provides integration and execution safety tests via the `pytest` framework. Because reinforcement learning and SUMO traffic simulations can be highly brittle to configuration typos, these tests rapidly execute "mock" environments to ensure the pipeline doesn't crash prior to deploying long-running training jobs.

## Dependencies & Imports

```python
import os
import pytest
import shutil
import subprocess
from pathlib import Path
```
- **External Framework**: Relies on `pytest` for discovery, parameterization, and assertion handling.
- **Standard Library**: Uses `subprocess` to launch the actual URB python scripts as independent shell commands, and `shutil` to verify the user's system PATH.

---

## Data Flow & Architecture

1. **Environment Verification:** Before running any test, the global `autouse=True` fixture checks if the `sumo` traffic simulator binary exists in the host machine's PATH. If not, testing aborts immediately.
2. **File Discovery:** The scripts use `Path.rglob("*.py")` to recursively find all execution scripts inside `baseline_models/` and `scripts/`.
3. **Execution Injection:** The tests use `@pytest.mark.parametrize` to loop over the discovered files. For each file, it fires a `subprocess.run` command feeding it a dummy configuration (`"test"` config files usually contain extremely brief simulation lengths, e.g., 1 episode).
4. **Validation:** If the subprocess returns a non-zero exit code (indicating a Python crash, PyTorch memory error, or SUMO crash), `pytest.fail` is triggered.

---

## Deep Function References

### `test_baselines.py`
Ensures all decentralized routing heuristics can initialize and complete a route.

**1. `check_sumo_installed()`**
```python
@pytest.fixture(scope="session", autouse=True)
def check_sumo_installed():
    sumo_executable = shutil.which("sumo")
    if sumo_executable is None:
        pytest.exit("[SUMO ERROR] SUMO is not installed or not in PATH.")
    # ... runs 'sumo --version'
```
Runs exactly once per test session. Crucial because URB will hard-crash midway if `sumo` cannot be spawned by `routerl`.

**2. `test_python_script_execution(baseline)`**
```python
@pytest.mark.parametrize("baseline", baseline_names)
def test_python_script_execution(baseline):
    result = subprocess.run(
        ["python", "baselines.py",
         "--id", f"test_{baseline_name}",
         "--alg-conf", "test",
         "--env-conf", "test",
         "--task-conf", "test",
         "--net", "saint_arnoult",
         "--model", baseline_name],
        capture_output=True, text=True, check=True, cwd=python_script.parent
    )
```
Notice how it dynamically injects `--model baseline_name`. It verifies that `baselines.py` correctly handles the registry lookup for every model found in the `baseline_models` folder.

### `test_scripts.py`
Ensures that all complex RL configurations (Centralized DQN, Independent PPO, MAPPO, QMIX, etc.) compile their PyTorch graphs successfully and can backpropagate at least once without crashing.

**1. File Filtering**
```python
python_scripts = list(SCRIPTS_DIR.rglob("*.py"))
excluded_scripts = ["utils.py", "base_script.py", "baselines.py", "greedy_utils.py"]
python_scripts = [s for s in python_scripts if s.name not in excluded_scripts]
```
Automatically excludes utility and abstract templates so that only executable pipeline scripts are tested.

**2. `test_python_script_execution(script_path)`**
Runs the parametrized command. Similar to the baseline test, but explicitly executes the discovered `script_filename` (e.g., `python ippo_torchrl.py ...`) to ensure the specific PyTorch architectures and TorchRL `SyncDataCollector` modules initialize cleanly.
