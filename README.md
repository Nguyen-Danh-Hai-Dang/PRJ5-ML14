# PRJ5-ML14 | CS188 FA25 Project 5

**Group 14** — implementation of the CS188 Fall 2025 Machine Learning Project 5.

- Official assignment: https://inst.eecs.berkeley.edu/~cs188/archive/fa25/projects/proj5/
- Repository: https://github.com/Nguyen-Danh-Hai-Dang/PRJ5-ML14
- Main branch: `main`

## Team

| Member | Student ID | Initial task allocation |
|---|---:|---|
| Nguyễn Danh Hải Đăng | 24020056 | Q1 – Perceptron; Q2 – Regression |
| Mai Thành Đức | 24020078 | Q3 – Digit Classification; Q5 – Convolutional Neural Network |
| Phạm Công Minh | 23020553 | Q4 – Language Identification |
| Trần Thế Đức Anh | 24020029 | Q6 – Attention; Q7 – Character-GPT (not included in the completed Q1–Q6 results below) |

The table records the team's initial task allocation. The repository's commit and Pull Request history records the implementation and integration work as it was actually completed. In particular, some Q3/Q5 implementation commits were authored by Phạm Công Minh, while PRs for those changes were opened by Mai Thành Đức. Nguyễn Danh Hải Đăng also contributed the Q4 dataset-iteration integration fix in PR #11. PR authorship, commit authorship, and initial task allocation are therefore not interchangeable.

## Project status

Questions **Q1–Q6 are implemented and passed the local autograder** in the team's final integrated run.

| Question | Topic | Autograder result | Additional reported metric |
|---|---|---:|---|
| Q1 | Perceptron | 5/5 | — |
| Q2 | Non-linear Regression | 5/5 | Final loss: 0.019308 |
| Q3 | Digit Classification | 5/5 | 97.38% test accuracy reported in PR #10 |
| Q4 | Language Identification | 5/5 | 87.2% accuracy in the full-suite run |
| Q5 | Convolutional Neural Network | 3/3 | — |
| Q6 | Attention | 2/2 | — |
| **Total** | | **25/25** | |

**Important notes about the metrics**

- The 25/25 score is the result of the team's final local Q1–Q6 autograder run.
- The Q3 accuracy is the value reported in PR #10. It is not presented as a metric measured in the same run as every other value in this table.
- Q4 training is stochastic, so accuracy can vary between runs. The full-suite run recorded 87.2%; a separate isolated Q4 run produced a different value.
- Q5's result is reported as an autograder score. No additional test-accuracy value is claimed here.

## Repository structure

- `machinelearning/`: CS188 Project 5 starter code, implementation files, datasets, and autograder.
- `README.md`: project overview, setup, test instructions, and team information.

Keep the supplied datasets in `machinelearning/data/`. The files `mnist.npz` and `lang_id.npz` are required inputs.

## Environment

The project was tested locally on Windows using Python 3.12 and Python 3.14 environments, with PyTorch, NumPy, and Matplotlib. CPU execution is supported; a GPU is not required.

Recommended setup for a fresh Windows clone:

- Git
- Python 3.12
- pip
- An internet connection for installing Python packages

## Clone the repository

Open **Windows PowerShell** in the directory where you want the project folder to be created:

```powershell
git clone https://github.com/Nguyen-Danh-Hai-Dang/PRJ5-ML14.git
cd PRJ5-ML14
git switch main
git pull origin main
```

These commands download the repository, enter the project directory, select the submission branch, and synchronize it with the latest remote `main`.

To confirm that you are on the correct branch and have no local modifications:

```powershell
git branch --show-current
git status
git log -1 --oneline
```

The expected branch name is `main`. Before making a submission copy, `git status` should show a clean working tree.

## Set up Python

Run the following commands from the repository root.

### 1. Create a virtual environment

```powershell
py -3.12 -m venv .venv
```

This creates an isolated Python environment in the `.venv` folder. It keeps the project's packages separate from the system-wide Python installation.

### 2. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, either allow activation for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Or run the environment's Python executable directly without activating it:

```powershell
.\.venv\Scripts\python.exe --version
```

### 3. Install dependencies

With the virtual environment activated:

```powershell
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib
```

Verify the installation:

```powershell
python -c "import torch, numpy, matplotlib; print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); print('Matplotlib:', matplotlib.__version__)"
```

## Run the autograder

The autograder must be run from inside the `machinelearning` directory.

```powershell
cd machinelearning
python autograder.py --no-graphics
```

To test individual questions:

```powershell
python autograder.py -q q1 --no-graphics
python autograder.py -q q2 --no-graphics
python autograder.py -q q3 --no-graphics
python autograder.py -q q4 --no-graphics
python autograder.py -q q5 --no-graphics
python autograder.py -q q6 --no-graphics
```

Return to the repository root after testing:

```powershell
cd ..
```

Training-based tests can take several minutes. Allow each test to finish and inspect the reported score rather than assuming that starting the autograder means the question passed.

## Prepare a clean submission copy

The submission archive should be assembled from the final integrated `main` branch, not from an unfinished feature branch or an old local working folder.

1. Clone or update `main` using the commands above.
2. Confirm that `git status` is clean.
3. Confirm that `machinelearning/` contains the required source files and both datasets.
4. Include the required IEEE report/template files specified by the course submission instructions.
5. Do **not** include the `.venv/` directory, `__pycache__/`, temporary logs, local editor settings, or generated checkpoints unless the instructor explicitly requests them.
6. Create the required archive with the exact filename: `INT3401_11_Nhom14.zip`.

Do not rename or remove the supplied data files. Keep an untouched copy of the original starter archive separately if it is needed for comparison.

## Git collaboration workflow

The team used a branch-and-Pull-Request workflow:

1. Keep `main` as the integrated branch.
2. Create a separate feature branch for each task.
3. Commit focused changes and push the feature branch.
4. Open a Pull Request targeting `main`.
5. Review the changes and resolve any conflicts before merging.
6. Run the relevant autograder tests after integration.

Avoid committing virtual environments, Python caches, secrets, or machine-specific generated files.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'` (or `numpy` / `matplotlib`):** activate `.venv` and reinstall the dependencies.
- **`can't open file 'autograder.py'`:** change into `machinelearning` before running the autograder.
- **Dataset not found:** check that `machinelearning/data/mnist.npz` and `machinelearning/data/lang_id.npz` exist.
- **PowerShell cannot find `py`:** install Python and enable the Python Launcher, or use the full path to the Python executable.
- **A test takes a long time:** training-based questions may need several minutes, especially on CPU. Wait for the test to finish unless it has clearly stopped responding.
