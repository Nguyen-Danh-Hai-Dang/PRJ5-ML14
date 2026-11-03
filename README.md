# PRJ5-ML14 | CS188 FA25 Project 5

Team 14 project workspace for the CS188 Fall 2025 Machine Learning project.

- Official project page: https://inst.eecs.berkeley.edu/~cs188/archive/fa25/projects/proj5/
- Team repository: https://github.com/Nguyen-Danh-Hai-Dang/PRJ5-ML14

## Repository workflow

1. Keep `main` stable; do not push feature work directly to `main`.
2. Each member creates a branch for their assigned work, for example `member1/perceptron` or `member3/rnn`.
3. Make small, focused commits and push the branch.
4. Open a Pull Request into `main`, explain the change and how it was tested, and request another member to review it.
5. Merge only after review and resolving conflicts.
6. Link the relevant branch/PR to the corresponding CoReach task.

## Project structure

- `machinelearning/`: official starter code and required data files from the provided archive.
- `report/`: report source and supporting materials (to be added).
- `docs/`: setup, experiment notes, and team workflow documentation.

## Local environment requirements

Install the following on your computer:

- **Python 3.10 or 3.11** (recommended for a consistent team environment).
- **pip**, included with most Python installations.
- Python packages: **PyTorch**, **NumPy**, and **Matplotlib**.
- **Git** to clone the repository and switch branches.
- Enough free disk space for the repository, datasets, and Python packages.

A GPU is **not required**. CPU execution is fine, although some training questions may take several minutes. Keep the same Python version and package setup across team members where possible.

### 1. Clone the repository

Open PowerShell on Windows, or a terminal on Linux/macOS, then run:

```bash
git clone https://github.com/Nguyen-Danh-Hai-Dang/PRJ5-ML14.git
cd PRJ5-ML14
```

If the repository is private, sign in to GitHub and make sure you have accepted the repository invitation before cloning.

### 2. Create and activate a virtual environment

Run these commands from the repository root.

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the Python commands below through the environment directly, or use Command Prompt and activate with `.venv\\Scripts\\activate.bat`.

**Linux/macOS:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

If Python 3.11 is not installed, install it first or use another supported Python version consistently across the team.

### 3. Install dependencies

With the virtual environment activated, run:

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib
```

Verify the packages import successfully:

```bash
python -c "import torch, numpy, matplotlib; print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); print('Matplotlib:', matplotlib.__version__)"
```

### 4. Run the local autograder

**Important:** change into the `machinelearning` directory before running the autograder. This ensures imports such as `models`, `train`, `losses`, and `backend` resolve correctly.

```bash
cd machinelearning
python autograder.py --no-graphics
```

To test one question only, use `-q` followed by the question number:

```bash
python autograder.py -q q1 --no-graphics
python autograder.py -q q2 --no-graphics
python autograder.py -q q3 --no-graphics
python autograder.py -q q4 --no-graphics
python autograder.py -q q5 --no-graphics
python autograder.py -q q6 --no-graphics
```

Run the full autograder again after integrating changes from other members. Some questions train models and may take time. Read the output carefully: a test can run without receiving full points if the model does not reach the required accuracy or loss threshold. The provided autograder currently defines checks for Q1–Q6; follow the assignment instructions for any separate Q7/Character-GPT task.

To return to the repository root after testing:

```bash
cd ..
```

### Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'` (or `numpy`/`matplotlib`):** activate the virtual environment and rerun the dependency installation command.
- **`can't open file 'autograder.py'`:** check that your current directory is `machinelearning`.
- **Dataset not found:** do not delete or rename the provided data files; check that `machinelearning/data/mnist.npz` and `machinelearning/data/lang_id.npz` exist.
- **PowerShell does not recognize `python`:** try `py` or reinstall Python with the PATH option enabled.
- **Long-running test:** training tests may take several minutes. Let them finish unless they are clearly stuck.

## Source archive

The provided `machinelearning.zip` archive is the baseline. Preserve it unchanged locally and make any implementation changes on feature branches. The `data/mnist.npz` and `data/lang_id.npz` files are required project data, not generated artifacts.

## Contribution checklist

- [ ] Confirm the baseline runs before changing it.
- [ ] Work on a feature branch.
- [ ] Add or update tests / autograder checks where appropriate.
- [ ] Record commands, configuration, and results so another member can reproduce them.
- [ ] Open a PR and obtain review before merging.

## Completed Questions

This section summarizes the implementation and testing results for Questions 1 and 2. The remaining questions are still part of the project.

### Question 1: Perceptron

**Problem:** Implement a Perceptron model for binary classification.

**Implementation:**
- The model uses a weight vector to calculate a score for each input.
- The prediction is based on the sign of the score.
- When the model makes a wrong prediction, it updates the weights using the Perceptron update rule:

  \[
  w \leftarrow w + yx
  \]

- The training process continues until the model correctly classifies the training examples.

**Testing:**

Run the Question 1 autograder from the `machinelearning` directory:

```bash
python autograder.py -q q1 --no-graphics
```

**Result:** 5/5 points.

### Question 2: Regression

**Problem:** Train a neural network to approximate the function \(y = \sin(x)\) over the interval \([-2\pi, 2\pi]\).

**Implementation:**
- `RegressionModel` predicts a continuous value from a real-valued input.
- The model uses a neural network to learn the relationship between input values and their target values.
- Mean Squared Error (MSE) is used to measure the difference between predictions and true labels.
- The training process uses backpropagation and an optimizer to update the model parameters.
- The required final loss is no more than `0.02`.

**Testing:**

Run the Question 2 autograder from the `machinelearning` directory:

```bash
python autograder.py -q q2 --no-graphics
```

**Actual result:**

```text
Your final loss is: 0.019308
*** PASS: check_regression
Question q2: 5/5
```

The final loss is `0.019308`, which is below the required threshold of `0.02`.

### Current Progress

| Question | Result |
|---|---:|
| Question 1: Perceptron | 5/5 |
| Question 2: Regression | 5/5 |
| **Total** | **10/10** |

These results confirm that Questions 1 and 2 passed the autograder. They do not mean that the entire project is complete.
## Keep the repository clean

Do not commit virtual environments (`.venv/`), Python caches (`__pycache__/`), local editor settings, secrets, or generated model checkpoints. If these appear in `git status`, do not add them to a commit.
