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

## Setup

Use the Python version and packages specified by the official project instructions. Keep the starter code structure unchanged until the team has verified it runs. Do not commit virtual environments, caches, local editor settings, secrets, or generated model checkpoints.

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
