# CS188 FA25 Project 5: Implementation Plan and Work Assignment

## 1. Current repository status

- Repository: `Nguyen-Danh-Hai-Dang/PRJ5-ML14`
- Working branch for importing the starter: `import/cs188-fa25-original`
- The original starter archive is expected to contain:
  - `machinelearning/backend.py`
  - `machinelearning/models.py`
  - `machinelearning/chargpt.py`
  - `machinelearning/gpt_model.py`
  - `machinelearning/train.py`
  - `machinelearning/losses.py`
  - `machinelearning/autograder.py`
  - `machinelearning/VERSION`
  - `machinelearning/data/mnist.npz`
  - `machinelearning/data/lang_id.npz`
- Source import status: pending local extraction, review, commit, and push. Do not report the starter as imported until the files are visible in the repository and verified.

## 2. Task breakdown

The assignment contains six core technical questions (Q1-Q6), an optional bonus (Q7), and a collaboration/AI reflection (Q8). Confirm exact wording, scoring, and requirements against the official assignment before implementation.

| Workstream | Scope | Primary owner | Required evidence |
|---|---|---|---|
| A | Q1 Perceptron and Q2 nonlinear regression / sine | Member 1 | Implementation, targeted tests, explanation of approach, latest run results |
| B | Q3 digit classification and Q5 CNN | Member 2 | Implementation, targeted tests, model/training results, explanation of choices |
| C | Q4 language identification / RNN | Member 3 | Implementation, targeted tests, training/evaluation results, explanation of sequence handling |
| D | Q6 attention / causal mask and Q7 Character-GPT bonus | Member 4 | Implementation, mask/shape tests, generation or evaluation results where applicable |
| Integration and reporting | Shared integration, regression checks, README/report, Q8 collaboration and AI-use reflection | All members; coordinator tracks completion | Final integrated test log, known issues, contributions, accurate AI-use disclosure |

Replace “Member 1-4” with the actual team member names/handles before treating this as a final assignment. The Q7 bonus is lower priority than completing and validating Q1-Q6.

## 3. Definition of done for every task

A task is not complete merely because AI has generated code or a file has been edited. Before asking for review, the owner and any AI assistant must:

1. Read the relevant assignment requirements and existing starter interfaces.
2. Make changes on a task-specific branch, not directly on `main`.
3. Explain the implementation and list files changed.
4. Run the applicable tests or autograder commands that are actually available.
5. Record the exact commands and observed results. Do not invent results or claim a test passed if it was not run.
6. Review the diff for accidental changes, debug code, generated artifacts, and unrelated edits.
7. Open a Pull Request with a concise summary, test evidence, limitations, and any AI-assisted work that needs disclosure.
8. After fixing review feedback, rerun affected tests and update the report with the newest results.

If a test cannot be run, state why and what remains unverified. A failed test must be reported as failed, with the error and next step.

## 4. Cross-review plan

- Each PR must be reviewed by at least one teammate who did not author the change.
- The reviewer checks requirement coverage, tensor/data shapes, edge cases, reproducibility, readability, test evidence, and whether the change breaks another question.
- The author responds to every substantive review comment and reruns relevant tests after changes.
- Integrate small, coherent PRs. Avoid having multiple members edit the same core function simultaneously without coordinating.
- Do not merge a PR while required checks are failing or while important review questions remain unanswered.

## 5. AI work-report format

Every AI-assisted task report should include:

- **Task / question:**
- **Branch and commit:**
- **Files changed:**
- **Approach:** a short explanation the student can independently defend.
- **Commands run:**
- **Observed results:** exact output or a concise, faithful summary.
- **Review performed:** edge cases, interfaces, and potential regressions checked.
- **Known issues / unverified items:**
- **Next step:**

The person responsible for the task remains accountable for understanding, checking, and explaining the submitted work. AI output is a proposal, not proof of correctness.

## 6. Integration checklist

- [ ] Starter files and datasets are present and match the supplied archive.
- [ ] Starter and assignment instructions are preserved; no required files are silently removed.
- [ ] Each question has an owner, branch, PR, and test evidence.
- [ ] Q1-Q6 are implemented and cross-reviewed.
- [ ] Q7 bonus is attempted only after core requirements are on track.
- [ ] Tests are rerun on the integrated branch.
- [ ] README documents setup and reproducible test commands.
- [ ] Q8 accurately describes team collaboration and AI use, based on what actually happened.
- [ ] Final report records results, limitations, and each member's actual contribution.

## 7. Important status rule

This document is a plan, not proof of implementation or test success. Update completion status only after inspecting the repository and the latest test evidence. Never fabricate metrics, screenshots, test outputs, or individual contributions.
