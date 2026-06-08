# Agent Notes

## Environment

- Work from the repository root: `/mnt/e/MatchAndProject/alpha101`.
- Use the `alpha101` conda environment for Python commands.
- Preferred test invocation:

```bash
conda run -n alpha101 python -m pytest <paths>
```

## Code Test Invocation

- During focused code changes, defer test execution until the end and run the selected test command once.
- If the final test fails because of environment or dependency issues, report the exact command and failure.

## Factor Search Data Split

- In factor search, never use the test segment for generation, selection, pruning, hyperparameter choice, or mid-search validation.
- Use train data for genetic search scoring and validation data for mid-search robustness checks.
- Reserve the test segment for final out-of-sample evaluation of frozen candidate factors only.
- Reports may include test metrics only after candidate expressions and selection rules are fixed.

## Code Practices

- Use `rg` for searching files or text.
- Keep edits scoped to the requested factor search, validation, or evaluation behavior.
- Do not revert unrelated working tree changes.
