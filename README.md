# HCC
This is the repository of "A Novel Two-Phase Cooperative Co-evolution Framework for Large-Scale Global Optimization with Complex Overlapping". 

## Branching

- `main`: stable baseline that should stay runnable.
- `exp/<topic>`: experimental algorithm or ablation work.
- `fix/<topic>`: bug fixes or rollback repairs.
- `chore/<topic>`: repo cleanup, docs, or tooling updates.

Suggested habits:

- Start every new method trial from `main`.
- Commit before large experiments or structural edits.
- Merge back to `main` only after the branch is verified.

## Experiment Workflow

For Windows PowerShell, the repo includes a helper script:

```powershell
.\scripts\start-experiment.ps1 grg-v2
```

What it does:

- checks that you are on `main`
- checks that the working tree is clean
- creates an annotated baseline tag like `baseline/20250827-153000-grg-v2`
- switches to a new branch like `exp/grg-v2`

Useful options:

```powershell
.\scripts\start-experiment.ps1 grg-v2 -DryRun
.\scripts\start-experiment.ps1 grg-v2 -NoTag
.\scripts\start-experiment.ps1 grg-v2 -AllowDirty
```

Recommended loop:

1. Run `.\scripts\start-experiment.ps1 <topic>`
2. Make code changes and commit normally
3. Run verification and experiments
4. Compare against the printed baseline tag before merging back to `main`
