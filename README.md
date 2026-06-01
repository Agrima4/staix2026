# STAIX 2026 — Award B Pipeline

## Reproduction

```bash
# 1. Clone and enter repo
git clone https://github.com/<org>/<repo>.git
cd <repo>

# 2. Place held-out dataset into data/
#    data/ must contain DATA_DESCRIPTION.md + CSV files

# 3. Launch agent
claude --dangerously-skip-permissions

# 4. Issue single prompt
Do the data analysis
```

## Outputs

| File             | Location  |
|------------------|-----------|
| `submission.csv` | repo root |
| `report.pdf`     | repo root |

## Requirements

Claude Code with `claude-sonnet-4-6`. No other setup needed — the agent installs Python packages at runtime via `pip`.

## How it works

The agent reads `CLAUDE.md` at startup which defines a 9-phase autonomous workflow: discover schema → determine task type → data quality → feature engineering → baselines → model competition → self-verify predictions → write submission → generate report.

All decisions (task type, model selection, feature engineering) are made at runtime by reading the data. No domain-specific tuning.