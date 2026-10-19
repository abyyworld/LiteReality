# Working in this repo

This is a private copy of LiteReality-Agent, imported from upstream at 28ffe1f (Apache-2.0). Read
[PROJECT_NOTES.md](PROJECT_NOTES.md) first: it records what this copy adds, how it is set up, the
known issues and what is next.

## Rules

- **Ask before every commit and every push.**
- Commits are authored AND committed as `abyyworld <annolieberto@gmail.com>`. No
  `Co-Authored-By`, `Generated with Claude Code` or other Claude lines in commit messages or PR
  descriptions, and no `claude/...` branch names — work on `main` or a descriptive branch.
- **Never print, cat or echo `.env`** or any token/key value. To see which keys are set:
  `grep -E '^[A-Z_]+=.+' .env | cut -d= -f1`
- Never commit `.env`, keys, `run/` outputs or any file over 50 MB. Put shareable results under
  `results/<scan>/` (small text files only).
- Keep the checkout in a permanent folder (e.g. `~/LiteReality-Agent`), never a temp directory.

## Checks before proposing a commit

```bash
uv run ruff check src tests sanity.py scripts deploy
uv run pytest -q -n 4
```
