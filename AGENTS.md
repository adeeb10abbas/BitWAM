# BitWAM agent guide

Read `docs/EXECUTION_PLAN.md` before editing. Complete its phases in order and
keep each completed phase in a small commit on `ali/claude`.

Run only the phase-relevant checks:

```bash
uv run ruff check
uv run pytest -q
```

Do not commit datasets, checkpoints, rollout videos, caches, environments, or
raw simulator output. Store exact upstream revisions, checkpoint identifiers,
configs, seeds, and compact metrics in Git. A CPU/import smoke test is not a
CUDA smoke test, and an open-loop forward pass is not a closed-loop LIBERO result.
