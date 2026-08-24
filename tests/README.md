# Tests

Run from the project root:

```bash
.venv/bin/pytest
```

The full default model is intentionally not instantiated by tests. Its exact
parameter count is checked from the configuration, while the other tests use a
small two-cycle model and exercise block dispatch, routing balance, masks, and
the token-shard data loader.
