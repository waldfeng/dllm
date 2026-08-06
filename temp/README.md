# Tests

Run from the project root:

```bash
.venv/bin/pytest
```

The full default model is intentionally not instantiated by tests. Its parameter
estimate is checked from the configuration, while the other tests use a 24-block
memory with a small hidden size.
