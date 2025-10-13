# Reports Directory

This folder collects generated artifacts such as compliance reports, performance benchmarks, and test summaries. They are produced by scripts like `run_all_tests.sh` and are intentionally not version-controlled. After running the automation locally or in CI you will find files such as `mcp_compliance_report.json`, `performance_benchmark_report.json`, and `test_summary.md` here.

To keep the repository clean, these generated files are ignored by git. Regenerate them as needed by running:

```bash
python scripts/quick_internal_test.py
pytest -q
./run_all_tests.sh
```

Each command will recreate the relevant report files inside this directory.
