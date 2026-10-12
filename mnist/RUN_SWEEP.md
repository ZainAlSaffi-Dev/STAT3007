# How to run the MNIST sweep

Run every command from the `mnist/` folder.

1. Check that the kernels and the logger are correct. This takes a few minutes.

   ```bash
   uv run python mnist_sweep.py check
   ```

2. Time a few short runs and print the projected wall time. This takes a few minutes.

   ```bash
   uv run python mnist_sweep.py pilot
   ```

3. Start the sweep.

   ```bash
   uv run python mnist_sweep.py run
   ```

   The sweep uses 3 workers for the wide networks and 5 for the narrow ones.
   Set `--wide-workers` and `--narrow-workers` so that their sum is about the
   number of processor cores.

   To run one batch only, add options such as `--widths 100 200`. If the sweep
   stops, run the same command again. Finished runs are skipped.

4. Watch the progress. `status` counts the finished runs, and the log file
   `sweep/logs/sweep.log` records each run as it starts and finishes.

   ```bash
   uv run python mnist_sweep.py status
   ```

5. Build the tables when the runs have finished. This also runs the checks in
   `validate`.

   ```bash
   uv run python mnist_sweep.py assemble
   ```

The tables are written to `sweep/`: `runs.parquet`, `checkpoints.parquet`,
`report_kernel.parquet`, the data dictionary and the manifest. The saved
weights take a few gigabytes. To write somewhere else, for example a folder
that is not synced to cloud storage, add `--out <folder>` to every command.
