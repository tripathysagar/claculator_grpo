# Colab CLI recovery reference

## Executed notebook preservation

After `colab exec` exits successfully:

1. Before execution, remove or quarantine any stale `<notebook>_output.ipynb`.
2. Confirm the newly created output exists locally, is non-empty, and has a
   modification time after the run started.
3. Validate it with `nbformat`; verify the embedded `run_id`, all expected code
   cells, and the final summary output.
4. Move it immediately to
   `outputs/notebooks/<run_id>_<notebook>_output.ipynb`.
5. Record its SHA-256 in the local artifact manifest.
6. Keep the source notebook unchanged and output-free unless requested.

The CLI serializes the output notebook only after execution returns. Runtime
reclamation, interruption, timeout, or crash may leave no valid output notebook
even when remote checkpoints exist. Treat remote metrics/checkpoints as the
recovery source of truth, export `colab log` during long runs, and verify the
output notebook before stopping the session. A session-history export is
recovery evidence, not a substitute for a fully executed notebook.

Notebook output must not contain tokens, environment values, authorization
headers, signed URLs, or exception dumps containing credentials. Fix the source
cell and rerun rather than only redacting generated output.

For sampled or multi-seed work, make each seed independently recoverable. Write,
download, and verify one seed's artifacts before starting the next.

## Failure handling

- **Remote parent missing:** create it under `/content`, then retry upload.
- **Session reclaimed/not found:** run `colab sessions`, provision a new named
  session, and resume from downloaded checkpoints.
- **Accelerator unavailable/quota denied:** report the denial; use T4 or CPU only
  if that still satisfies the experiment.
- **Execution timed out:** inspect status and logs first. A local CLI timeout
  does not prove the remote kernel stopped.
- **Failure after secret injection:** preserve safe diagnostics, then stop the
  session or restart the kernel so secrets do not remain in memory.
- **Out of memory:** preserve the last checkpoint and failed configuration,
  reduce batch/sequence size or enable gradient checkpointing, then use a new
  run ID.
- **NaN/Inf metrics or loss:** stop, preserve diagnostics/checkpoint, and treat
  the experiment as failed.
- **Output artifacts missing:** keep the VM alive, inspect paths with `colab ls`,
  and download recoverable artifacts before rerunning.
- **Evaluator bug:** fix the source and rerun; never hand-edit generated metrics
  or predictions.
- **VM reclaimed after completion:** list recovered artifacts and rerun only the
  missing bounded units.
