---
name: colab-cli
description: Runs reproducible experiments through the Colab CLI: provisions CPU/GPU sessions, injects project secrets safely, uploads inputs, smoke-tests and executes scripts or notebooks, checkpoints progress, verifies and downloads artifacts, diagnoses failures, and enforces and verifies runtime cleanup. Use whenever the user asks to run experiments or code on Colab, use a Colab accelerator, execute a notebook remotely, retrieve Colab outputs, or manage Colab CLI sessions.
---

# Colab CLI

Run local project code on Google Colab while preserving outputs and avoiding
leaked compute sessions.

## Core rules

1. Inspect the installed CLI before relying on remembered syntax:

   ```bash
   colab version
   colab help <command>
   ```

2. Run `colab sessions` before provisioning. Reuse the intended healthy session
   or stop an accidental orphan before creating another.
3. Use a unique, descriptive session name and pass it with `-s` on every
   session-scoped command.
4. Use CPU unless the workload benefits from an accelerator. Prefer T4 as the
   economical GPU default; request L4/A100/H100 only when justified and available.
5. Treat `/content` as ephemeral. Download every required artifact before stopping
   or before a long multi-stage run can be reclaimed.
6. Always stop the session after artifacts are verified locally.
7. Never call an experiment complete because the process exited successfully.
   Completion requires validated metrics, predictions/checkpoints, configuration,
   provenance, and a stopped session.
8. Never modify generated metrics or predictions by hand. Fix source code and
   regenerate them.

## Experiment contract

Before allocating compute, identify these values from the request and code. Ask
only when a missing choice materially changes the experiment:

- hypothesis or objective;
- primary metric and success criterion;
- baseline/comparison;
- dataset split and immutable input paths;
- model ID plus revision when applicable;
- seed or seed set;
- accelerator and rough time budget;
- smoke-test size;
- checkpoint frequency and resume behavior;
- required output artifacts.

Every run needs a unique `run_id`, for example
`qwen06b-sft-seed20260721-20260721T1500Z`. Use it in the session name and remote
result directory. Never include secret values in names.

Use this remote layout:

```text
/content/experiments/<run_id>/
  config.json
  runtime.json
  metrics.json
  predictions.jsonl
  checkpoints/
```

Omit artifacts that do not apply, but always keep `config.json`, `runtime.json`,
and the experiment's primary output.

Use this repository layout:

```text
notebooks/             # tracked source notebooks without generated outputs
outputs/notebooks/     # tracked, sanitized executed notebooks
outputs/evaluations/   # tracked metrics and auditable predictions
```

Keep lightweight reproducibility artifacts in Git. Ignore secrets, caches,
temporary/partial files, model binaries, and checkpoint directories. Store large
future artifacts in Git LFS, a release, Hugging Face, or object storage and record
their URI plus checksum in the manifest.

An experiment must record:

- `run_id`, UTC start/end timestamps, seed, complete arguments/configuration;
- source notebook/script SHA-256 and dataset/input SHA-256;
- model ID and exact revision;
- Python, framework, CUDA and package versions;
- allocated hardware, dtype and decoding/training parameters;
- checkpoint/resume status and evaluator/trainer version.

Use atomic artifact writes (`path.tmp` then rename) so downloads never capture a
partially written JSON/JSONL file. Include a run signature derived from all
behavior-changing configuration and reject incompatible checkpoints.

## Choose the execution mode

### Persistent session

Use `new` + `exec` when the task:

- runs a notebook;
- requires uploads or downloads;
- has multiple stages;
- needs retained kernel or filesystem state;
- may require debugging or reruns.

### One-shot script

Use `colab run` for a self-contained Python script whose outputs are printed or
otherwise retrieved without a follow-up session:

```bash
colab run --gpu T4 --timeout 3600 script.py
```

Use `--keep -s <name>` only when follow-up access is required. Otherwise allow
`run` to release the VM automatically.

## Persistent notebook workflow

Set a descriptive name mentally and substitute it consistently below.

### 1. Preflight

```bash
colab sessions
colab version
```

Confirm local inputs exist before allocating compute. Validate the notebook
locally when practical. Confirm the source has:

- bounded runtime/step/token limits;
- deterministic seeds where supported;
- periodic, atomic checkpoints;
- a configuration-safe resume signature;
- explicit output paths under the run directory;
- no code that prints environment variables or tokens.

Hash source and immutable inputs before upload. Record hashes in the run config;
do not infer reproducibility later from filenames.

### 2. Provision

CPU:

```bash
colab new -s <session>
```

GPU:

```bash
colab new -s <session> --gpu T4
```

Immediately verify the allocated runtime:

```bash
colab status -s <session>
```

If provisioning is interrupted, run `colab sessions` before retrying. An
interrupted command may still have allocated a billable VM.

For independent experiments, prefer fresh sessions to prevent hidden state from
imports, variables, patched packages, RNG state, or caches in a previous run.
Kernel state persists across `colab exec` calls. If reusing a session, restart the
kernel, reinstall/verify dependencies, reinject `.env`, and rerun the smoke test.

For parallel agents, isolate CLI session state with the global flag on every
command:

```bash
colab --config "/tmp/colab-<run_id>.json" new -s <session> --gpu T4
```

When isolation is enabled, prepend that same `--config` path to every subsequent
`status`, `upload`, `exec`, `download`, `log`, `ls`, and `stop` command. Never mix
isolated and default state for one session.

### 3. Prepare remote directories

`colab upload` does not reliably create missing parent directories. Create them
first through Python:

```bash
echo "from pathlib import Path; Path('/content/data').mkdir(parents=True, exist_ok=True)" \
  | colab exec -s <session>
```

Create output directories the same way when the notebook does not create them.
Inline Python must be piped on stdin; passing it as a positional argument to
`colab exec` is rejected.

Install pinned dependencies after provisioning:

```bash
colab install -s <session> -r requirements-colab.txt
```

Do not silently upgrade to unpinned latest packages during a benchmark. Record
the resolved versions in `runtime.json`.
If packages are changed in a live kernel, restart it, reinject `.env`, and rerun
the import smoke test. Import caches survive package installation. Upgrade
compatibility-linked packages together; for example, current PEFT may require a
newer `torchao` than Colab preinstalls.

### 4. Upload inputs

```bash
colab upload -s <session> \
  "data/input.jsonl" \
  "/content/data/input.jsonl"
```

Use absolute remote paths. Upload only dependencies not already embedded in the
script or notebook. A file passed with `colab exec -f` is sent for execution and
does not need a separate upload.

### 5. Inject the project `.env` securely

Colab Secrets (`google.colab.userdata`) cannot be accessed by notebooks launched
with `colab exec`. For this project, inject the local `.env` into the persistent
kernel before running the notebook.

Never read, print, summarize, log, or include `.env` contents in tool output.
Do not pass secret values directly on a command line because shell history and
process listings may expose them. Confirm `.env` is ignored by Git before use.

Upload the file without inspecting it:

```bash
colab upload -s <session> ".env" "/content/.env"
```

Load its values into `os.environ` and immediately delete the remote copy:

```bash
colab exec -s <session> \
  -f ".cursor/skills/colab-cli/scripts/load-env.py" \
  --timeout 60
```

The loader reports only the number of variables loaded; it never prints names or
values. It deletes `/content/.env` in a `finally` block, including when parsing
fails.

The loader accepts blank lines, full-line and inline comments, optional `export`,
ASCII shell-style variable names, and quoted or unquoted single-line values.
Double-quoted values support `\\`, `\"`, `\n`, `\r`, and `\t`. It does not
support multiline values or variable interpolation such as `${NAME}`; those
strings remain literal. It validates the complete file before changing
`os.environ`, so malformed input cannot partially inject secrets.

If upload succeeds but loader execution fails or is interrupted, remove the
remote plaintext file before any retry:

```bash
colab rm -s <session> "/content/.env"
```

If an experiment aborts after injection, stop the session or restart its kernel;
deleting `/content/.env` does not remove secrets already held in kernel memory.

Run the target notebook on the same session after the loader succeeds:

```bash
colab exec -s <session> -f "notebook.ipynb" --timeout 3600
```

Notebook code reads secrets normally:

```python
import os

token = os.environ["HF_TOKEN"]
```

Kernel environment variables persist across separate `colab exec` calls in the
same session. Repeat this injection for every new session or after
`colab restart-kernel`. Never save environment values into notebook outputs,
metrics, logs, exceptions, or downloaded artifacts.

### 6. Smoke test

Before the full run, execute the exact production path with a tiny bounded sample
or step count. Do not maintain a separate simplified implementation.

The smoke test must prove:

- imports and accelerator placement succeed;
- dataset/model paths resolve;
- one complete inference or training step finishes;
- checkpoint and resume code can read what it writes;
- every required artifact is created and parseable;
- no secret appears in output or artifacts.

Inspect the produced artifact schema and at least one record. Fix failures before
starting the expensive run. Use a distinct smoke-test `run_id` or clear its
outputs so they cannot be aggregated with the full experiment.

### 7. Execute

Notebook:

```bash
colab exec -s <session> -f "notebook.ipynb" --timeout 3600
```

Python script:

```bash
colab upload -s <session> "script.py" "/content/script.py"
echo "import runpy,sys; sys.argv=['script.py']; _=runpy.run_path('/content/script.py',run_name='__main__')" \
  | colab exec -s <session> --timeout 3600
```

Use the `runpy` wrapper for reproducible script execution. Code sent with
`colab exec -f script.py` runs as a notebook cell: `__file__` may be undefined,
and the persistent kernel may retain stale `sys.argv`. Add script arguments
explicitly to the assigned list.

Notebook execution writes `<notebook>_output.ipynb` locally beside the source
notebook. It does not overwrite the source notebook.

#### Preserve the executed notebook

The output notebook is a required experiment artifact. Validate and preserve it
before stopping the session, keep it distinct from session-history exports, and
never retain outputs containing secrets. Follow the detailed preservation and
recovery procedure in [reference.md](reference.md).

For long commands, start them as a background shell job through the agent's shell
tool. Check once that execution started. Monitor closely only when a hang,
reclamation, or expensive failure needs intervention.

For sampled or multi-seed work, make each seed independently recoverable. Write
and download one seed's artifacts before starting the next, then aggregate only
from verified files.

### 8. Monitor and diagnose

```bash
colab status -s <session>
colab log -s <session> -n 30
```

Do not assume missing streamed `tqdm` output means a hang; carriage-return
progress can be buffered. Prefer application checkpoints or status evidence.
Do not trust the local CLI exit code alone: a remote Python traceback can still
produce exit code zero. Require an explicit completion sentinel plus parseable
final artifacts.

If the local control connection times out, first check `status`, `log`, and the
remote result directory. Retry only when those prove the execution did not
start; otherwise a retry can duplicate an active run.

When the application writes incremental progress, download a checkpoint to a
temporary local path and inspect it:

```bash
colab download -s <session> \
  "/content/experiments/<run_id>/predictions.jsonl" \
  "/tmp/predictions-progress.jsonl"
```

If the kernel is wedged:

```bash
colab restart-kernel -s <session>
```

Recreate the session only after confirming the existing one cannot be recovered.

Healthy execution evidence is advancing checkpoints, completed-example counts,
or application sentinels—not merely a BUSY runtime. If progress stops beyond the
expected interval, inspect status, logs, GPU use from notebook instrumentation,
and the newest checkpoint before intervening.

### 9. Download and verify artifacts immediately

```bash
colab download -s <session> \
  "/content/experiments/<run_id>/metrics.json" \
  "outputs/evaluations/<run_id>/metrics.json"

colab download -s <session> \
  "/content/experiments/<run_id>/predictions.jsonl" \
  "outputs/evaluations/<run_id>/predictions.jsonl"
```

Verify each local file exists and is parseable before releasing the VM.
Also verify:

- expected row/example/step counts;
- required JSON keys and finite metric values;
- run signatures match across artifacts;
- no duplicate IDs and no train/eval/test overlap when relevant;
- local SHA-256 values are recorded;
- the executed output notebook exists and contains the final summary.

Export CLI history for long or failed runs:

```bash
colab log -s <session> \
  -o "outputs/evaluations/<run_id>/colab-session.ipynb"
```

Export this while the session is healthy and again after completion. The executed
`<notebook>_output.ipynb`, session-history notebook, metrics, predictions,
checkpoints, config, and runtime metadata are distinct artifacts; preserve each
under the same `run_id`.

For multi-seed or multi-stage evaluations:

1. Run one bounded unit.
2. Write that unit's predictions and metrics.
3. Download and verify them immediately.
4. Continue to the next unit.
5. Aggregate locally from downloaded per-unit metrics.

Prefer one seed per fresh session for runs near Colab's reclamation window. Do
not wait until every seed finishes to download all artifacts.

### 10. Stop and verify cleanup

```bash
colab stop -s <session>
colab sessions
```

Stopping is mandatory even after failures or interrupted local commands. Idle
sessions consume compute.

Before stopping, list the remote result directory and compare it with the local
artifact inventory:

```bash
colab ls -s <session> "/content/experiments/<run_id>"
```

After stopping, `colab sessions` must show no unintended active assignment.

## Authentication

First test with the read-only command:

```bash
colab sessions
```

Follow the installed CLI's current help because authentication defaults can vary
by version:

```bash
colab -h
```

If browser OAuth is required, tell the user clearly and let them complete the
interactive consent. Do not repeatedly retry definitive authentication or quota
failures.

`colab auth -s <session>` configures credentials inside the VM; it does not fix
CLI authentication. Do not invoke interactive `auth`, `drivemount`, `repl`, or
`console` from a non-interactive agent shell.

## Failure handling

Preserve recoverable artifacts before intervention, distinguish local CLI
timeouts from remote failure, and never overwrite or hand-edit failed results.
Use the failure-specific actions in [reference.md](reference.md).

## Completion report

Report:

- session and accelerator used;
- command or notebook executed;
- whether execution succeeded;
- local paths of downloaded artifacts and executed notebook;
- verification performed;
- confirmation that the session was stopped;
- any missing artifact or reproducibility caveat.

Use one of these final statuses:

- `COMPLETE`: all acceptance checks passed and the session is stopped.
- `PARTIAL`: useful artifacts recovered, but explicitly listed items are missing.
- `FAILED`: experiment result is invalid; diagnostics and recovery action listed.

Never report `COMPLETE` while a required artifact is missing or a session remains
active.
