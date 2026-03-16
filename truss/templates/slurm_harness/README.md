# SLURM Harness for Baseten Training

Run SLURM jobs on Baseten's training infrastructure. A persistent CPU login node runs `slurmctld`, and GPU worker nodes run `slurmd` and execute your batch scripts.

## Architecture

```
 You (local machine)
  |
  |-- truss train slurm login --> Login Node (CPU, persistent)
  |                                 - runs slurmctld
  |                                 - watches for new workers
  |
  |-- truss train slurm sbatch -> Worker Nodes (GPU, ephemeral)
                                    - run slurmd
                                    - node 0 submits sbatch after all workers register
                                    - exit when the SLURM job completes
```

Login and worker nodes discover each other through a shared filesystem (`BT_PROJECT_CACHE_DIR`), which is available to all jobs in the same project. The login node writes its IP, munge key, and `slurm.conf` to this cache; workers read from it.

## Prerequisites

```bash
pip install 'truss @ git+https://github.com/basetenlabs/truss.git@rcano/slurm-cli'
truss login  # or set BASETEN_API_KEY
```

## Usage

### 1. Start the login node

```bash
truss train slurm login --project my-slurm-cluster
```

This pushes a CPU-only job that runs `slurmctld`. It stays alive indefinitely, accepting worker registrations and job submissions. Wait for `LOGIN_READY` in the logs before submitting jobs.

Options:

```bash
# Use a specific GPU partition (default is CPU-only)
truss train slurm login --project my-slurm-cluster -p H100

# Self-test: automatically push a test worker to verify the setup
truss train slurm login --project my-slurm-cluster --self-test

# Custom base image
truss train slurm login --project my-slurm-cluster --image my-registry/my-image:latest
```

### 2. Submit jobs

```bash
# Run a script
truss train slurm sbatch train.sh --project my-slurm-cluster

# Inline command
truss train slurm sbatch --wrap "hostname && nvidia-smi" --project my-slurm-cluster

# Multi-node with custom resources
truss train slurm sbatch train.sh -N 4 --gres gpu:8 -p H200 --project my-slurm-cluster

# Custom job name and image
truss train slurm sbatch train.sh -J my-training-run --image nvcr.io/nvidia/pytorch:24.01-py3 --project my-slurm-cluster
```

Each `sbatch` call pushes a new set of GPU worker nodes. The login node detects them, reconfigures `slurmctld`, and the workers run your script via SLURM's `sbatch`.

**On a login node:** If you're SSH'd into the login node, you can run `truss train slurm sbatch` directly — the `--project` flag is auto-detected from `/workspace/runtime_config.json`.

### 3. Monitor

```bash
# Worker logs (includes sbatch output)
truss train logs --job-id <worker-job-id>

# Login node logs
truss train logs --job-id <login-job-id>

# Job status
truss train view --job-id <job-id>

# SSH into the login node (interactive session)
truss train isession --job-id <login-job-id>
# Follow the instructions to authenticate, then connect via VS Code/Cursor remote tunnels
```

## CLI Reference

### `truss train slurm login`

| Flag | Default | Description |
|---|---|---|
| `--project` | `slurm-harness` | Project name (shared cache scope) |
| `--gpus-per-node` | `8` | GPUs per worker |
| `--partition`, `-p` | none (CPU-only) | GPU type (e.g. H100, H200, A100) |
| `--self-test` | off | Push a test worker from inside the login node |
| `--image` | `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` | Base Docker image |
| `--remote` | auto-detected | Baseten remote name |

### `truss train slurm sbatch`

| Flag | Default | Description |
|---|---|---|
| `SCRIPT` (positional) | | Path to batch script |
| `--wrap` | | Inline command (alternative to script) |
| `--nodes`, `-N` | `1` | Number of worker nodes |
| `--gres` | `gpu:8` | GPU resources per node |
| `--partition`, `-p` | `H200` | GPU type (e.g. H100, H200, A100) |
| `--job-name`, `-J` | `slurm-worker` | Name for the worker job |
| `--project` | auto-detected | Project name (must match login node) |
| `--image` | `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` | Base Docker image |
| `--remote` | auto-detected | Baseten remote name |

## How It Works

1. **Login node starts**: Installs SLURM, generates a munge key, writes its IP and key to shared cache, then waits for workers.

2. **Workers start**: Install SLURM, write their IPs to shared cache, copy the munge key, wait for `slurm.conf` to appear with their hostname, then start `slurmd`.

3. **Login detects workers**: Workers write their count to the shared cache. Once all worker IPs appear, the login node generates `slurm.conf` with the worker hostnames/IPs and starts `slurmctld`.

4. **Worker 0 submits the job**: After `sinfo` shows all workers as idle, worker 0 writes the batch script (passed via env var) and runs `sbatch`.

5. **Job runs and workers exit**: Workers monitor the SLURM job and exit when it completes.

6. **Login persists**: The login node keeps running in a watcher loop. When you run `sbatch` again, it pushes new workers; the login detects the new job ID, regenerates `slurm.conf`, and restarts `slurmctld`.
