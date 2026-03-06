# Truss CLI Agent Context

## Authentication

Set `BASETEN_API_KEY` as an environment variable. Never use interactive login.

```bash
export BASETEN_API_KEY="your-api-key"
```

## Essential Commands

### Deploy a model

```bash
# Published deployment (production-ready, autoscaling)
truss push --remote baseten --model-name "my-model" --non-interactive

# Development deployment (fast iteration, live reload)
truss push --remote baseten --model-name "my-model" --watch --non-interactive
```

### Iterate on a development deployment

```bash
# Re-attach to existing development deployment
truss watch --remote baseten --model-name "my-model" --non-interactive
```

### Call a model

```bash
# Falls back to model name from config.yaml in the truss directory
truss predict --remote baseten -d '{"prompt": "hello"}'

# By model ID (targets development deployment by default)
truss predict --remote baseten --model "model-id" -d '{"prompt": "hello"}'

# By deployment ID (targets specific deployment)
truss predict --remote baseten --model-deployment "deploy-id" -d '{"prompt": "hello"}'

# Target the published (production) deployment
truss predict --remote baseten --published -d '{"prompt": "hello"}'
```

### Deploy a chain

```bash
truss chains push my_chain.py --name "my-chain" --remote baseten --non-interactive
```

## Rules for Agents

- Always use `--non-interactive` to disable prompts (available on `push`, `watch`, `chains push`, and most commands via `common_options`)
- Always use `--remote baseten` (or whatever remote is configured)
- For `predict`, pass JSON with `-d` or `-f path/to/file.json`
- `truss push` without `--watch` creates a published deployment (production)
- `truss push --watch` creates a development deployment and starts watching for code changes
- `truss watch` re-attaches to an existing development deployment (fails if none exists)
- `--promote` deploys directly to the production environment
- `--environment staging` deploys to a named environment
- `config.yaml` is the source of truth for model resources, dependencies, and settings
- Model names are set in `config.yaml` under `model_name:` or overridden with `--model-name`

## Configuration (config.yaml)

Key fields:

```yaml
model_name: my-model
resources:
  accelerator: L4        # GPU type (L4, A10G, A100, H100, etc.)
  instance_type: "L4:4x16"  # Or specify exact instance
requirements:
  - torch
  - transformers
secrets:
  hf_access_token: null  # Set in Baseten workspace
environment_variables: {}
```

## Common Mistakes to Avoid

- Do NOT use `truss login` in automated contexts — use `BASETEN_API_KEY` env var
- Do NOT omit `--non-interactive` — the CLI may prompt for input and hang
- Do NOT confuse `truss push --watch` (creates new dev deployment) with `truss watch` (re-attaches to existing)
- Do NOT use `--promote` with `--watch` — they are mutually exclusive
- Do NOT use `--environment` with `--watch` — they are mutually exclusive
- Do NOT use `--wait` with `--tail` — they are mutually exclusive
- Do NOT use `--promote` with `--environment` — they are mutually exclusive
