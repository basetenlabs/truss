# BDN dataset mounts

Sync your dataset into a BDN volume before submitting a training job. Enable BDN
volumes for your organization, then declare read-only mounts in your Python config:

```python
import truss_train as train

job = train.TrainingJob(
    image=train.Image(base_image="your-training-image:latest"),
    compute=train.Compute(accelerator="H100:8"),
    bdn=train.BDNConfig(
        mounts=[
            train.BDNVolumeMount(
                source="bdn:datasets/plaid:prod",
                path="/data/training",
                fetch_mode="eager",
            )
        ]
    ),
    runtime=train.Runtime(start_commands=["python train.py --data /data/training"]),
)
```

The API payload uses the same `bdn.mounts` shape as dedicated inference. Each
mount accepts `source`, `path`, `fetch_mode`, and optional `include`/`exclude`
relative glob lists. `fetch_mode` defaults to `eager`: fetch the volume in the
background while allowing reads immediately. `lazy` skips the bulk download and
uses on-demand reads and BDN's existing read-ahead. Neither waits for the whole
dataset to fit on local disk. Benchmark both with your actual data loader;
neither guarantees a particular throughput.

Submission resolves each tag, head, or digest prefix to a full immutable digest.
Every worker and retry uses that version, and recreating a job preserves it.
The job response includes `resolved_source` alongside each requested `source`.
To use a newly synced dataset, submit a new job. Files are ordinary read-only
files at the requested path; write checkpoints elsewhere using the existing
checkpoint configuration.

The platform support must be deployed before using these fields. BDN mounts
require a personal or creator service account API key and a supported Baseten
Cloud workload plane/GPU. Private/customer-installed hosting remains subject to
the existing BDN restrictions. This feature does not increase volume sync limits
or add checkpoint publication to BDN.
