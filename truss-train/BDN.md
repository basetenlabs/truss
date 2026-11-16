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
                source="bdn:datasets/training-data:prod",
                path="/data/training",
            )
        ]
    ),
    runtime=train.Runtime(start_commands=["python train.py --data /data/training"]),
)
```

The API payload uses the same `bdn.mounts` shape as dedicated inference. Each
mount accepts `source`, `path`, and optional `include`/`exclude` relative glob
lists. Training mounts always use lazy fetching: BDN serves data on demand with
its existing read-ahead instead of bulk-downloading the dataset. This policy is
set by the platform and is not a job configuration option. Benchmark with your
actual data loader to establish throughput.

Training passes the requested source directly to BDN, just like dedicated
inference. Tags and head references retain BDN's normal resolution behavior;
training does not pin them at submission. Use an explicit immutable digest when
you want a fixed dataset version. Retries and recreated jobs preserve the
requested reference. Files are ordinary read-only files at the requested path;
write checkpoints elsewhere using the existing checkpoint configuration.

The platform support must be deployed before using these fields. BDN mounts
require a supported Baseten Cloud workload plane/GPU.
Private/customer-installed hosting remains subject to
the existing BDN restrictions. This feature does not increase volume sync limits
or add checkpoint publication to BDN.
