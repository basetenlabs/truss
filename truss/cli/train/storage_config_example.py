from truss_train import definitions

# Implicit Storage: Allows multiple jobs to write to same underlying storage, but requires
# specific logic around huggingface writes.
runtime = definitions.Runtime(
    init_commands=[
        # Allows user to fill the cache using the same image as the training job
        # Baseten will run this in an init container on the leader pod. Our LWS
        # start up mode guarantees that worker nodes will only come up after write is complete.
        "python fill_cache.py"
    ],
    start_commands=["pip install -r requirements.txt", "python train.py"],
    environment_variables={"MY_ENV_VAR": "my-value"},
    cache_config=definitions.CacheConfig(
        # we use a name for the cache so that it can be referenced by other jobs
        # consider: having a default name that is like {project-name}-cache
        name="training-cache-1",
        enabled=True,
        # we put size limits on the cache; we will need org-level constraints
        size_gb=3000,
        # Not necessary, but can be used to provide protections on the cache
        access_mode=definitions.CacheAccessMode.READ_WRITE_ALL,  # or READ_ONLY
    ),
)


# Declaritive Storage; Works independent of the underlying storage layer
runtime = definitions.Runtime(
    init_commands=[
        # Allows user to fill the cache using the same image as the training job
        # Baseten will run this in an init container on the leader pod. Our LWS
        # start up mode guarantees that worker nodes will only come up after write is complete.
        "python fill_cache.py"
    ],
    start_commands=["pip install -r requirements.txt", "python train.py"],
    environment_variables={"MY_ENV_VAR": "my-value"},
    cache_config=definitions.CacheConfig(
        # we use a nme for the cache so that it can be referenced by other jobs
        # consider: having a default name that is like {project-name}-cache
        name="training-cache-1",
        enabled=True,
        # we put size limits on the cache; we will need org-level constraints
        size_gb=3000,
        # this access mode is necessary to ensure that the cache is only written to
        # by one job at a time.
        access_mode=definitions.CacheAccessMode.LEADER_WRITE,  # or READ_ONLY
    ),
)

# Block Storage: leader_write
runtime = definitions.Runtime(
    start_commands=[
        "pip install -r requirements.txt",
        # FOOTGUN: in multinode scenarios, the user needs to implement logic
        # s.t. only the leader is writing to the cache
        # FOOTGUN: in multinode scenarios training job timeouts will have to be
        # handled by the user.
        "python fill_cache.pypython train.py",
    ],
    environment_variables={"MY_ENV_VAR": "my-value"},
    cache_config=definitions.CacheConfig(
        name="training-cache-1",
        enabled=True,
        # we put size limits on the cache; we will need org-level constraints
        size_gb=3000,
        # this access mode is necessary to ensure that the cache is only written to
        # by one job at a time.
        access_mode=definitions.CacheAccessMode.LEADER_WRITE,  # or READ_ONLY
    ),
)

job = definitions.TrainingJob(name="my-job", runtime=runtime)

print(job)
