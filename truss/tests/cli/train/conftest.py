def pytest_addoption(parser):
    parser.addoption("--project", default="slurm-e2e-test")
    parser.addoption("--image", default=None)
    parser.addoption("--docker-auth-method", default=None)
    parser.addoption("--docker-auth-secret", default=None)
    parser.addoption("--partition", default="H200")
    parser.addoption("--remote", default="baseten")
    parser.addoption("--script", default=None, help="Path to sbatch script file")
    parser.addoption("--nodes", default="1", help="Number of worker nodes")
