def pytest_addoption(parser):
    parser.addoption("--project", default="slurm-e2e-test")
    parser.addoption("--image", default=None)
    parser.addoption("--docker-auth-method", default=None)
    parser.addoption("--docker-auth-secret", default=None)
    parser.addoption("--partition", default="H200")
    parser.addoption("--remote", default="baseten")
