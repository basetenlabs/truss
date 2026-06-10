from types import SimpleNamespace

from truss_chains.deployment import deployment_client


def _descriptor(*external_package_dirs, is_truss_chainlet=False):
    docker_image = SimpleNamespace(
        external_package_dirs=[
            SimpleNamespace(abs_path=str(path)) for path in external_package_dirs
        ]
    )
    remote_config = SimpleNamespace(docker_image=docker_image)
    chainlet_cls = SimpleNamespace(remote_config=remote_config)
    return SimpleNamespace(
        chainlet_cls=chainlet_cls,
        is_truss_chainlet=is_truss_chainlet,
    )


def test_chain_watch_roots_include_external_package_dirs(tmp_path):
    chain_root = tmp_path / "chain"
    external_packages = tmp_path / "packages"
    chain_root.mkdir()
    external_packages.mkdir()

    roots = deployment_client._get_chain_watch_roots(
        chain_root, [_descriptor(external_packages)]
    )

    assert roots == [chain_root.resolve(), external_packages.resolve()]


def test_chain_watch_roots_dedupe_external_package_dirs(tmp_path):
    chain_root = tmp_path / "chain"
    external_packages = tmp_path / "packages"
    chain_root.mkdir()
    external_packages.mkdir()

    roots = deployment_client._get_chain_watch_roots(
        chain_root,
        [_descriptor(external_packages), _descriptor(external_packages)],
    )

    assert roots == [chain_root.resolve(), external_packages.resolve()]


def test_chain_watch_roots_skip_nested_external_package_dirs(tmp_path):
    chain_root = tmp_path / "chain"
    external_packages = chain_root / "packages"
    external_packages.mkdir(parents=True)

    roots = deployment_client._get_chain_watch_roots(
        chain_root, [_descriptor(external_packages)]
    )

    assert roots == [chain_root.resolve()]


def test_chain_watch_roots_skip_truss_chainlet_external_package_dirs(tmp_path):
    chain_root = tmp_path / "chain"
    external_packages = tmp_path / "packages"
    chain_root.mkdir()
    external_packages.mkdir()

    roots = deployment_client._get_chain_watch_roots(
        chain_root, [_descriptor(external_packages, is_truss_chainlet=True)]
    )

    assert roots == [chain_root.resolve()]
