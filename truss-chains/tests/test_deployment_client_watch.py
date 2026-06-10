import pathlib
from types import SimpleNamespace

from truss_chains.deployment import deployment_client


class SelectedChainlet:
    pass


class OtherChainlet:
    pass


def _descriptor(
    *external_package_dirs,
    chainlet_cls=None,
    display_name="Chainlet",
    is_truss_chainlet=False,
):
    docker_image = SimpleNamespace(
        external_package_dirs=[
            SimpleNamespace(abs_path=str(path)) for path in external_package_dirs
        ]
    )
    remote_config = SimpleNamespace(docker_image=docker_image)
    chainlet_cls = chainlet_cls or SimpleNamespace()
    chainlet_cls.remote_config = remote_config
    return SimpleNamespace(
        display_name=display_name,
        chainlet_cls=chainlet_cls,
        is_truss_chainlet=is_truss_chainlet,
        src_path=str(pathlib.Path(__file__)),
        truss_dir=external_package_dirs[0] if is_truss_chainlet else None,
    )


def test_chain_watch_roots_include_external_package_dirs(tmp_path):
    chain_root = tmp_path / "chain"
    external_packages = tmp_path / "packages"
    chain_root.mkdir()
    external_packages.mkdir()

    roots, included_paths = deployment_client._get_chain_watch_paths(
        chain_root, [_descriptor(external_packages)]
    )

    assert roots == [chain_root.resolve(), external_packages.resolve()]
    assert included_paths is None


def test_chain_watch_roots_dedupe_external_package_dirs(tmp_path):
    chain_root = tmp_path / "chain"
    external_packages = tmp_path / "packages"
    chain_root.mkdir()
    external_packages.mkdir()

    roots, included_paths = deployment_client._get_chain_watch_paths(
        chain_root, [_descriptor(external_packages), _descriptor(external_packages)]
    )

    assert roots == [chain_root.resolve(), external_packages.resolve()]
    assert included_paths is None


def test_chain_watch_roots_skip_nested_external_package_dirs(tmp_path):
    chain_root = tmp_path / "chain"
    external_packages = chain_root / "packages"
    external_packages.mkdir(parents=True)

    roots, included_paths = deployment_client._get_chain_watch_paths(
        chain_root, [_descriptor(external_packages)]
    )

    assert roots == [chain_root.resolve()]
    assert included_paths is None


def test_chain_watch_roots_skip_truss_chainlet_external_package_dirs(tmp_path):
    chain_root = tmp_path / "chain"
    external_packages = tmp_path / "packages"
    chain_root.mkdir()
    external_packages.mkdir()

    roots, included_paths = deployment_client._get_chain_watch_paths(
        chain_root, [_descriptor(external_packages, is_truss_chainlet=True)]
    )

    assert roots == [chain_root.resolve()]
    assert included_paths is None


def test_selected_truss_chainlet_watch_roots_include_truss_external_packages(
    tmp_path, monkeypatch
):
    chain_root = tmp_path / "chain"
    truss_dir = tmp_path / "tts"
    external_packages = tmp_path / "packages"
    chain_root.mkdir()
    truss_dir.mkdir()
    external_packages.mkdir()
    expected_truss_dir = truss_dir

    class StubTrussHandle:
        def __init__(self, truss_dir):
            assert truss_dir == expected_truss_dir
            self.spec = SimpleNamespace(external_package_dirs_paths=[external_packages])

    monkeypatch.setattr(deployment_client.truss_handle, "TrussHandle", StubTrussHandle)

    roots, included_paths = deployment_client._get_chain_watch_paths(
        chain_root,
        [
            _descriptor(
                truss_dir,
                chainlet_cls=SelectedChainlet,
                display_name="TTS",
                is_truss_chainlet=True,
            )
        ],
        included_chainlets={"TTS"},
    )

    assert roots == [truss_dir.resolve(), external_packages.resolve()]
    assert included_paths == [truss_dir.resolve(), external_packages.resolve()]


def test_chain_watch_roots_only_include_selected_chainlet_dirs(tmp_path):
    chain_root = tmp_path / "chain"
    external_packages = tmp_path / "packages"
    chain_root.mkdir()
    external_packages.mkdir()

    roots, included_paths = deployment_client._get_chain_watch_paths(
        chain_root,
        [
            _descriptor(
                external_packages,
                chainlet_cls=SelectedChainlet,
                display_name="Selected",
            ),
            _descriptor(chainlet_cls=OtherChainlet, display_name="Other"),
        ],
        included_chainlets={"Selected"},
    )

    assert roots == [
        pathlib.Path(__file__).parent.resolve(),
        external_packages.resolve(),
    ]
    assert included_paths == [
        pathlib.Path(__file__).resolve(),
        external_packages.resolve(),
    ]


def test_chain_watch_filter_ignores_sibling_paths_for_selected_chainlet(tmp_path):
    chain_root = tmp_path / "chain"
    chain_root.mkdir()
    chain_file = chain_root / "chain.py"
    chain_file.write_text("# chain")
    orchestrator_packages = chain_root / "orchestrator" / "packages" / "agent"
    orchestrator_packages.mkdir(parents=True)
    orchestrator_file = orchestrator_packages / "agent.py"
    orchestrator_file.write_text("# orchestrator")
    tts_file = chain_root / "tts" / "data" / "do.sh"
    tts_file.parent.mkdir(parents=True)
    tts_file.write_text("# tts")

    _, watch_filter = deployment_client._create_watch_filter(
        chain_root, [chain_file.resolve(), orchestrator_packages.resolve()]
    )

    assert watch_filter(None, str(chain_file))
    assert watch_filter(None, str(orchestrator_file))
    assert not watch_filter(None, str(tts_file))
