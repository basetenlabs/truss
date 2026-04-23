from truss.base.truss_config import TrussConfig
from truss.base.vllm_config import VLLMConfiguration
from truss.contexts.image_builder.serving_image_builder import ServingImageBuilder


def test_prepare_vllm_build_dir_sets_base_image_and_docker_server(tmp_path):
    truss_dir = tmp_path / "truss"
    truss_dir.mkdir()
    config = TrussConfig(
        vllm=VLLMConfiguration(
            model="meta-llama/Llama-2-7b-hf", port=8000, tensor_parallel_size=1
        )
    )
    config.write_to_yaml_file(truss_dir / "config.yaml")

    builder = ServingImageBuilder(truss_dir)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    builder.prepare_image_build_dir(build_dir)

    # Verify base_image was set to vllm image
    assert builder._spec.config.base_image is not None
    assert builder._spec.config.base_image.image == "vllm/vllm-openai:v0.19.1"

    # Verify docker_server was configured
    assert builder._spec.config.docker_server is not None
    assert builder._spec.config.docker_server.server_port == 8000
    assert builder._spec.config.docker_server.predict_endpoint == "/v1/chat/completions"
    assert builder._spec.config.docker_server.readiness_endpoint == "/health"
    assert builder._spec.config.docker_server.liveness_endpoint == "/health"
    assert builder._spec.config.docker_server.start_command.startswith(
        "vllm serve meta-llama/Llama-2-7b-hf"
    )

    # Verify Dockerfile uses the vllm base image
    dockerfile = (build_dir / "Dockerfile").read_text()
    assert "FROM vllm/vllm-openai:v0.19.1" in dockerfile
