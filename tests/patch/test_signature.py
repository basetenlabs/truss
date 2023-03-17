from truss.patch.signature import calc_truss_signature


def test_calc_truss_signature(custom_model_truss_dir):
    sign = calc_truss_signature(custom_model_truss_dir)
    assert len(sign.content_hashes_by_path) > 0
    assert "config.yaml" in sign.content_hashes_by_path
    with (custom_model_truss_dir / "config.yaml").open() as config_file:
        assert config_file.read() == sign.config
