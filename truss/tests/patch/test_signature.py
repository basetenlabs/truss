from truss.truss_handle.patch.signature import calc_truss_signature


def test_calc_truss_signature(custom_model_truss_dir):
    sign = calc_truss_signature(custom_model_truss_dir)
    assert len(sign.content_hashes_by_path) > 0
    assert "config.yaml" in sign.content_hashes_by_path
    with (custom_model_truss_dir / "config.yaml").open() as config_file:
        assert config_file.read() == sign.config


def test_calc_truss_signature_ignore_data_dir(custom_model_data_dir):
    sign = calc_truss_signature(custom_model_data_dir, ["data/*"])
    assert (
        len(sign.content_hashes_by_path) > 0
        and "data/foo.bar" not in sign.content_hashes_by_path
    )
    assert "config.yaml" in sign.content_hashes_by_path
