from truss.contexts.image_builder.cache_warmer import split_gs_path

def test_split_gs_path():
    path = "gs://bucket-name/prefix"
    bucket_name, prefix = split_gs_path(path)
    assert bucket_name == "bucket-name"
    assert prefix == "prefix"