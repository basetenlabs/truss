from truss.patch.custom_types import TrussSignature
from truss.patch.signature import calc_truss_signature


def test_truss_signature_type(custom_model_truss_dir):
    sign = calc_truss_signature(custom_model_truss_dir)
    assert TrussSignature.from_dict(sign.to_dict()) == sign
