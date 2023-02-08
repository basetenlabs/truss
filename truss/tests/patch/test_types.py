from truss.patch.signature import calc_truss_signature
from truss.patch.types import TrussSignature


def test_truss_signature_type(custom_model_truss_dir):
    sign = calc_truss_signature(custom_model_truss_dir)
    assert TrussSignature.from_dict(sign.to_dict()) == sign
