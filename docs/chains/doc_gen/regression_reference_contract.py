"""Check the curated reference against public SDK signatures and fields."""

import inspect
import re
import unittest
from pathlib import Path

import truss_chains as chains

REFERENCE = Path(__file__).with_name("API-reference.mdx").read_text()


def section(symbol):
    match = re.search(
        r"^### .*`" + re.escape(symbol) + r"`.*$", REFERENCE, re.MULTILINE
    )
    if match is None:
        raise AssertionError(f"Missing section: {symbol}")
    end = REFERENCE.find("\n### ", match.end())
    return REFERENCE[match.end() : end if end != -1 else None]


class ReferenceContractTests(unittest.TestCase):
    def test_constructor_and_push_parameters_are_documented(self):
        for symbol, function in (
            ("truss_chains.Compute", chains.Compute.__init__),
            ("truss_chains.Assets", chains.Assets.__init__),
            ("truss_chains.push", chains.push),
        ):
            text = section(symbol)
            for name in inspect.signature(function).parameters:
                if name != "self":
                    with self.subTest(symbol=symbol, parameter=name):
                        self.assertIn(f"| `{name}`", text)

    def test_optional_internal_url_is_documented(self):
        row = next(
            line
            for line in section("truss_chains.DeployedServiceDescriptor").splitlines()
            if line.startswith("| `internal_url`")
        )
        self.assertIn(r"\|None", row)

    def test_enum_member_names_are_exact(self):
        text = section("truss_chains.BasetenImage")
        names = re.findall(r"^\| `([^`]+)`", text, re.MULTILINE)
        self.assertEqual(set(names), set(chains.BasetenImage.__members__))


if __name__ == "__main__":
    unittest.main()
