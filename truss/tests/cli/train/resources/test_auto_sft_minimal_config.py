"""Minimal AutoSFT config (required fields only)."""

from truss_train import AutoSFT

auto_sft = AutoSFT(
    model="gpt2",
    dataset="wikitext-2-raw-v1",
    num_epochs=1,
)
