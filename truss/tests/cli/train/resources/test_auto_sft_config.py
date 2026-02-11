"""Example AutoSFT config for testing."""

from truss_train import AutoSFT

auto_sft = AutoSFT(
    model="meta-llama/Llama-2-7b-hf",
    dataset="imdb",
    num_epochs=3,
    optimizer="adamw",
    learning_rate=2e-5,
    lr_scheduler="cosine",
)
