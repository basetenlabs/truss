from pathlib import Path

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Column, Table
from t5_trainer import t5_train
from torch import cuda
from util import clean_dataset, create_row_output_text


class Train:
    def __init__(
        self,
        config,
        output_dir: Path,
        variables: dict,
        secrets,
    ):
        self._config = config
        self._output_dir = output_dir
        self._variables = variables
        self._secrets = secrets

    def train(self):
        # Write your training code here, populating generated artifacts in
        # self._output_dir
        # let's define model parameters specific to T5
        model_params = {
            "MODEL": "t5-base",  # model_type: t5-base/t5-large
            "TRAIN_BATCH_SIZE": 1,  # training batch size
            "VALID_BATCH_SIZE": 1,  # validation batch size
            "TRAIN_EPOCHS": 1,  # number of training epochs
            "VAL_EPOCHS": 1,  # number of validation epochs
            "LEARNING_RATE": 3e-4,  # learning rate
            "MAX_SOURCE_TEXT_LENGTH": 128,  # max length of source text
            "MAX_TARGET_TEXT_LENGTH": 512,  # max length of target text
            "SEED": 28,  # set seed for reproducibility
        }

        # get gpu device
        device = "cuda" if cuda.is_available() else "cpu"

        # create logger for train statistics
        console = Console(record=True)
        training_logger = Table(
            Column("Epoch", justify="center"),
            Column("Steps", justify="center"),
            Column("Loss", justify="center"),
            title="Training Status",
            pad_edge=False,
            box=box.ASCII,
        )

        # load dataset in and preprocess data
        df = pd.read_csv(self._variables["dataset_url"])
        df = clean_dataset(df)
        df["target"] = df.apply(create_row_output_text, axis=1)

        t5_train(
            dataframe=df,
            source_text="title",
            target_text="target",
            model_params=model_params,
            device=device,
            console=console,
            training_logger=training_logger,
            output_dir=str(self._output_dir),
        )
