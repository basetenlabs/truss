import json

from rich import box
from rich.table import Column, Table


def clean_dataset(df):
    df = df.loc[df["source"] == "Gathered"]
    return df


def create_row_output_text(df_row):
    # parse ingredients correctly
    ingredients = json.loads(df_row["ingredients"])
    ingredients = list(map(lambda x: f" - {x} \n ", ingredients))
    ingredients = " ".join(ingredients)
    ingredients = f"Ingredients \n {ingredients}"

    # parse directions
    directions = json.loads(df_row["directions"])
    directions = list(map(lambda x: f" - {x} \n ", directions))
    directions = " ".join(directions)
    directions = f"Directions \n {directions}"
    return f" {ingredients} \n \n {directions} "


def display_df(df, console):
    """display dataframe in ASCII format"""

    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)
