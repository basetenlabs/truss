import rich
from InquirerPy import inquirer


def inquire_copy_confirm(msg: str) -> bool:
    rich.print(msg)
    return inquirer.confirm(
        "Do you want to proceed deploying and copy all data (⚠️ proceed with cation)?",
        default=False,
    ).execute()
