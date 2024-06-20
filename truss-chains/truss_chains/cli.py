import rich
from InquirerPy import inquirer


def inquire_copy_confirm(msg: str) -> bool:
    rich.print(msg)
    return inquirer.confirm(
        "Do you want to proceed with deploying and copying "
        "all data (⚠️ proceed with cation)?",
        default=False,
    ).execute()
