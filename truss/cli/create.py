from InquirerPy import inquirer


def ask_name() -> str:
    return inquirer.text(message="What's the name of your model?").execute()
