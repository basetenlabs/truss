import re

SECRET_NAME_MATCH_REGEX = re.compile(r'^[-._a-zA-Z0-9]+$')


def validate_secret_name(secret_name: str):
    if secret_name is None or not isinstance(secret_name, str) or secret_name == '':
        raise ValueError(f'Invalid secret name `{secret_name}`')

    def constraint_violation_msg():
        return f'Constraint violation for {secret_name}'

    if len(secret_name) > 253:
        raise ValueError(f'Secret name `{secret_name}` is longer than max allowed 253 chars.')

    if not SECRET_NAME_MATCH_REGEX.match(secret_name):
        raise ValueError(constraint_violation_msg() + ', invalid characters found in secret name.')

    if secret_name == '.':
        raise ValueError(constraint_violation_msg() + ', secret name cannot be `.`')

    if secret_name == '..':
        raise ValueError(constraint_violation_msg() + ', secret name cannot be `..`')
