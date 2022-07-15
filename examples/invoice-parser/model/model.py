from typing import Dict, List

from .parser_utils import extract_pdf_data


class InvoiceParserModel(object):
    """
    This model does not work out of the box. You must provide
    creds to access the Google Cloud service used here. These
    creds can be placed in the `config.yaml` file (under secrets).
    To access these secrets, you can find them in the `_config` object
    under `_config["secrets"][KEY_NAME]`.
    """
    def __init__(self, **kwargs) -> None:
        self._config = kwargs['config']

    def predict(self, request: Dict) -> Dict[str, List]:
        inputs = request['inputs']
        parsed_invoices = []
        secret = self._config['secrets']
        for url in inputs:
            parsed_invoices.append(
                {
                    "parsed_data": extract_pdf_data(url, secret),
                    "pdf_url": url,
                }
            )
        return parsed_invoices
