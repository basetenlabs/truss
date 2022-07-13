from typing import Dict, List

from .parser_utils import extract_pdf_data



class InvoiceParserModel(object):
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
