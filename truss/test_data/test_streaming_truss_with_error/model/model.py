from typing import Any, Dict, List

import helpers_1


class Model:
    def predict(self, model_input: Any) -> Dict[str, List]:
        def inner():
            for i in range(5):
                # Raise error partway through if throw_error is set
                if i == 3 and model_input.get("throw_error"):
                    helpers_1.foo(123)
                yield str(i)

        return inner()
