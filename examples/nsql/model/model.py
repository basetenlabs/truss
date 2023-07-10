from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self, **kwargs) -> None:
        self.tokenizer = None
        self.model = None

    def load(self):
        # Load model here and assign to self._model.
        self.tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-350M")
        self.model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-350M")

    def predict(self, model_input: Any) -> Any:
        schema = model_input["schema"]
        query = model_input["query"]

        prompt = generate_prompt(schema, query)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        generated_ids = self.model.generate(input_ids, max_length=500)
        result = f"SELECT{self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)[len(prompt):]}"

        return {"result": result}


def generate_prompt(schema: str, query: str):
    # Example schema:
    #
    # CREATE TABLE stadium (
    #     stadium_id number,
    #     location text,
    #     name text,
    #     capacity number,
    # )

    text = f"""{schema}

-- Using valid SQLite, answer the following questions for the tables provided above.

-- {query}

SELECT"""

    return text
