from pydantic import BaseModel


class ModelInput(BaseModel):
    prompt: str


class ModelOutput(BaseModel):
    generated_text: str


class Model:
    def predict(self, model_input: ModelInput) -> ModelOutput:
        return ModelOutput(generated_text=model_input.prompt)
