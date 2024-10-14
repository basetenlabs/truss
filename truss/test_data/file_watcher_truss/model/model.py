import json

FILEPATH = "/tmp/environment"


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def setup_environment(self, environment: dict):
        print("setup_environment called with", environment)
        environment_name = environment.get("environment_name", None)
        if environment_name == "production":
            print("DOING IT LIVE")
        else:
            print("DOING IT IN", environment_name)

    def load(self):
        # Load model here and assign to self._model.
        pass

    def write_to_tmp_environment(self, data: dict):
        # Ensure the data is in string format
        data_str = json.dumps(data)
        try:
            with open(FILEPATH, "w") as file:
                file.write(data_str)
            print(f"Data {data_str} successfully written to {FILEPATH}")
        except Exception as e:
            print(f"An error occurred while writing to {FILEPATH}: {e}")

    def predict(self, model_input: dict):
        # Run model inference here
        print(f"Writing {model_input} to {FILEPATH}")
        self.write_to_tmp_environment(model_input)
        return model_input
