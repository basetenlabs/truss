class Model:
    def load(self):
        import time

        time.sleep(3)

    def predict(self, request):
        return {"prediction": [1]}
