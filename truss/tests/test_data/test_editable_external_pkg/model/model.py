import local_pkg


class Model:
    def predict(self, request):
        return {"predictions": [local_pkg.VALUE]}


