import time

from locust import HttpUser, between, task


class ModelUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict(self):
        for _ in range(1, 9):
            self.client.post(f"http://localhost:9000/v1/models/model:predict", json={})
