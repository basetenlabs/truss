import time

from locust import HttpUser, between, task


class ModelUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict(self):
        for i in range(1, 9):
            self.client.post(
                f"http://localhost:8080/v1/models/sbert_{i}:predict", json={}
            )
