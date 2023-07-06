from truss.templates.server.common.logging import APIJsonFormatter


def test_log_parsing():
    formatter = APIJsonFormatter()
    log_record = {
        "message": '172.17.0.1:52518 - "POST /v1/models/model%3Apredict HTTP/1.1" 200'
    }

    formatter.process_log_record(log_record)
    assert log_record["message"] == "POST /predict 200 OK"

    log_record = {
        "message": '172.17.0.1:52518 - "POST /v1/models/model%3Apredict_binary HTTP/1.1" 500'
    }
    formatter.process_log_record(log_record)
    assert log_record["message"] == "POST /predict_binary 500 INTERNAL_ERROR"

    log_record = {
        "message": '172.17.0.1:52518 - "POST /v1/models/model%3Apredict HTTP/1.1" 503'
    }
    formatter.process_log_record(log_record)
    assert log_record["message"] == "POST /predict 503 SERVICE_UNAVAILABLE"

    # test below guarantees any other logs are untouched
    log_record = {
        "message": "2023/07/05 14:19:12 POST http://localhost:8090/v1/models/model:predict"
    }
    formatter.process_log_record(log_record)
    assert (
        log_record["message"]
        == "2023/07/05 14:19:12 POST http://localhost:8090/v1/models/model:predict"
    )
