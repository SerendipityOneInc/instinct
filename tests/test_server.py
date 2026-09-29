"""instinct-serve request handling with a fake model (no GPU, no weights)."""

import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from instinct.server import MAX_BODY_BYTES, MAX_QUESTIONS, make_handler


class FakeModel:
    """Answers every question with a uniform distribution; can be told to fail."""

    def __init__(self):
        self.error = None

    def decide(self, record):
        if self.error is not None:
            raise self.error
        ids = [c["id"] for c in record["criteria"]]
        probabilities = {cid: 1 / len(ids) for cid in ids}
        return {"type": record["primitive"], "probabilities": probabilities,
                "prediction": ids[0], "prompt_tokens": 1}


@pytest.fixture
def server():
    model = FakeModel()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(model, "fake"))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, model
    httpd.shutdown()
    httpd.server_close()


def post(httpd, body=None, headers=None):
    conn = http.client.HTTPConnection(*httpd.server_address, timeout=10)
    conn.putrequest("POST", "/v1/systemone")
    for key, value in (headers or {}).items():
        conn.putheader(key, value)
    conn.endheaders(body)
    response = conn.getresponse()
    payload = json.loads(response.read())
    conn.close()
    return response.status, payload


def post_json(httpd, value):
    data = json.dumps(value).encode()
    return post(httpd, data, {"Content-Length": str(len(data))})


def request(n=1):
    return {"state": "s", "questions": {f"q{i}": {"type": "noul", "instructions": "i"}
                                       for i in range(n)}}


def test_valid_request(server):
    status, payload = post_json(server[0], request())
    assert status == 200 and payload["answers"]["q0"] == {"type": "noul", "noul": 0.5}


@pytest.mark.parametrize("value,status", [
    ("-1", 400), ("abc", 400), (str(MAX_BODY_BYTES + 1), 413),
])
def test_content_length_is_clamped(server, value, status):
    assert post(server[0], b"", {"Content-Length": value})[0] == status


def test_missing_content_length(server):
    assert post(server[0])[0] == 411


def test_invalid_json(server):
    assert post(server[0], b"{", {"Content-Length": "1"})[0] == 400


def test_deeply_nested_json(server):
    data = b"[" * 100000 + b"]" * 100000
    # Python < 3.14 raises RecursionError while parsing (400); 3.14 parses it and
    # the body is rejected as a non-object request (422). Either way, no crash.
    assert post(server[0], data, {"Content-Length": str(len(data))})[0] in (400, 422)


def test_question_cap(server):
    assert post_json(server[0], request(MAX_QUESTIONS))[0] == 200
    status, payload = post_json(server[0], request(MAX_QUESTIONS + 1))
    assert status == 422 and str(MAX_QUESTIONS) in payload["error"]


def test_invalid_request_is_422(server):
    assert post_json(server[0], {"state": "s", "questions": {}})[0] == 422


def test_recursion_during_scoring_is_400(server):
    server[1].error = RecursionError()
    assert post_json(server[0], request())[0] == 400


def test_unexpected_error_is_500_json(server):
    server[1].error = RuntimeError("boom")
    status, payload = post_json(server[0], request())
    assert status == 500 and payload == {"error": "internal server error"}
