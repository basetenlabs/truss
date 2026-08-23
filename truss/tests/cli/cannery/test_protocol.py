import io
import json

from truss.cli.cannery.protocol import Phase0ProtocolConsumer


def test_phase_zero_consumer_retains_bounded_progress_state():
    def progress_lines():
        for completed in range(10_000):
            yield (
                json.dumps(
                    {
                        "protocol_version": 1,
                        "type": "progress",
                        "operation": "push",
                        "phase": "upload",
                        "bytes_done": completed,
                    }
                )
                + "\n"
            )

    session = Phase0ProtocolConsumer().start(
        io.StringIO('{"protocol_version":1}'), progress_lines(), lambda _message: None
    )

    assert session.read_result() == {"protocol_version": 1}
    session.finish()
    assert session.last_phase == "upload"
    assert session.terminal_error is None
    assert not any(isinstance(value, list) for value in vars(session).values())
