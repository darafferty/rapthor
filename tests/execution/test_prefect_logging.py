import logging

from prefect import flow
from prefect.testing.utilities import prefect_test_harness

from rapthor.execution import prefect_logging


class RecordingAPIHandler(logging.Handler):
    records: list[tuple[str, str]]

    def __init__(self, level=0):
        super().__init__(level=level)
        self.records = []

    def emit(self, record):
        self.records.append((record.name, self.format(record)))


def _active_bridge_handlers():
    return [
        handler
        for handler in logging.root.handlers
        if handler.name == prefect_logging.PREFECT_API_LOG_HANDLER_NAME
    ]


def test_publish_python_logs_to_prefect_is_noop_without_prefect_context(monkeypatch):
    handlers = []

    def fake_loader():
        def factory(level=0):
            handler = RecordingAPIHandler(level=level)
            handlers.append(handler)
            return handler

        return factory

    monkeypatch.setattr(prefect_logging, "_load_api_log_handler", fake_loader)

    with prefect_logging.publish_python_logs_to_prefect():
        logging.getLogger("rapthor:test").info("outside run")

    assert handlers == []


def test_publish_python_logs_to_prefect_forwards_rapthor_logs(monkeypatch):
    handlers = []

    def fake_loader():
        def factory(level=0):
            handler = RecordingAPIHandler(level=level)
            handlers.append(handler)
            return handler

        return factory

    monkeypatch.setattr(prefect_logging, "_load_api_log_handler", fake_loader)

    @flow
    def logging_flow():
        with prefect_logging.publish_python_logs_to_prefect("info"):
            with prefect_logging.publish_python_logs_to_prefect("info"):
                logging.getLogger("rapthor:test").info("visible message")
                logging.getLogger("rapthor.test").warning("also visible")
                logging.getLogger("not-rapthor").info("hidden message")
                logging.getLogger("rapthor:test").debug("below level")
            return list(handlers[0].records)

    with prefect_test_harness(server_startup_timeout=None):
        records = logging_flow()

    assert records == [
        ("rapthor:test", "visible message"),
        ("rapthor.test", "also visible"),
    ]
    assert len(handlers) == 1
    assert all("\x1b" not in message for _, message in records)
    assert _active_bridge_handlers() == []
