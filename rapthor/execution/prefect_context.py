"""Small Prefect context helpers used by execution adapters."""


def in_prefect_run_context() -> bool:
    """Return True when called inside an active Prefect run context."""
    try:
        from prefect.context import get_run_context

        get_run_context()
    except Exception:
        return False
    return True
