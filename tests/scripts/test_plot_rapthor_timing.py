"""
Tests for the plot_rapthor_timing script.
"""

import pytest

try:
    from rapthor.scripts.plot_rapthor_timing import (MainLogParser,
                                                     SubLogParser, main,
                                                     make_cycle_pdfs_sublogs)
except ImportError as e:
    pytest.skip(f"Skipping tests due to ImportError: {e}", allow_module_level=True)


@pytest.fixture
def main_log_parser():
    log_file = "rapthor.log"
    parser = MainLogParser(log_file)
    yield parser


@pytest.fixture
def sub_log_parser():
    logdir = "logs"
    operation = "calibrate_1"
    try:
        parser = SubLogParser(logdir, operation)
    except ValueError:
        pytest.skip(f"Skipping SubLogParser test due to ValueError: {logdir} or {operation} not found")
    yield parser


class TestMainLogParser:
    def test_add_key(self, main_log_parser):
        # indict = {'key1': 'value1', 'key2': 'value2'}
        # inkey = 'key1'
        # newkey = 'new_key'
        # newval = 'new_value'
        # main_log_parser.add_key(indict, inkey, newkey, newval)
        pass

    def test_group_by_cycle(self, main_log_parser):
        # cycledict = {'cycle1': ['log1', 'log2'], 'cycle2': ['log3']}
        # main_log_parser.group_by_cycle(cycledict)
        pass

    def test_process(self, main_log_parser):
        # main_log_parser.process()
        pass

    def test_plot(self, main_log_parser):
        # main_log_parser.plot()
        pass


class TestSubLogParser:
    def test_get_run_times(self, sub_log_parser):
        # sub_log_parser.get_run_times()
        pass

    def test_plot(self, sub_log_parser):
        # sub_log_parser.plot()
        pass


def test_make_cycle_pdfs_sublogs():
    # make_cycle_pdfs_sublogs()
    pass


def test_main():
    # logdir = "logs"
    # main(logdir, detailed=True)
    pass
