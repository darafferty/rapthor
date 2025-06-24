"""
Tests for the plot_rapthor_timing script.
"""

import pytest

try:
    from rapthor.scripts.plot_rapthor_timing import (
    MainLogParser,
    SubLogParser,
    make_cycle_pdfs_sublogs,
    main
)
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
    parser = SubLogParser(logdir, operation)
    yield parser
    
class TestMainLogParser:
    
    def test_add_key(self, main_log_parser):
        indict = {'key1': 'value1', 'key2': 'value2'}
        inkey = 'key1'
        newkey = 'new_key'
        newval = 'new_value'
        main_log_parser.add_key(indict, inkey, newkey, newval)
        
    def test_group_by_cycle(self, main_log_parser):
        cycledict = {'cycle1': ['log1', 'log2'], 'cycle2': ['log3']}
        main_log_parser.group_by_cycle(cycledict)
        
    def test_process(self, main_log_parser):
        main_log_parser.process()
        
    def test_plot(self, main_log_parser):
        main_log_parser.plot()

class TestSubLogParser:
    def test_get_run_times(self, sub_log_parser):
        sub_log_parser.get_run_times()
        
    def test_plot(self, sub_log_parser):
        sub_log_parser.plot()
        
def test_make_cycle_pdfs_sublogs():
    make_cycle_pdfs_sublogs()
    
def test_main():
    logdir = "logs"
    main(logdir, detailed=True)
