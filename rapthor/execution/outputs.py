"""Compatibility imports for finalizer-compatible output records.

The record contract now lives in :mod:`rapthor.lib.records` because both
operation finalizers and execution flows use the same file/directory records.
"""

from rapthor.lib.records import (
    OUTPUT_CLASSES,
    OutputRecordError,
    directory_record,
    directory_record_path,
    file_record,
    file_record_path,
    is_output_record,
    optional_directory_record_path,
    optional_file_record_path,
    optional_output_record_path,
    output_record_path,
    validate_output_record,
)

__all__ = [
    "OUTPUT_CLASSES",
    "OutputRecordError",
    "directory_record",
    "directory_record_path",
    "file_record",
    "file_record_path",
    "is_output_record",
    "optional_directory_record_path",
    "optional_file_record_path",
    "optional_output_record_path",
    "output_record_path",
    "validate_output_record",
]
