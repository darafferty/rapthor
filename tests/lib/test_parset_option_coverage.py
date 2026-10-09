import configparser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARSET = REPO_ROOT / "rapthor" / "settings" / "defaults.parset"
TEST_ROOT = REPO_ROOT / "tests"
THIS_FILE = Path(__file__).resolve()

# Use this only when an option is intentionally covered outside direct pytest
# references, or when focused coverage is explicitly deferred with a reason.
OPTION_COVERAGE_ALLOWLIST: dict[tuple[str, str], str] = {}


def _default_options():
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read(DEFAULT_PARSET)
    return {
        (section, option) for section in parser.sections() for option in parser.options(section)
    }


def _test_python_text():
    chunks = []
    for path in sorted(TEST_ROOT.rglob("*.py")):
        if path.resolve() == THIS_FILE:
            continue
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def _format_option_list(options):
    return "\n".join(f"  - [{section}] {option}" for section, option in options)


def test_default_parset_options_have_test_attention_or_explicit_reason():
    test_text = _test_python_text()
    unmentioned_options = {
        (section, option) for section, option in _default_options() if option not in test_text
    }
    unrecorded_options = sorted(unmentioned_options - set(OPTION_COVERAGE_ALLOWLIST))

    assert not unrecorded_options, (
        "Every option in rapthor/settings/defaults.parset needs direct test "
        "attention or an explicit reason in OPTION_COVERAGE_ALLOWLIST. Missing:\n"
        f"{_format_option_list(unrecorded_options)}"
    )


def test_default_parset_option_coverage_allowlist_is_current():
    default_options = _default_options()
    test_text = _test_python_text()
    allowlisted_options = set(OPTION_COVERAGE_ALLOWLIST)

    unknown_options = sorted(allowlisted_options - default_options)
    now_tested_options = sorted(
        (section, option)
        for section, option in allowlisted_options & default_options
        if option in test_text
    )
    missing_reasons = sorted(
        (section, option)
        for (section, option), reason in OPTION_COVERAGE_ALLOWLIST.items()
        if not reason.strip()
    )

    assert not unknown_options, (
        f"Remove stale option coverage allow-list entries:\n{_format_option_list(unknown_options)}"
    )
    assert not now_tested_options, (
        "Remove allow-list entries once direct tests mention the option:\n"
        f"{_format_option_list(now_tested_options)}"
    )
    assert not missing_reasons, (
        "Every option coverage allow-list entry needs a reason:\n"
        f"{_format_option_list(missing_reasons)}"
    )
