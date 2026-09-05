"""Reject an empty or partially skipped required integration test run."""

import sys
from pathlib import Path
from xml.etree import ElementTree


def main(path: Path) -> None:
    cases = ElementTree.parse(path).findall(".//testcase")
    if not cases:
        raise SystemExit("Required integration suite collected no tests")
    unsuccessful = [
        case.get("name") for case in cases if any(case.find(tag) is not None for tag in ("skipped", "failure", "error"))
    ]
    if unsuccessful:
        raise SystemExit(f"Required integration tests did not all pass: {unsuccessful}")
    print(f"All {len(cases)} required integration tests passed without skips")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
