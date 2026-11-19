"""Check the ELF architecture and glibc symbol versions in the bootloader and its payload."""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader


def required_glibc(path):
    version_info = subprocess.check_output(["readelf", "--wide", "--version-info", path])
    versions = re.findall(rb"\bName: GLIBC_(\d+(?:\.\d+)+)\b", version_info)
    return max((tuple(map(int, v.split(b"."))) for v in versions), default=(0,))


binary = Path(sys.argv[1])
archive = CArchiveReader(binary)
required = {binary.name: required_glibc(binary)}
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "object"
    for name, entry in archive.toc.items():
        if entry[-1] != "b":
            continue
        data = archive.extract(name)
        if not data.startswith(b"\x7fELF"):
            continue
        assert data[4:6] == b"\x02\x01" and data[18:20] == b"\x3e\x00", name
        path.write_bytes(data)
        required[name] = required_glibc(path)
highest = max(required, key=required.get)
version = ".".join(map(str, required[highest]))
print(f"{len(required)} ELF objects checked; highest required GLIBC_{version} in {highest}")
assert required[highest] <= (2, 39), "Bundle exceeds Ubuntu 24.04 glibc ABI"
