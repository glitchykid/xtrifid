"""Collect the Ubuntu toolchain's copyright records and frozen Python tools' licenses."""

import gzip
import io
import subprocess
import sysconfig
import tarfile
from importlib.metadata import distributions
from pathlib import Path


def reproducible_header(info):
    info.mtime = info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


with (
    Path("dist/xtrifid-linux-x86_64-notices.tar.gz").open("wb") as output,
    gzip.GzipFile(fileobj=output, filename="", mode="wb", mtime=0) as compressed,
    tarfile.open(fileobj=compressed, mode="w", dereference=True) as archive,
):
    versions = subprocess.check_output(
        ["dpkg-query", "-W", "-f", r"${Package}\t${Version}\n"], text=True
    )
    for source in sorted(Path("/usr/share/doc").glob("*/copyright")):
        archive.add(
            source, f"ubuntu/{source.parent.name}/copyright", filter=reproducible_header
        )
    archive.add(
        "/usr/share/common-licenses", "ubuntu/common-licenses", filter=reproducible_header
    )
    archive.add("/opt/ollama/licenses", "ollama", filter=reproducible_header)
    archive.add("packaging/OLLAMA-LICENSE", "ollama/LICENSE", filter=reproducible_header)
    for notice in sorted(Path("xtrifid/assets").glob("*.md")):
        archive.add(notice, f"anatomy/{notice.name}", filter=reproducible_header)
    versions = "Ubuntu build environment\n" + versions + "\nPyPI build environment\n"
    for dist in sorted(
        distributions(path=[sysconfig.get_path("purelib")]), key=lambda d: d.metadata["Name"]
    ):
        versions += f"{dist.metadata['Name']}\t{dist.version}\n"
        for entry in dist.files or []:
            if entry.name.lower().startswith(("license", "copying", "copyright")):
                archive.add(
                    dist.locate_file(entry), f"pypi/{entry}", filter=reproducible_header
                )
    manifest = versions.encode()
    info = tarfile.TarInfo("versions.txt")
    info.size = len(manifest)
    archive.addfile(info, io.BytesIO(manifest))
