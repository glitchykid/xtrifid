import os
import sys
from pathlib import Path

# Minimal desktops may have fonts and drivers but no system Fontconfig configuration.
if not Path("/etc/fonts/fonts.conf").is_file():
    os.environ.setdefault("FONTCONFIG_FILE", str(Path(sys._MEIPASS) / "fontconfig/fonts.conf"))

from xtrifid.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
