"""``python -m pinneapple.check [paths]`` is ``pinneapple check [paths]``."""
import sys

from pinneapple_check.cli import check_main

sys.exit(check_main())
