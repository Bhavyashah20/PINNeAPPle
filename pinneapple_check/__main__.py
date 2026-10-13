"""``python -m pinneapple_check [paths]`` is ``pinneapple check [paths]``."""
import sys

from .cli import check_main

sys.exit(check_main())
