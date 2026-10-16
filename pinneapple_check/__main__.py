"""``python -m pinneapple_check [paths]`` is ``pinneapple check [paths]``."""
import sys

from .cli import check_main

if __name__ == "__main__":                 # importing the module (pkgutil.walk_packages, docs) must not run it
    sys.exit(check_main())
