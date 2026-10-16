import sys

from .cli import main

if __name__ == "__main__":                 # importing the module (pkgutil.walk_packages, docs) must not run it
    sys.exit(main(["benchmark", *sys.argv[1:]] if sys.argv[1:2] and sys.argv[1] != "benchmark" else sys.argv[1:]))
