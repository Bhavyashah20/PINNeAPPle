import sys

from .cli import main

sys.exit(main(["benchmark", *sys.argv[1:]] if sys.argv[1:2] and sys.argv[1] != "benchmark" else sys.argv[1:]))
