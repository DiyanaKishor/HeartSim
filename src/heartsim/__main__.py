import sys

from .cli import main

# The guard matters: with multiprocessing's "spawn" start method (Windows, macOS) worker
# processes re-import this module, and an unguarded call would re-run the CLI inside each worker.
if __name__ == "__main__":
    sys.exit(main())
