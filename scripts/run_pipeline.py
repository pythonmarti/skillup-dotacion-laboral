#!/usr/bin/env python3
"""Runner generico para stages de pipeline por dominio."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.domains.cli import build_parser, run_from_args


def main() -> None:
    parser = build_parser(default_stage="full")
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
