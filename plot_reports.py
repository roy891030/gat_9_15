# -*- coding: utf-8 -*-
"""
Backward-compatible report entrypoint.

`plot_reports.py` and `evaluate_portfolio.py` had duplicated logic.
This file now delegates to `evaluate_portfolio.py` so there is one source of truth.
CLI usage remains unchanged.
"""

from evaluate_portfolio import main


if __name__ == "__main__":
    main()
