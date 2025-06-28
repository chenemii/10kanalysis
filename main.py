#!/usr/bin/env python3
"""
SEC Filing Analyzer - Main Entry Point

A tool to analyze SEC 10-K filings for hiring and workforce challenges using RAG (Retrieval-Augmented Generation).

Usage:
    python main.py [options] [tickers...]

Examples:
    python main.py AAPL MSFT                    # Analyze Apple and Microsoft
    python main.py --all                        # Analyze default set of companies
    python main.py --query-company AAPL         # Query existing Apple data
    python main.py --compare AAPL MSFT          # Compare companies
    python main.py --status-only                # Show database status only
"""

import logging
from cli import main

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    main() 