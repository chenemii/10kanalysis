# SEC Filing Analyzer

A tool to analyze SEC 10-K filings for hiring and workforce challenges using RAG (Retrieval-Augmented Generation).


### Core Modules

#### 1. `sec_client.py`
**Responsibility**: SEC API interactions and filing downloads
- Company CIK lookup
- 10-K filing retrieval
- Document content download
- Rate limiting and error handling

#### 2. `text_processor.py`
**Responsibility**: Text processing and analysis
- Text chunking with intelligent overlap
- Section extraction (Risk Factors, MD&A, Business, etc.)
- Generic content filtering
- Document preprocessing

#### 3. `vector_store.py`
**Responsibility**: ChromaDB vector database operations
- Company-specific collection management
- Embedding generation and storage
- Semantic search and querying
- Database migration and persistence

#### 4. `cache_manager.py`
**Responsibility**: File caching operations
- Filing content caching
- Cache statistics and management
- Disk space optimization

#### 5. `ai_analyzer.py`
**Responsibility**: OpenAI integration and AI analysis
- Company-specific summary generation
- Overall trend analysis
- Hiring challenge identification
- AI-powered insights

#### 6. `hiring_queries.py`
**Responsibility**: Query constants and patterns
- Semantic query definitions
- Category-based query organization
- Comparison query templates

### Application Layer

#### 7. `filing_analyzer.py`
**Responsibility**: Main orchestrator class
- Coordinates all modules
- Provides unified API
- Manages company mappings (CIK ↔ Ticker)
- Orchestrates complex workflows

#### 8. `cli.py`
**Responsibility**: Command-line interface
- Argument parsing
- User interaction handling
- Output formatting
- Error reporting

#### 9. `main.py`
**Responsibility**: Entry point
- Application startup
- Logging configuration
- CLI delegation


## Usage


```bash
# Basic usage
python main.py AAPL MSFT

# Query specific company
python main.py --query-company AAPL

# Compare companies
python main.py --compare AAPL MSFT GOOGL

# Show status only
python main.py --status-only

# List available companies
python main.py --list-companies
```

## Installation

```bash
pip install -r requirements.txt
```

## Module Dependencies

```
main.py
└── cli.py
    └── filing_analyzer.py
        ├── sec_client.py
        ├── text_processor.py
        ├── vector_store.py
        ├── cache_manager.py
        ├── ai_analyzer.py
        └── hiring_queries.py
```

## Configuration

Environment variables:
- `OPENAI_API_KEY`: Optional, for AI-powered analysis

Command-line options:
- `--cache-dir`: Cache directory location
- `--threshold`: Similarity score threshold
- `--filings`: Number of filings per company

## Development

To extend the analyzer:

1. **Add new data sources**: Implement new clients following `SECClient` pattern
2. **Add new text processors**: Extend `TextProcessor` or create specialized processors
3. **Add new AI models**: Extend `AIAnalyzer` with additional model integrations
4. **Add new storage backends**: Implement new stores following `VectorStore` pattern
