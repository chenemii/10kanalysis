import os
import time
import logging
from typing import List, Dict, Any

from sec_client import SECClient
from text_processor import TextProcessor
from vector_store import VectorStore
from cache_manager import CacheManager
from ai_analyzer import AIAnalyzer
from hiring_queries import QUERY_CATEGORIES

logger = logging.getLogger(__name__)


class SECFilingAnalyzer:
    """
    Main orchestrator for SEC filing analysis with RAG capabilities
    """
    
    def __init__(self, openai_api_key: str = None, cache_dir: str = "./filing_cache"):
        """
        Initialize the SEC Filing Analyzer
        
        Args:
            openai_api_key: OpenAI API key for GPT queries (optional)
            cache_dir: Directory to cache downloaded filings
        """
        # Initialize all components
        self.sec_client = SECClient()
        self.text_processor = TextProcessor()
        self.vector_store = VectorStore()
        self.cache_manager = CacheManager(cache_dir)
        self.ai_analyzer = AIAnalyzer(openai_api_key)
        
        # CIK to company mapping for better readability
        self.cik_to_company = {}
        
        # Minimum similarity score threshold for meaningful results
        self.min_similarity_score = 0.10
    
    def ticker_to_cik(self, ticker: str) -> str:
        """
        Convert ticker symbol to CIK, checking cache first
        
        Args:
            ticker: Ticker symbol or CIK
            
        Returns:
            CIK string
        """
        ticker = ticker.upper()
        
        # If it's already a CIK (10 digits), return as-is
        if len(ticker) == 10 and ticker.isdigit():
            return ticker
        
        # Check if we already have the mapping cached
        for cik, cached_ticker in self.cik_to_company.items():
            if cached_ticker == ticker:
                return cik
        
        # Try to get CIK from SEC API
        try:
            cik = self.sec_client.get_company_cik(ticker)
            self.cik_to_company[cik] = ticker
            return cik
        except ValueError:
            # If not found, return the original string
            return ticker
    
    def get_available_companies(self) -> List[str]:
        """
        Get list of companies that have data in the database
        
        Returns:
            List of company ticker symbols
        """
        company_ciks = self.vector_store.get_available_companies()
        companies = []
        
        for cik in company_ciks:
            # Try to get the ticker symbol, fall back to CIK if not found
            ticker = self.cik_to_company.get(cik, cik)
            
            # If we still have a CIK, try to resolve it from SEC data
            if ticker == cik and len(cik) == 10 and cik.isdigit():
                ticker = self.sec_client.resolve_cik_to_ticker(cik)
                if ticker != cik:  # Successfully resolved
                    self.cik_to_company[cik] = ticker
            
            companies.append(ticker)
        
        return sorted(companies)
    
    def get_company_stats(self, ticker: str = None) -> Dict:
        """
        Get statistics for a specific company or all companies
        
        Args:
            ticker: Company ticker symbol (optional)
            
        Returns:
            Dictionary with statistics
        """
        if ticker:
            # Stats for specific company
            try:
                cik = self.ticker_to_cik(ticker)
                collection = self.vector_store.get_company_collection(cik, ticker)
                count = collection.count()
                return {
                    "company": ticker.upper(),
                    "total_chunks": count,
                    "collection_name": collection.name
                }
            except Exception as e:
                return {"company": ticker.upper(), "error": str(e)}
        else:
            # Stats for all companies
            stats = self.vector_store.get_collection_stats()
            
            # Convert CIKs to tickers in the stats
            if 'companies' in stats:
                updated_companies = {}
                for cik, company_stats in stats['companies'].items():
                    ticker = self.cik_to_company.get(cik, cik)
                    updated_companies[ticker] = company_stats
                stats['companies'] = updated_companies
            
            return stats
    
    def process_company_filings(self, ticker: str, filing_count: int = 3) -> None:
        """
        Process all 10-K filings for a company with intelligent parsing
        
        Args:
            ticker: Stock ticker
            filing_count: Number of recent filings to process
        """
        logger.info(f"Processing filings for {ticker}")
        
        try:
            # Get company CIK
            cik = self.sec_client.get_company_cik(ticker)
            self.cik_to_company[cik] = ticker.upper()
            logger.info(f"Found CIK {cik} for {ticker}")
            
            # Check what's already processed
            processed_filings = self.vector_store.get_processed_filings()
            already_processed = processed_filings.get(cik, set())
            
            if already_processed:
                logger.info(f"Found {len(already_processed)} already processed filings for {ticker} (CIK: {cik})")
            
            # Get 10-K filings
            filings = self.sec_client.get_10k_filings(cik, filing_count)
            logger.info(f"Found {len(filings)} 10-K filings")
            
            for filing in filings:
                accession_number = filing['accessionNumber']
                
                # Skip if already processed in ChromaDB
                if accession_number in already_processed:
                    logger.info(f"Skipping already processed filing {accession_number} from {filing['filingDate']}")
                    continue
                
                logger.info(f"Processing filing {accession_number} from {filing['filingDate']}")
                
                # Check if chunks already exist in ChromaDB
                if self.vector_store.is_filing_in_db(cik, accession_number):
                    logger.info(f"Chunks for filing {accession_number} already exist in database, skipping processing")
                    continue
                
                # Check if cached first
                if self.cache_manager.is_filing_cached(cik, accession_number):
                    logger.info(f"Loading filing {accession_number} from cache")
                    content_dict = self.cache_manager.load_filing_from_cache(cik, accession_number)
                else:
                    # Download and parse content
                    logger.info(f"Downloading filing {accession_number}")
                    content_dict = self.sec_client.download_filing_content(filing)
                    
                    # Extract sections using text processor
                    if content_dict['full_text']:
                        content_dict['sections'] = self.text_processor.extract_sections(content_dict['full_text'])
                    
                    # Cache the downloaded content
                    if content_dict['full_text']:
                        self.cache_manager.save_filing_to_cache(filing, content_dict)
                
                if not content_dict['full_text']:
                    logger.warning(f"No content found for filing {accession_number}")
                    continue
                
                logger.info(f"Extracted {len(content_dict['sections'])} sections: {list(content_dict['sections'].keys())}")
                
                # Create intelligent chunks
                chunk_dicts = self.text_processor.intelligent_chunking(content_dict, filing)
                logger.info(f"Created {len(chunk_dicts)} intelligent chunks")
                
                # Store all chunks in vector store
                if chunk_dicts:
                    self.vector_store.store_chunks(chunk_dicts, self.cik_to_company)
                    logger.info(f"Successfully processed and stored filing {accession_number}")
                
                # Rate limiting
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def query_rag(self, query: str, n_results: int = 5, companies: List[str] = None) -> List[Dict]:
        """
        Query the RAG system for relevant content
        
        Args:
            query: Search query
            n_results: Number of results to return per company
            companies: List of company tickers to search (None = all companies)
            
        Returns:
            List of relevant chunks with metadata
        """
        if companies is None:
            # Search all companies
            return self.vector_store.query_all_companies(query, n_results, self.cik_to_company)
        else:
            # Search specific companies
            all_results = []
            for ticker in companies:
                cik = self.ticker_to_cik(ticker.upper())
                company_name = self.cik_to_company.get(cik, ticker.upper())
                results = self.vector_store.query_by_cik(query, cik, company_name, n_results)
                all_results.extend(results)
            
            # Sort all results by similarity score
            all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return all_results
    
    def query_company_specific(self, query: str, ticker: str, n_results: int = 10) -> List[Dict]:
        """
        Query a specific company's data
        
        Args:
            query: Search query
            ticker: Company ticker symbol
            n_results: Number of results to return
            
        Returns:
            List of relevant chunks for the specific company
        """
        return self.query_rag(query, n_results, companies=[ticker])
    
    def compare_companies(self, query: str, companies: List[str], n_results: int = 5) -> Dict:
        """
        Compare specific companies on a particular query
        
        Args:
            query: Search query
            companies: List of company tickers to compare
            n_results: Number of results per company
            
        Returns:
            Dictionary with results organized by company
        """
        results_by_company = {}
        
        for ticker in companies:
            company_results = self.query_company_specific(query, ticker, n_results)
            filtered_results = [r for r in company_results if r['similarity_score'] >= self.min_similarity_score]
            
            # Always include the company in results, even with zero meaningful results
            results_by_company[ticker] = {
                'results': filtered_results,
                'count': len(filtered_results),
                'total_searched': len(company_results),
                'avg_similarity': sum(r['similarity_score'] for r in filtered_results) / len(filtered_results) if filtered_results else 0,
                'top_similarity': max(r['similarity_score'] for r in filtered_results) if filtered_results else (max(r['similarity_score'] for r in company_results) if company_results else 0),
                'threshold_used': self.min_similarity_score
            }
        
        return results_by_company
    
    def comprehensive_hiring_analysis(self) -> Dict:
        """
        Perform comprehensive hiring analysis using multiple semantic queries
        
        Returns:
            Comprehensive analysis results
        """
        all_results = []
        query_results = {}
        
        # Execute both general and specific semantic queries
        all_queries = QUERY_CATEGORIES['all']
        
        for query in all_queries:
            results = self.query_rag(query, n_results=15)
            # Filter by similarity score AND generic content
            filtered_results = []
            for r in results:
                if (r['similarity_score'] >= self.min_similarity_score and 
                    not self.text_processor.is_generic_content(r['document'])):
                    filtered_results.append(r)
                elif self.text_processor.is_generic_content(r['document']):
                    logger.debug(f"Filtered out generic content (score: {r['similarity_score']:.3f}): {r['document'][:100]}...")
            
            query_results[query] = filtered_results
            all_results.extend(filtered_results)
        
        if not all_results:
            return {
                "message": "No specific hiring-related content found above similarity threshold (generic content filtered out)",
                "threshold_used": self.min_similarity_score,
                "generic_filtering": "enabled"
            }
        
        # Deduplicate results by document ID
        seen_ids = set()
        unique_results = []
        for result in all_results:
            # Create unique ID from metadata
            result_id = f"{result['metadata']['cik']}_{result['metadata']['accession_number']}_{result['metadata']['chunk_index']}"
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        # Sort by similarity score
        unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Group by company and filing
        companies = {}
        for result in unique_results[:30]:  # Top 30 results
            cik = result['metadata']['cik']
            filing_date = result['metadata']['filing_date']
            section = result['metadata'].get('section', 'unknown')
            key = f"{cik}_{filing_date}"
            
            if key not in companies:
                company_name = self.cik_to_company.get(cik, f"Company-{cik}")
                companies[key] = {
                    'cik': cik,
                    'company_name': company_name,
                    'filing_date': filing_date,
                    'accession_number': result['metadata']['accession_number'],
                    'sections_found': set(),
                    'relevant_segments': [],
                    'avg_similarity': 0,
                    'top_concerns': []
                }
            
            companies[key]['sections_found'].add(section)
            companies[key]['relevant_segments'].append({
                'text': result['document'],
                'section': section,
                'similarity_score': result['similarity_score'],
                'priority': result['metadata'].get('priority', 'unknown')
            })
        
        # Calculate average similarity scores and convert sections_found to list
        for company_data in companies.values():
            if company_data['relevant_segments']:
                avg_sim = sum(seg['similarity_score'] for seg in company_data['relevant_segments']) / len(company_data['relevant_segments'])
                company_data['avg_similarity'] = avg_sim
                company_data['sections_found'] = list(company_data['sections_found'])
        
        # Sort companies by relevance
        sorted_companies = sorted(companies.values(), key=lambda x: x['avg_similarity'], reverse=True)
        
        # Generate AI analysis if available
        ai_analysis = self.ai_analyzer.analyze_hiring_query_results(
            query_results, sorted_companies, unique_results, self.cik_to_company
        )
        
        return {
            "total_relevant_segments": len(unique_results),
            "companies_analyzed": len(companies),
            "similarity_threshold": self.min_similarity_score,
            "generic_filtering": "enabled",
            "query_breakdown": {query: len(results) for query, results in query_results.items()},
            "company_details": ai_analysis['company_details'],
            "top_findings": unique_results[:10],
            "ai_summary": ai_analysis['ai_summary'],
            "section_distribution": self._get_section_distribution(unique_results)
        }
    
    def _get_section_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Get distribution of findings across different 10-K sections"""
        distribution = {}
        for result in results:
            section = result['metadata'].get('section', 'unknown')
            distribution[section] = distribution.get(section, 0) + 1
        return distribution
    
    def persist_database(self) -> None:
        """Persist vector database data to disk"""
        self.vector_store.persist_database()
    
    def migrate_old_collection(self) -> bool:
        """
        Migrate data from old single collection to company-specific collections
        
        Returns:
            True if migration was performed, False if no migration needed
        """
        return self.vector_store.migrate_old_collection(self.cik_to_company)
    
    def initialize_company_mappings(self) -> None:
        """Initialize CIK to ticker mappings for existing collections"""
        try:
            company_ciks = self.vector_store.get_available_companies()
            ciks_to_resolve = []
            
            for cik in company_ciks:
                if cik not in self.cik_to_company and len(cik) == 10 and cik.isdigit():
                    ciks_to_resolve.append(cik)
            
            # Resolve CIKs to tickers
            if ciks_to_resolve:
                logger.info(f"Resolving {len(ciks_to_resolve)} CIKs to ticker symbols...")
                for cik in ciks_to_resolve:
                    ticker = self.sec_client.resolve_cik_to_ticker(cik)
                    if ticker != cik:  # Successfully resolved
                        logger.info(f"Resolved CIK {cik} to ticker {ticker}")
                        self.cik_to_company[cik] = ticker
                        
        except Exception as e:
            logger.warning(f"Error initializing company mappings: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cached files"""
        return self.cache_manager.get_cache_stats() 