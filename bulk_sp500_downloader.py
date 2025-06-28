#!/usr/bin/env python3
"""
S&P 500 Bulk Filing Downloader and Analyzer

This program downloads all S&P 500 companies and their 10-K filings from 2015 to 2025,
then runs comprehensive hiring and workforce analysis queries.

Usage:
    python bulk_sp500_downloader.py [--start-year 2015] [--end-year 2025] [--dry-run]

Examples:
    python bulk_sp500_downloader.py                    # Download all S&P 500 from 2015-2025
    python bulk_sp500_downloader.py --start-year 2020   # Download from 2020-2025
    python bulk_sp500_downloader.py --dry-run          # Show what would be downloaded
"""

import os
import sys
import time
import logging
import argparse
import requests
import pandas as pd
from typing import List, Dict, Set, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

from filing_analyzer import SECFilingAnalyzer
from hiring_queries import QUERY_CATEGORIES
from sec_client import SECClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SP500BulkDownloader:
    """Handles bulk downloading of S&P 500 companies and their 10-K filings"""
    
    def __init__(self, cache_dir: str = './filing_cache', openai_api_key: str = None):
        self.cache_dir = cache_dir
        self.analyzer = SECFilingAnalyzer(openai_api_key=openai_api_key, cache_dir=cache_dir)
        self.sec_client = SECClient()
        
        # Create cache directory if it doesn't exist
        Path(cache_dir).mkdir(exist_ok=True)
        
        # File to store S&P 500 company list
        self.sp500_cache_file = os.path.join(cache_dir, 'sp500_companies.json')
        
    def get_sp500_companies(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get current S&P 500 companies list from Wikipedia
        
        Args:
            force_refresh: Force refresh of cached data
            
        Returns:
            List of company dictionaries with symbol, name, sector info
        """
        # Check cache first
        if not force_refresh and os.path.exists(self.sp500_cache_file):
            try:
                with open(self.sp500_cache_file, 'r') as f:
                    cached_data = json.load(f)
                    # Check if cache is less than 7 days old
                    cache_age = datetime.now() - datetime.fromisoformat(cached_data['timestamp'])
                    if cache_age.days < 7:
                        logger.info(f"Using cached S&P 500 data ({len(cached_data['companies'])} companies)")
                        return cached_data['companies']
            except Exception as e:
                logger.warning(f"Error reading cached S&P 500 data: {e}")
        
        logger.info("Fetching current S&P 500 companies from Wikipedia...")
        
        try:
            # Scrape S&P 500 list from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {
                'User-Agent': 'SECFilingAnalyzer/1.0 (research@example.com)'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse with pandas
            tables = pd.read_html(response.content)
            sp500_table = tables[0]  # First table contains the S&P 500 companies
            
            companies = []
            for _, row in sp500_table.iterrows():
                try:
                    # Helper function to safely convert to string
                    def safe_str(value):
                        if pd.isna(value):
                            return None
                        return str(value).strip()
                    
                    company = {
                        'symbol': safe_str(row['Symbol']),
                        'name': safe_str(row['Security']),
                        'sector': safe_str(row['GICS Sector']),
                        'sub_industry': safe_str(row['GICS Sub-Industry']),
                        'headquarters': safe_str(row['Headquarters Location']),
                        'date_added': safe_str(row.get('Date Added')),
                        'cik': safe_str(row.get('CIK'))
                    }
                    
                    # Only add if we have at least symbol and name
                    if company['symbol'] and company['name']:
                        companies.append(company)
                        
                except Exception as e:
                    logger.warning(f"Error parsing company row: {e}")
                    continue
            
            # Cache the results
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'companies': companies
            }
            with open(self.sp500_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Successfully fetched {len(companies)} S&P 500 companies")
            return companies
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500 companies: {e}")
            exit(1)
    
    def get_filings_in_date_range(self, cik: str, start_year: int, end_year: int) -> List[Dict]:
        """
        Get 10-K filings for a company within a specific date range
        
        Args:
            cik: Company CIK
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            
        Returns:
            List of filing information dictionaries within date range
        """
        try:
            # Get all recent filings (we'll get more than needed to cover the range)
            all_filings = self.sec_client.get_10k_filings(cik, count=15)  # Get more to cover date range
            
            # Filter filings by date range
            filtered_filings = []
            for filing in all_filings:
                try:
                    filing_date = datetime.strptime(filing['filingDate'], '%Y-%m-%d')
                    filing_year = filing_date.year
                    
                    if start_year <= filing_year <= end_year:
                        filtered_filings.append(filing)
                except ValueError as e:
                    logger.warning(f"Error parsing filing date {filing.get('filingDate')}: {e}")
                    continue
            
            return filtered_filings
            
        except Exception as e:
            logger.error(f"Error getting filings for CIK {cik}: {e}")
            return []
    
    def process_companies_bulk(self, companies: List[Dict], start_year: int, end_year: int, 
                              dry_run: bool = False, max_companies: Optional[int] = None) -> Dict:
        """
        Process multiple companies in bulk
        
        Args:
            companies: List of company dictionaries
            start_year: Start year for filings
            end_year: End year for filings  
            dry_run: If True, only show what would be processed
            max_companies: Maximum number of companies to process (None = all)
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'total_companies': len(companies),
            'processed_companies': 0,
            'failed_companies': 0,
            'total_filings_processed': 0,
            'companies_with_data': [],
            'failed_company_details': [],
            'processing_errors': []
        }
        
        # Limit companies if specified
        if max_companies:
            companies = companies[:max_companies]
            stats['total_companies'] = len(companies)
        
        logger.info(f"{'DRY RUN: ' if dry_run else ''}Processing {len(companies)} companies for filings from {start_year} to {end_year}")
        
        for i, company in enumerate(companies, 1):
            ticker = company['symbol']
            company_name = company.get('name', ticker)
            
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}[{i}/{len(companies)}] Processing {ticker} ({company_name})")
            
            if dry_run:
                # In dry run, just show what we would do
                try:
                    cik = self.sec_client.get_company_cik(ticker)
                    potential_filings = self.get_filings_in_date_range(cik, start_year, end_year)
                    logger.info(f"  Would process {len(potential_filings)} filings from {start_year}-{end_year}")
                    stats['total_filings_processed'] += len(potential_filings)
                    stats['processed_companies'] += 1
                except Exception as e:
                    logger.warning(f"  Error getting info for {ticker}: {e}")
                    stats['failed_companies'] += 1
                continue
            
            try:
                # Get CIK and check for filings in date range
                cik = self.sec_client.get_company_cik(ticker)
                filings_in_range = self.get_filings_in_date_range(cik, start_year, end_year)
                
                if not filings_in_range:
                    logger.info(f"  No 10-K filings found for {ticker} in {start_year}-{end_year}")
                    continue
                
                logger.info(f"  Found {len(filings_in_range)} filings in date range")
                
                # Check which filings are already processed
                already_processed = self.analyzer.vector_store.get_processed_filings().get(cik, set())
                new_filings = [f for f in filings_in_range if f['accessionNumber'] not in already_processed]
                
                if not new_filings:
                    logger.info(f"  All filings for {ticker} already processed")
                    stats['processed_companies'] += 1
                    stats['companies_with_data'].append({
                        'ticker': ticker,
                        'name': company_name,
                        'sector': company.get('sector'),
                        'filings_count': len(filings_in_range),
                        'new_filings': 0
                    })
                    continue
                
                logger.info(f"  Processing {len(new_filings)} new filings for {ticker}")
                
                # Process each filing
                filings_processed = 0
                for filing in new_filings:
                    try:
                        # Download and process filing
                        content_dict = self.sec_client.download_filing_content(filing)
                        
                        if not content_dict.get('full_text'):
                            logger.warning(f"    No content downloaded for filing {filing['accessionNumber']}")
                            continue
                        
                        # Extract sections using text processor
                        if content_dict['full_text']:
                            content_dict['sections'] = self.analyzer.text_processor.extract_sections(content_dict['full_text'])
                        
                        chunk_dicts = self.analyzer.text_processor.intelligent_chunking(content_dict, filing)
                        
                        if chunk_dicts:
                            self.analyzer.vector_store.store_chunks(chunk_dicts, {cik: ticker})
                            filings_processed += 1
                            logger.info(f"    Processed filing {filing['filingDate']} with {len(chunk_dicts)} chunks")
                        
                        # Rate limiting
                        time.sleep(0.2)
                        
                    except Exception as e:
                        logger.error(f"    Error processing filing {filing['accessionNumber']}: {e}")
                        continue
                
                if filings_processed > 0:
                    stats['processed_companies'] += 1
                    stats['total_filings_processed'] += filings_processed
                    stats['companies_with_data'].append({
                        'ticker': ticker,
                        'name': company_name,
                        'sector': company.get('sector'),
                        'filings_count': len(filings_in_range),
                        'new_filings': filings_processed
                    })
                    
                    logger.info(f"  Successfully processed {filings_processed} filings for {ticker}")
                else:
                    logger.warning(f"  No filings successfully processed for {ticker}")
                    stats['failed_companies'] += 1
                
                # Delay between companies to be respectful to SEC servers
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"  Error processing {ticker}: {e}")
                stats['failed_companies'] += 1
                stats['failed_company_details'].append({
                    'ticker': ticker,
                    'name': company_name,
                    'error': str(e)
                })
                continue
        
        return stats
    
    def run_comprehensive_analysis(self) -> None:
        """Run comprehensive hiring analysis on all processed data"""
        logger.info("Running comprehensive hiring analysis...")
        
        # Get all available companies
        available_companies = self.analyzer.get_available_companies()
        if not available_companies:
            logger.warning("No companies available for analysis")
            return
        
        logger.info(f"Analyzing hiring patterns across {len(available_companies)} companies")
        
        # Run comprehensive analysis
        analysis = self.analyzer.comprehensive_hiring_analysis()
        
        # Display results
        print("\n" + "="*80)
        print("COMPREHENSIVE S&P 500 HIRING ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\nTotal Companies Analyzed: {len(available_companies)}")
        print(f"Total Relevant Segments Found: {analysis.get('total_results', 0)}")
        print(f"Average Similarity Score: {analysis.get('avg_similarity', 0):.3f}")
        
        # Show top findings
        if analysis.get('top_findings'):
            print(f"\nTOP HIRING CHALLENGES ACROSS S&P 500:")
            print("-" * 50)
            for i, finding in enumerate(analysis['top_findings'][:10], 1):
                # Try multiple ways to get company name
                company_name = finding.get('company')  # From query_all_companies method
                if not company_name:
                    # Try metadata fields
                    cik = finding['metadata'].get('cik')
                    ticker = finding['metadata'].get('ticker')
                    if ticker:
                        company_name = ticker
                    elif cik and cik in self.analyzer.cik_to_company:
                        company_name = self.analyzer.cik_to_company[cik]
                    else:
                        company_name = f"CIK-{cik}" if cik else "Unknown"
                        
                print(f"{i}. {company_name} (Score: {finding['similarity_score']:.3f})")
                preview = finding['document'][:150] + "..." if len(finding['document']) > 150 else finding['document']
                print(f"   {preview}")
                print()
        
        # Show company-specific hiring difficulty scores if available
        if analysis.get('company_details'):
            print(f"\nHIRING DIFFICULTY SCORES (1-10 scale):")
            print("-" * 50)
            
            # Group companies by name, then sort by years
            companies_by_name = {}
            for company in analysis['company_details']:
                company_name = company['company_name']
                if company_name not in companies_by_name:
                    companies_by_name[company_name] = []
                companies_by_name[company_name].append(company)
            
            # Sort companies by their highest difficulty score
            sorted_company_names = sorted(companies_by_name.keys(), 
                                        key=lambda name: max(c.get('hiring_difficulty_score', 0) for c in companies_by_name[name]), 
                                        reverse=True)
            
            company_count = 0
            for company_name in sorted_company_names:
                if company_count >= 10:  # Limit to top 10 companies
                    break
                    
                company_entries = companies_by_name[company_name]
                # Sort entries by year (newest first)
                company_entries.sort(key=lambda x: x.get('filing_date', ''), reverse=True)
                
                print(f"\n{company_name}:")
                for company in company_entries:
                    difficulty_score = company.get('hiring_difficulty_score', 1)
                    difficulty_label = "LOW" if difficulty_score <= 3 else "MEDIUM" if difficulty_score <= 6 else "HIGH"
                    
                    # Extract year from filing_date for temporal tracking
                    filing_year = company.get('filing_date', 'Unknown')[:4] if company.get('filing_date') else 'Unknown'
                    
                    print(f"  {filing_year}: {difficulty_score}/10 ({difficulty_label}) - "
                          f"Segments: {len(company.get('relevant_segments', []))}, "
                          f"Avg Similarity: {company.get('avg_similarity', 0):.3f}")
                
                company_count += 1
            
            print(f"\nShowing top {min(company_count, 10)} companies with hiring-related content")
            print("Scale: 1-3 (LOW likelihood), 4-6 (MEDIUM likelihood), 7-10 (HIGH likelihood)")
            print("Factors: Number of hiring segments, similarity scores, section diversity, content severity")
            print("Note: Years sorted newest first to show recent trends")
        
        # Show sector analysis if available
        if analysis.get('section_distribution'):
            print("\nFINDINGS BY SEC FILING SECTION:")
            print("-" * 40)
            for section, count in analysis['section_distribution'].items():
                print(f"   {section}: {count} segments")
        
        # Run AI analysis if available
        if self.analyzer.ai_analyzer.is_available() and analysis.get('top_findings'):
            print(f"\n" + "="*50)
            print("AI-POWERED SUMMARY")
            print("="*50)
            
            ai_summary = self.analyzer.ai_analyzer.generate_overall_summary(
                analysis['top_findings'][:15], 
                self.analyzer.cik_to_company
            )
            
            if ai_summary:
                print(ai_summary)
            else:
                print("AI analysis not available")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="S&P 500 Bulk Filing Downloader and Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bulk_sp500_downloader.py                    # Download all S&P 500 from 2015-2025
  python bulk_sp500_downloader.py --start-year 2020   # Download from 2020-2025
  python bulk_sp500_downloader.py --dry-run          # Show what would be downloaded
  python bulk_sp500_downloader.py --max-companies 50  # Process only first 50 companies
        """
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        default=2015,
        help='Start year for downloading filings (default: 2015)'
    )
    
    parser.add_argument(
        '--end-year',
        type=int,
        default=2025,
        help='End year for downloading filings (default: 2025)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )
    
    parser.add_argument(
        '--max-companies',
        type=int,
        help='Maximum number of companies to process (for testing)'
    )
    
    parser.add_argument(
        '--cache-dir',
        default='./filing_cache',
        help='Directory for caching downloaded filings (default: ./filing_cache)'
    )
    
    parser.add_argument(
        '--refresh-sp500',
        action='store_true',
        help='Force refresh of S&P 500 company list'
    )
    
    parser.add_argument(
        '--analysis-only',
        action='store_true',
        help='Skip downloading, only run analysis on existing data'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Get OpenAI API key for AI analysis
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not set. AI analysis will be disabled.")
    
    # Initialize downloader
    downloader = SP500BulkDownloader(
        cache_dir=args.cache_dir,
        openai_api_key=openai_api_key
    )
    
    # Show current database status
    print("="*80)
    print("S&P 500 BULK FILING DOWNLOADER AND ANALYZER")
    print("="*80)
    
    current_stats = downloader.analyzer.get_company_stats()
    print(f"Current Database Status: {current_stats}")
    
    if args.analysis_only:
        print("\nAnalysis-only mode: Skipping download, running analysis on existing data...")
        downloader.run_comprehensive_analysis()
        return
    
    # Get S&P 500 companies
    companies = downloader.get_sp500_companies(force_refresh=args.refresh_sp500)
    
    if not companies:
        logger.error("Could not get S&P 500 companies list")
        return
    
    # Process companies
    stats = downloader.process_companies_bulk(
        companies=companies,
        start_year=args.start_year,
        end_year=args.end_year,
        dry_run=args.dry_run,
        max_companies=args.max_companies
    )
    
    # Show processing statistics
    print("\n" + "="*80)
    print("PROCESSING STATISTICS")
    print("="*80)
    print(f"Total Companies: {stats['total_companies']}")
    print(f"Successfully Processed: {stats['processed_companies']}")
    print(f"Failed: {stats['failed_companies']}")
    print(f"Total Filings Processed: {stats['total_filings_processed']}")
    
    if stats['companies_with_data']:
        print(f"\nCompanies with Data ({len(stats['companies_with_data'])}):")
        for company in stats['companies_with_data'][:20]:  # Show first 20
            print(f"  {company['ticker']}: {company['new_filings']} new filings "
                  f"({company['filings_count']} total in range)")
        
        if len(stats['companies_with_data']) > 20:
            print(f"  ... and {len(stats['companies_with_data']) - 20} more")
    
    if stats['failed_company_details']:
        print(f"\nFailed Companies ({len(stats['failed_company_details'])}):")
        for failure in stats['failed_company_details'][:10]:  # Show first 10 failures
            print(f"  {failure['ticker']}: {failure['error']}")
    
    if not args.dry_run and stats['total_filings_processed'] > 0:
        # Persist database
        downloader.analyzer.persist_database()
        
        # Run comprehensive analysis
        downloader.run_comprehensive_analysis()
    
    print("\nBulk download and analysis completed!")


if __name__ == "__main__":
    main() 