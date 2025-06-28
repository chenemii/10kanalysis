import os
import sys
import time
import logging
import argparse
from typing import List

from filing_analyzer import SECFilingAnalyzer
from hiring_queries import COMPARISON_QUERIES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="SEC Filing Analyzer - Analyze 10-K filings for hiring and workforce challenges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL MSFT                    # Analyze Apple and Microsoft
  python main.py --companies AAPL GOOGL AMZN  # Analyze multiple companies
  python main.py TSLA --filings 3             # Analyze Tesla with 3 recent filings
  python main.py --all                        # Analyze default set of companies
  python main.py --query-company AAPL         # Query only Apple's data
  python main.py --compare AAPL MSFT          # Compare Apple and Microsoft
  python main.py --help                       # Show this help message
        """
    )
    
    # Main argument: company tickers
    parser.add_argument(
        'tickers', 
        nargs='*', 
        help='Stock ticker symbols to analyze (e.g., AAPL MSFT GOOGL)'
    )
    
    # Alternative way to specify companies
    parser.add_argument(
        '--companies', '-c',
        nargs='+',
        help='Stock ticker symbols to analyze (alternative to positional args)'
    )
    
    # Number of filings to process per company
    parser.add_argument(
        '--filings', '-f',
        type=int,
        default=2,
        help='Number of recent 10-K filings to process per company (default: 2)'
    )
    
    # Similarity threshold
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.10,
        help='Minimum similarity score threshold for relevant results (default: 0.10)'
    )
    
    # Use default company set
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Analyze default set of companies (AAPL, MSFT, GOOGL, AMZN, TSLA)'
    )
    
    # Skip analysis, just show status
    parser.add_argument(
        '--status-only', '-s',
        action='store_true',
        help='Show database status and exit without processing'
    )
    
    # Query specific company
    parser.add_argument(
        '--query-company', '-q',
        help='Query hiring data for a specific company only'
    )
    
    # Compare companies
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Compare hiring challenges between specific companies'
    )
    
    # List available companies
    parser.add_argument(
        '--list-companies', '-l',
        action='store_true',
        help='List all companies available in the database'
    )
    
    # Enable/disable generic filtering
    parser.add_argument(
        '--filter-generic',
        action='store_true',
        default=True,
        help='Filter out generic boilerplate content (default: enabled)'
    )
    
    parser.add_argument(
        '--no-filter-generic',
        action='store_true',
        help='Disable generic content filtering (show all results)'
    )
    
    # Cache directory
    parser.add_argument(
        '--cache-dir',
        default='./filing_cache',
        help='Directory for caching downloaded filings (default: ./filing_cache)'
    )
    
    return parser.parse_args()


def handle_query_company(analyzer: SECFilingAnalyzer, query_ticker: str, args) -> None:
    """Handle querying a specific company"""
    available_companies = analyzer.get_available_companies()
    
    if query_ticker not in available_companies:
        print(f"\nCompany {query_ticker} not found in database.")
        print(f"Available companies: {', '.join(available_companies)}")
        print(f"\nAutomatically downloading and processing {query_ticker} filings...")
        
        try:
            # Process the company with default 2 filings
            analyzer.process_company_filings(query_ticker, filing_count=args.filings)
            
            # Persist the new data
            analyzer.persist_database()
            
            # Update available companies list
            available_companies = analyzer.get_available_companies()
            
            if query_ticker not in available_companies:
                print(f"Error: Failed to process {query_ticker}. Company may not exist or filings may not be available.")
                return
            else:
                print(f"Successfully added {query_ticker} to database!")
                
        except Exception as e:
            print(f"Error processing {query_ticker}: {str(e)}")
            return
    
    print(f"\nQuerying hiring data for {query_ticker}...")
    
    # Query specific company with detailed analysis
    from hiring_queries import QUERY_CATEGORIES
    total_meaningful_results = 0
    all_company_results = []
    
    # Use both general and specific queries
    all_queries = QUERY_CATEGORIES['all']
    
    for query in all_queries:
        results = analyzer.query_company_specific(query, query_ticker, n_results=10)
        # Filter by similarity score AND generic content
        meaningful_results = []
        for r in results:
            if (r['similarity_score'] >= analyzer.min_similarity_score and 
                not analyzer.text_processor.is_generic_content(r['document'])):
                meaningful_results.append(r)
            elif analyzer.text_processor.is_generic_content(r['document']):
                print(f"  Filtered generic content (score: {r['similarity_score']:.3f}): {r['document'][:80]}...")
        
        total_meaningful_results += len(meaningful_results)
        all_company_results.extend(meaningful_results)
        
        if meaningful_results:
            top_score = max(meaningful_results, key=lambda x: x['similarity_score'])['similarity_score']
            print(f"'{query}': {len(meaningful_results)} specific segments (top score: {top_score:.3f})")
    
    print(f"\nTotal specific, non-generic segments for {query_ticker}: {total_meaningful_results}")
    
    # Generate detailed analysis if we have results
    if all_company_results and analyzer.ai_analyzer.is_available():
        print(f"\n" + "="*50)
        print(f"DETAILED ANALYSIS FOR {query_ticker}")
        print("="*50)
        
        # Deduplicate results by chunk ID
        seen_ids = set()
        unique_results = []
        for result in all_company_results:
            chunk_id = f"{result['metadata']['cik']}_{result['metadata']['accession_number']}_{result['metadata']['chunk_index']}"
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_results.append(result)
        
        # Sort by similarity score
        unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Create company data structure for summary generation
        company_data = {
            'cik': analyzer.ticker_to_cik(query_ticker),
            'company_name': query_ticker,
            'relevant_segments': [
                {
                    'text': result['document'],
                    'section': result['metadata'].get('section', 'unknown'),
                    'similarity_score': result['similarity_score'],
                    'priority': result['metadata'].get('priority', 'unknown')
                }
                for result in unique_results[:10]  # Top 10 segments
            ]
        }
        
        # Calculate additional fields needed for scoring
        if company_data['relevant_segments']:
            # Calculate average similarity score
            avg_sim = sum(seg['similarity_score'] for seg in company_data['relevant_segments']) / len(company_data['relevant_segments'])
            company_data['avg_similarity'] = avg_sim
            
            # Get unique sections found
            sections_found = set(seg['section'] for seg in company_data['relevant_segments'])
            company_data['sections_found'] = list(sections_found)
        else:
            company_data['avg_similarity'] = 0
            company_data['sections_found'] = []
        
        # Generate company-specific summary
        if len(company_data['relevant_segments']) >= 1:
            company_summary = analyzer.ai_analyzer.generate_company_summary(company_data)
            if company_summary:
                print(f"AI Summary for {query_ticker}:")
                print(f"{company_summary}")
                
                # Calculate and display hiring difficulty score
                difficulty_score = analyzer.ai_analyzer.calculate_hiring_difficulty_score(company_data)
                difficulty_label = "LOW" if difficulty_score <= 3 else "MEDIUM" if difficulty_score <= 6 else "HIGH"
                print(f"\nHiring Difficulty Score: {difficulty_score}/10 ({difficulty_label})")
                print("Scale: 1-3 (LOW likelihood), 4-6 (MEDIUM likelihood), 7-10 (HIGH likelihood)")
                
                print(f"\n" + "-"*50)
                print("SOURCE TEXT VERIFICATION:")
                print("-"*50)
                print("The AI summary above was based on the following actual text from SEC filings:")
                for i, segment in enumerate(company_data['relevant_segments'][:3], 1):
                    print(f"\nSource {i} (Score: {segment['similarity_score']:.3f}, Section: {segment['section']}):")
                    print(f'"{segment["text"]}"')
                    print()
        
        # Show top segments
        print(f"\nTop Relevant Segments:")
        for i, segment in enumerate(company_data['relevant_segments'][:3], 1):
            preview = segment['text'][:300] + "..." if len(segment['text']) > 300 else segment['text']
            print(f"\n{i}. Section: {segment['section']} (Score: {segment['similarity_score']:.3f})")
            print(f"   {preview}")
    
    elif all_company_results:
        print(f"\nFound {len(all_company_results)} relevant segments but no OpenAI API key available for detailed analysis.")
    else:
        print(f"\nNo relevant hiring content found for {query_ticker} above similarity threshold {analyzer.min_similarity_score}")


def handle_company_comparison(analyzer: SECFilingAnalyzer, compare_companies: List[str], args) -> None:
    """Handle company comparison analysis"""
    compare_companies = [ticker.upper() for ticker in compare_companies]
    available_companies = analyzer.get_available_companies()
    
    # Check if all companies exist, auto-download missing ones
    missing_companies = [c for c in compare_companies if c not in available_companies]
    if missing_companies:
        print(f"\nMissing companies: {', '.join(missing_companies)}")
        print(f"Available companies: {', '.join(available_companies)}")
        print(f"\nAutomatically downloading and processing missing companies...")
        
        for missing_ticker in missing_companies:
            try:
                print(f"\nProcessing {missing_ticker}...")
                analyzer.process_company_filings(missing_ticker, filing_count=args.filings)
            except Exception as e:
                print(f"Error processing {missing_ticker}: {str(e)}")
                print(f"Skipping {missing_ticker} from comparison.")
                compare_companies.remove(missing_ticker)
        
        # Persist all new data
        analyzer.persist_database()
        
        # Update available companies list
        available_companies = analyzer.get_available_companies()
        
        # Final check for any still missing companies
        still_missing = [c for c in compare_companies if c not in available_companies]
        if still_missing:
            print(f"\nWarning: Could not process these companies: {', '.join(still_missing)}")
            compare_companies = [c for c in compare_companies if c in available_companies]
            
        if not compare_companies:
            print("\nNo valid companies available for comparison.")
            return
            
        print(f"\nSuccessfully processed missing companies!")
    
    print(f"\nComparing hiring challenges: {', '.join(compare_companies)}")
    
    # Compare companies on key hiring topics
    for query in COMPARISON_QUERIES:
        print(f"\n--- {query.title()} ---")
        comparison = analyzer.compare_companies(query, compare_companies, n_results=3)
        
        for company, data in comparison.items():
            if data['count'] > 0:
                print(f"{company}: {data['count']} segments (avg sim: {data['avg_similarity']:.3f}, top: {data['top_similarity']:.3f})")
            else:
                print(f"{company}: 0 segments above threshold {data['threshold_used']:.2f} (searched {data['total_searched']}, best: {data['top_similarity']:.3f})")


def show_status(analyzer: SECFilingAnalyzer) -> None:
    """Show database status and statistics"""
    stats = analyzer.get_company_stats()
    processed_filings = analyzer.vector_store.get_processed_filings()
    cache_stats = analyzer.get_cache_stats()
    
    print("="*60)
    print("SEC FILING ANALYZER - DATABASE STATUS")
    print("="*60)
    
    if stats.get('total_companies', 0) > 0:
        print(f"Total Companies in Database: {stats['total_companies']}")
        print(f"Total Chunks Across All Companies: {stats.get('total_chunks_all_companies', 0)}")
        print(f"Cache Directory: {cache_stats['cache_directory']}")
        
        # Show per-company statistics
        print("\nCompany-Specific Statistics:")
        for company, company_stats in stats.get('companies', {}).items():
            if 'chunks' in company_stats:
                print(f"  {company}: {company_stats['chunks']} chunks")
    else:
        print("No companies found in database")
        print(f"Cache Directory: {cache_stats['cache_directory']}")
    
    if processed_filings:
        print("\nAlready Processed Filings:")
        for cik, accessions in processed_filings.items():
            company_name = analyzer.cik_to_company.get(cik, f"CIK-{cik}")
            print(f"  {company_name} (CIK {cik}): {len(accessions)} filings")
    else:
        print("No previously processed filings found.")
    
    print(f"\nCache Statistics:")
    print(f"  Cached Files: {cache_stats['total_files']} filing cache files")
    print(f"  Cache Size: {cache_stats['total_size_mb']} MB")
    print("="*60)


def run_comprehensive_analysis(analyzer: SECFilingAnalyzer) -> None:
    """Run comprehensive hiring analysis"""
    print("\n" + "="*50)
    print("COMPREHENSIVE HIRING TREND ANALYSIS")
    print("="*50)
    
    analysis = analyzer.comprehensive_hiring_analysis()
    
    if 'message' in analysis:
        print(f"Result: {analysis['message']}")
        print(f"Similarity threshold used: {analyzer.min_similarity_score}")
        return
    
    print(f"Total relevant segments found: {analysis.get('total_relevant_segments', 0)}")
    print(f"Companies with hiring-related content: {analysis.get('companies_analyzed', 0)}")
    print(f"Similarity threshold: {analysis.get('similarity_threshold', 'N/A')}")
    
    # Display overall AI summary if available
    if analysis.get('ai_summary'):
        print("\n" + "="*50)
        print("OVERALL HIRING TRENDS SUMMARY")
        print("="*50)
        print(analysis['ai_summary'])
    
    # Display company-specific findings
    if 'company_details' in analysis and analysis['company_details']:
        print("\n" + "="*50)
        print("COMPANY-SPECIFIC HIRING CHALLENGES")
        print("="*50)
        
        # Group companies by name for better temporal analysis
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
        
        shown_companies = 0
        for company_name in sorted_company_names:
            if shown_companies >= 5:  # Limit to top 5 companies for detailed view
                break
                
            company_entries = companies_by_name[company_name]
            # Sort entries by year (newest first)
            company_entries.sort(key=lambda x: x.get('filing_date', ''), reverse=True)
            
            print(f"\n{shown_companies + 1}. {company_name} - Hiring Trends Over Time:")
            
            for j, company in enumerate(company_entries):
                # Extract year from filing_date for temporal tracking
                filing_year = company.get('filing_date', 'Unknown')[:4] if company.get('filing_date') else 'Unknown'
                
                print(f"\n   {filing_year} Filing (CIK: {company['cik']}):")
                print(f"   Filing Date: {company['filing_date']}")
                print(f"   Average Similarity Score: {company['avg_similarity']:.3f}")
                print(f"   Relevant Segments: {len(company['relevant_segments'])}")
                print(f"   Sections Found: {', '.join(company['sections_found'])}")
                
                # Display hiring difficulty score
                difficulty_score = company.get('hiring_difficulty_score', 1)
                difficulty_label = "LOW" if difficulty_score <= 3 else "MEDIUM" if difficulty_score <= 6 else "HIGH"
                print(f"   Hiring Difficulty Score: {difficulty_score}/10 ({difficulty_label})")
                
                # Show AI-generated company summary if available (only for most recent year)
                if j == 0 and company.get('ai_summary'):
                    print(f"\n   Latest AI Summary ({filing_year}):")
                    print(f"   {company['ai_summary']}")
                elif j == 0 and company['relevant_segments']:
                    # Show top segment as fallback for most recent year
                    top_segment = company['relevant_segments'][0]
                    preview = top_segment['text'][:200] + "..." if len(top_segment['text']) > 200 else top_segment['text']
                    print(f"\n   Latest Top Segment (Score: {top_segment['similarity_score']:.3f}):")
                    print(f"   {preview}")
            
            shown_companies += 1
    
    # Show section distribution
    if analysis.get('section_distribution'):
        print(f"\n" + "="*50)
        print("FINDINGS BY SEC FILING SECTION")
        print("="*50)
        for section, count in analysis['section_distribution'].items():
            print(f"   {section}: {count} segments")


def main():
    """Main execution function"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Determine which companies to analyze
    if args.all:
        companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    elif args.companies:
        companies = [ticker.upper() for ticker in args.companies]
    elif args.tickers:
        companies = [ticker.upper() for ticker in args.tickers]
    else:
        # No companies specified for processing, but might be query-only
        companies = []
    
    # Initialize analyzer
    openai_api_key = os.getenv("OPENAI_API_KEY")
    analyzer = SECFilingAnalyzer(
        openai_api_key=openai_api_key, 
        cache_dir=args.cache_dir
    )
    
    # Override similarity threshold if specified
    if args.threshold != 0.10:
        analyzer.min_similarity_score = args.threshold
        print(f"Using custom similarity threshold: {args.threshold}")
    
    # Check for and perform migration if needed
    migration_performed = analyzer.migrate_old_collection()
    if migration_performed:
        print("Database migration completed successfully.\n")
    
    # Initialize CIK to ticker mappings for existing collections
    analyzer.initialize_company_mappings()
    
    # Show status
    show_status(analyzer)
    
    # Handle list companies option
    if args.list_companies:
        available_companies = analyzer.get_available_companies()
        if available_companies:
            print(f"\nAvailable Companies: {', '.join(available_companies)}")
        else:
            print("\nNo companies available in database")
        return
    
    # Handle query-specific company option
    if args.query_company:
        handle_query_company(analyzer, args.query_company.upper(), args)
        return
    
    # Handle company comparison option
    if args.compare:
        handle_company_comparison(analyzer, args.compare, args)
        return
    
    # Exit early if only status requested
    if args.status_only:
        print("\nStatus-only mode: Exiting without processing.")
        return
    
    # If no companies specified, show help
    if not companies:
        print("No companies specified for processing. Use --help for usage information.")
        print("Use --query-company TICKER to query existing data or --list-companies to see available data.")
        return
    
    print(f"\nCompanies to analyze: {', '.join(companies)}")
    print(f"Filings per company: {args.filings}")
    print(f"Similarity threshold: {analyzer.min_similarity_score}")
    print("="*60)
    
    # Process filings for each company
    for ticker in companies:
        print(f"\nProcessing {ticker}...")
        try:
            analyzer.process_company_filings(ticker, filing_count=args.filings)
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
        
        # Add delay between companies to be respectful to SEC servers
        time.sleep(1)
    
    # Persist database after all processing
    analyzer.persist_database()
    
    # Show final collection statistics
    final_stats = analyzer.get_company_stats()
    print(f"\nFinal Database Stats: {final_stats}")
    
    # Run comprehensive analysis
    run_comprehensive_analysis(analyzer)


if __name__ == "__main__":
    main() 