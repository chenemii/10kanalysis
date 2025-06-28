import os
import requests
import time
import logging
from typing import List, Dict
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


class SECClient:
    """Handles all SEC API interactions and filing downloads"""
    
    def __init__(self):
        # SEC EDGAR headers (required by SEC)
        self.headers = {
            'User-Agent': 'SECFilingAnalyzer/1.0 (research@example.com)',
            'Accept': 'application/json, text/html, */*',
            'Accept-Encoding': 'gzip, deflate',
        }
    
    def get_company_cik(self, ticker: str) -> str:
        """
        Get CIK (Central Index Key) for a company ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CIK string
        """
        url = f"https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            data = response.json()
            for key, company_info in data.items():
                if company_info['ticker'].upper() == ticker.upper():
                    cik = str(company_info['cik_str']).zfill(10)
                    return cik
        
        raise ValueError(f"Could not find CIK for ticker: {ticker}")
    
    def get_10k_filings(self, cik: str, count: int = 3) -> List[Dict]:
        """
        Get recent 10-K filings for a company
        
        Args:
            cik: Company CIK
            count: Number of recent filings to retrieve
            
        Returns:
            List of filing information dictionaries
        """
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        logger.info(f"Requesting URL: {url}")
        
        response = requests.get(url, headers=self.headers)
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Response content: {response.text[:500]}")
            raise Exception(f"Failed to get submissions data: {response.status_code}")
        
        data = response.json()
        filings = data['filings']['recent']
        
        # Filter for 10-K filings
        ten_k_filings = []
        for i, form in enumerate(filings['form']):
            if form == '10-K' and len(ten_k_filings) < count:
                filing_info = {
                    'accessionNumber': filings['accessionNumber'][i],
                    'filingDate': filings['filingDate'][i],
                    'reportDate': filings['reportDate'][i],
                    'form': form,
                    'cik': cik
                }
                ten_k_filings.append(filing_info)
        
        return ten_k_filings
    
    def download_filing_content(self, filing_info: Dict) -> Dict[str, str]:
        """
        Download and extract structured content from a 10-K filing
        
        Args:
            filing_info: Filing information dictionary
            
        Returns:
            Dictionary with full text and extracted sections
        """
        accession_no = filing_info['accessionNumber'].replace('-', '')
        cik = filing_info['cik']
        
        # Try multiple URL patterns for different filing formats
        urls_to_try = [
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no}/{filing_info['accessionNumber']}-index.html",
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no}/{filing_info['accessionNumber']}.txt",
            f"https://data.sec.gov/submissions/CIK{cik}.json"
        ]
        
        main_doc_url = None
        content = ""
        
        # First try to get the index page to find the main document
        for url in urls_to_try[:1]:  # Start with index
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for various document patterns
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    text = link.get_text().lower()
                    
                    # More comprehensive document detection
                    if any(pattern in href.lower() for pattern in ['10-k.htm', '10k.htm', '10-k.txt', '10k.txt']) or \
                       any(pattern in text for pattern in ['10-k', '10k', 'form 10-k']):
                        main_doc_url = f"https://www.sec.gov{href}" if href.startswith('/') else href
                        break
                
                if main_doc_url:
                    break
        
        # If no main doc found, try direct URLs
        if not main_doc_url:
            for url in urls_to_try[1:]:
                if url.endswith('.txt'):
                    main_doc_url = url
                    break
        
        if not main_doc_url:
            logger.warning(f"Could not find main document URL for filing {filing_info['accessionNumber']}")
            return {"full_text": "", "sections": {}}
        
        # Download the main document
        time.sleep(0.1)  # Rate limiting
        doc_response = requests.get(main_doc_url, headers=self.headers)
        
        if doc_response.status_code != 200:
            logger.warning(f"Could not download filing content: {main_doc_url}")
            return {"full_text": "", "sections": {}}
        
        # Handle different content types
        content_type = doc_response.headers.get('content-type', '').lower()
        raw_content = doc_response.text
        
        if 'html' in content_type or '<html' in raw_content.lower():
            # HTML format
            soup = BeautifulSoup(raw_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "head"]):
                element.decompose()
            
            # Get clean text
            content = soup.get_text()
        else:
            # Plain text or SGML format
            content = raw_content
            
            # Clean SGML tags if present
            content = re.sub(r'<[^>]+>', '', content)
        
        # Normalize whitespace
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return {
            "full_text": content,
            "sections": {}  # Will be filled by TextProcessor
        }
    
    def resolve_cik_to_ticker(self, cik: str) -> str:
        """
        Try to resolve a CIK to ticker symbol from SEC data
        
        Args:
            cik: CIK string
            
        Returns:
            Ticker symbol or CIK if not found
        """
        try:
            url = f"https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                for key, company_info in data.items():
                    if str(company_info['cik_str']).zfill(10) == cik:
                        ticker = company_info['ticker'].upper()
                        return ticker
        except Exception as e:
            logger.debug(f"Error resolving CIK {cik} to ticker: {e}")
        
        return cik  # Return CIK if ticker not found 