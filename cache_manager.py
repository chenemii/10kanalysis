import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class CacheManager:
    """Handles file caching for downloaded SEC filings"""
    
    def __init__(self, cache_dir: str = "./filing_cache"):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to store cached files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_file_path(self, cik: str, accession_number: str) -> Path:
        """
        Get the cache file path for a specific filing
        
        Args:
            cik: Company CIK
            accession_number: SEC accession number
            
        Returns:
            Path to cache file
        """
        filename = f"{cik}_{accession_number.replace('-', '')}.json"
        return self.cache_dir / filename
    
    def is_filing_cached(self, cik: str, accession_number: str) -> bool:
        """
        Check if a filing is already cached
        
        Args:
            cik: Company CIK
            accession_number: SEC accession number
            
        Returns:
            True if filing is cached
        """
        cache_file = self._get_cache_file_path(cik, accession_number)
        return cache_file.exists()
    
    def save_filing_to_cache(self, filing_info: Dict, content_dict: Dict) -> None:
        """
        Save filing content to cache
        
        Args:
            filing_info: Filing metadata
            content_dict: Filing content
        """
        cache_file = self._get_cache_file_path(filing_info['cik'], filing_info['accessionNumber'])
        
        cache_data = {
            'filing_info': filing_info,
            'content': content_dict,
            'cached_at': datetime.now().isoformat()
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Cached filing {filing_info['accessionNumber']} to {cache_file}")
        except Exception as e:
            logger.error(f"Error saving filing to cache: {e}")
    
    def load_filing_from_cache(self, cik: str, accession_number: str) -> Dict:
        """
        Load filing content from cache
        
        Args:
            cik: Company CIK
            accession_number: SEC accession number
            
        Returns:
            Filing content dictionary
        """
        cache_file = self._get_cache_file_path(cik, accession_number)
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            logger.info(f"Loaded filing {accession_number} from cache")
            return cache_data['content']
        except Exception as e:
            logger.error(f"Error loading filing from cache: {e}")
            return {}
    
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about cached files
        
        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob('*.json'))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'total_files': len(cache_files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_directory': str(self.cache_dir)
        }
    
    def clear_cache(self) -> int:
        """
        Clear all cached files
        
        Returns:
            Number of files deleted
        """
        cache_files = list(self.cache_dir.glob('*.json'))
        deleted_count = 0
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting cache file {cache_file}: {e}")
        
        logger.info(f"Deleted {deleted_count} cache files")
        return deleted_count 