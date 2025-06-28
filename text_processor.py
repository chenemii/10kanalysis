import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text processing, chunking, and section extraction"""
    
    def __init__(self):
        # 10-K section patterns for intelligent parsing
        self.section_patterns = {
            'risk_factors': [
                r'item\s*1a\.?\s*risk\s*factors',
                r'risk\s*factors',
                r'principal\s*risks'
            ],
            'business': [
                r'item\s*1\.?\s*business',
                r'our\s*business',
                r'business\s*overview'
            ],
            'md_a': [
                r'item\s*7\.?\s*management[\'s]*\s*discussion',
                r'management[\'s]*\s*discussion\s*and\s*analysis',
                r'md&a'
            ],
            'human_capital': [
                r'human\s*capital',
                r'our\s*employees',
                r'workforce',
                r'personnel'
            ]
        }
        
        # Generic phrases to filter out (too common to be meaningful)
        self.generic_phrases = [
            "our business is based on successfully attracting"
        ]
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def extract_sections(self, content: str) -> Dict[str, str]:
        """
        Extract specific sections from 10-K filing content using regex patterns
        
        Args:
            content: Full filing text
            
        Returns:
            Dictionary of extracted sections
        """
        sections = {}
        content_lower = content.lower()
        
        for section_name, patterns in self.section_patterns.items():
            section_text = ""
            
            for pattern in patterns:
                # Find section start
                matches = list(re.finditer(pattern, content_lower, re.IGNORECASE | re.MULTILINE))
                
                if matches:
                    start_pos = matches[0].start()
                    
                    # Find section end (next major section or end of document)
                    end_patterns = [
                        r'\bitem\s*\d+[a-z]*\.?\s*[a-z]',  # Next item
                        r'\bpart\s*[iv]+',  # Next part
                        r'\bsignatures?\b',  # Signatures section
                        r'\bexhibits?\b'  # Exhibits section
                    ]
                    
                    end_pos = len(content)
                    for end_pattern in end_patterns:
                        end_matches = list(re.finditer(end_pattern, content_lower[start_pos + 100:], re.IGNORECASE))
                        if end_matches:
                            potential_end = start_pos + 100 + end_matches[0].start()
                            if potential_end > start_pos:
                                end_pos = min(end_pos, potential_end)
                    
                    # Extract section text
                    section_text = content[start_pos:end_pos].strip()
                    
                    # Clean up section text
                    section_text = re.sub(r'\s+', ' ', section_text)
                    section_text = section_text[:10000]  # Limit section size
                    
                    break
            
            if section_text:
                sections[section_name] = section_text
        
        return sections
    
    def intelligent_chunking(self, content_dict: Dict[str, str], filing_info: Dict) -> List[Dict]:
        """
        Create intelligent chunks prioritizing relevant sections
        
        Args:
            content_dict: Dictionary with full_text and sections
            filing_info: Filing metadata
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        chunk_id = 0
        
        # Priority order for sections (most likely to contain hiring info)
        section_priority = ['risk_factors', 'md_a', 'business', 'human_capital']
        
        # Process high-priority sections first with smaller chunks
        for section_name in section_priority:
            if section_name in content_dict['sections']:
                section_text = content_dict['sections'][section_name]
                section_chunks = self.chunk_text(section_text, chunk_size=800, overlap=150)
                
                for chunk_text in section_chunks:
                    chunks.append({
                        'text': chunk_text,
                        'section': section_name,
                        'priority': 'high',
                        'chunk_id': chunk_id,
                        'filing_info': filing_info
                    })
                    chunk_id += 1
        
        # Process remaining full text with larger chunks for general context
        if content_dict['full_text']:
            # Skip already processed sections to avoid duplication
            remaining_text = content_dict['full_text']
            for section_text in content_dict['sections'].values():
                remaining_text = remaining_text.replace(section_text[:500], '', 1)  # Remove section starts
            
            general_chunks = self.chunk_text(remaining_text, chunk_size=1200, overlap=200)
            
            for chunk_text in general_chunks:
                chunks.append({
                    'text': chunk_text,
                    'section': 'general',
                    'priority': 'medium',
                    'chunk_id': chunk_id,
                    'filing_info': filing_info
                })
                chunk_id += 1
        
        return chunks
    
    def is_generic_content(self, text: str) -> bool:
        """
        Check if content is too generic to be useful
        
        Args:
            text: Text to check
            
        Returns:
            True if content appears to be generic boilerplate
        """
        text_lower = text.lower()
        
        # Count how many generic phrases appear
        generic_count = sum(1 for phrase in self.generic_phrases if phrase in text_lower)
        
        # If multiple generic phrases appear, likely boilerplate
        if generic_count >= 2:
            return True
        
        # Check for very short content (likely incomplete)
        if len(text.strip()) < 100:
            return True
        
        # Check for overly repetitive content
        words = text_lower.split()
        if len(set(words)) / len(words) < 0.3:  # Less than 30% unique words
            return True
        
        return False 