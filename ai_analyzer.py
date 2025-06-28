import logging
from typing import Dict, List, Optional
from openai import OpenAI
import tiktoken

logger = logging.getLogger(__name__)


class AIAnalyzer:
    """Handles OpenAI interactions for AI-powered analysis"""
    
    def __init__(self, openai_api_key: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize AI analyzer
        
        Args:
            openai_api_key: OpenAI API key (optional)
            model: OpenAI model to use
        """
        self.openai_api_key = openai_api_key
        self.model = model
        self.max_context_tokens = self._get_model_context_limit(model)
        self.max_completion_tokens = min(1000, self.max_context_tokens // 4)  # Reserve 75% for input
        
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except:
                self.encoding = tiktoken.get_encoding("cl100k_base")  # Fallback
        else:
            self.openai_client = None
            
    def _get_model_context_limit(self, model: str) -> int:
        """Get context window limit for specific model"""
        model_limits = {
            "gpt-3.5-turbo": 16385,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }
        return model_limits.get(model, 16385)  # Conservative default
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not hasattr(self, 'encoding'):
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
        try:
            return len(self.encoding.encode(text))
        except:
            return len(text) // 4
    
    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to stay within token limit"""
        current_tokens = self._count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        # Binary search to find optimal truncation point
        chars = len(text)
        ratio = max_tokens / current_tokens
        target_chars = int(chars * ratio * 0.9)  # 90% to be safe
        
        truncated = text[:target_chars]
        while self._count_tokens(truncated) > max_tokens and len(truncated) > 100:
            truncated = truncated[:int(len(truncated) * 0.9)]
        
        return truncated + "... [truncated]"
        
    def _smart_segment_selection(self, segments: List[str], max_tokens: int) -> str:
        """Intelligently select and combine segments within token limit"""
        if not segments:
            return ""
            
        # Sort by length (shorter segments first for diversity)
        segments_with_tokens = [(seg, self._count_tokens(seg)) for seg in segments]
        segments_with_tokens.sort(key=lambda x: x[1])
        
        selected_segments = []
        total_tokens = 0
        separator_tokens = self._count_tokens("\n\n---\n\n")
        
        for segment, tokens in segments_with_tokens:
            if total_tokens + tokens + separator_tokens > max_tokens:
                break
            selected_segments.append(segment)
            total_tokens += tokens + separator_tokens
            
        return "\n\n---\n\n".join(selected_segments)
    
    def is_available(self) -> bool:
        """Check if AI analysis is available (API key provided)"""
        return self.openai_client is not None
    
    def generate_company_summary(self, company_data: Dict) -> Optional[str]:
        """
        Generate AI summary for a specific company's hiring challenges
        
        Args:
            company_data: Dictionary containing company filing segments
            
        Returns:
            AI-generated summary or None if not available
        """
        if not self.openai_client:
            return None
        
        # Calculate available tokens for input (reserve tokens for system prompt + completion)
        system_prompt = f"You are analyzing SEC 10-K filing content for {company_data['company_name']}. Summarize ONLY the hiring and workforce challenges that are explicitly mentioned in the provided text. Do not make inferences or connections that are not directly stated. If the text discusses regulatory compliance, only mention it if it explicitly relates to hiring/workforce. Be specific and quote directly from the source text."
        
        system_tokens = self._count_tokens(system_prompt)
        available_tokens = self.max_context_tokens - system_tokens - self.max_completion_tokens - 200  # Buffer
        
        # Select top segments within token limit
        segments = [seg['text'] for seg in company_data['relevant_segments'][:10]]  # Top 10 segments
        combined_text = self._smart_segment_selection(segments, available_tokens)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize the hiring and workforce challenges for {company_data['company_name']} based on these SEC filing segments. ONLY use information explicitly stated in the text about hiring, employees, workforce, or talent. Quote directly from the text to support your analysis:\n\n{combined_text}"}
                ],
                max_tokens=self.max_completion_tokens,
                temperature=0.2
            )
            
            logger.info(f"Generated summary for {company_data['company_name']} using {self._count_tokens(combined_text)} input tokens")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating company summary for {company_data['company_name']}: {e}")
            return None
    
    def generate_overall_summary(self, top_segments: List[Dict], cik_to_company: Dict[str, str]) -> Optional[str]:
        """
        Generate overall AI summary from top segments across all companies
        
        Args:
            top_segments: List of top relevant segments
            cik_to_company: Mapping of CIK to company names
            
        Returns:
            AI-generated overall summary or None if not available
        """
        if not self.openai_client or not top_segments:
            return None
        
        system_prompt = "You are an expert financial analyst specializing in SEC filings. Analyze the following segments from 10-K filings for specific hiring difficulties, talent shortages, workforce challenges, and labor market issues. ONLY summarize what is explicitly stated in the text about hiring, employees, workforce, or talent. Do not make inferences about hiring challenges unless they are directly mentioned. Focus on concrete, company-specific challenges rather than generic statements. Provide a structured summary with key themes, trends, and business impacts, but only based on what is explicitly stated in the source text."
        
        system_tokens = self._count_tokens(system_prompt)
        available_tokens = self.max_context_tokens - system_tokens - self.max_completion_tokens - 200
        
        # Prepare segments with company names
        segments_with_companies = []
        for segment in top_segments[:20]:  # Consider top 20 segments
            company_name = cik_to_company.get(segment['metadata']['cik'], f"Company-{segment['metadata']['cik']}")
            segments_with_companies.append(f"[{company_name}]: {segment['document']}")
        
        # Smart selection within token limits
        combined_text = self._smart_segment_selection(segments_with_companies, available_tokens)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze these SEC filing segments for specific hiring and workforce challenges. Don't use any generic phrases. Don't make up any information. ONLY include information that explicitly mentions hiring, employees, workforce, talent, or labor. Quote directly from the text to support your analysis:\n\n{combined_text}"}
                ],
                max_tokens=self.max_completion_tokens,
                temperature=0.7
            )
            
            logger.info(f"Generated overall summary using {self._count_tokens(combined_text)} input tokens from {len(top_segments)} segments")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating overall AI summary: {str(e)}")
            return None
    
    def generate_batch_analysis(self, company_batches: List[List[Dict]], cik_to_company: Dict[str, str]) -> List[str]:
        """
        Generate analysis for large datasets by processing in batches
        
        Args:
            company_batches: List of company batches to process
            cik_to_company: Mapping of CIK to company names
            
        Returns:
            List of batch summaries
        """
        if not self.openai_client:
            return []
            
        batch_summaries = []
        
        for i, batch in enumerate(company_batches):
            logger.info(f"Processing batch {i+1}/{len(company_batches)} with {len(batch)} companies")
            
            # Extract top segments from this batch
            batch_segments = []
            for company_data in batch:
                for segment in company_data.get('relevant_segments', [])[:3]:  # Top 3 per company
                    batch_segments.append({
                        'document': segment['text'],
                        'metadata': {'cik': company_data['cik']}
                    })
            
            # Generate summary for this batch
            batch_summary = self.generate_overall_summary(batch_segments, cik_to_company)
            if batch_summary:
                batch_summaries.append(f"Batch {i+1} Analysis:\n{batch_summary}")
                
        return batch_summaries
    
    def analyze_hiring_query_results(self, query_results: Dict, company_details: List[Dict], unique_results: List[Dict], cik_to_company: Dict[str, str]) -> Dict:
        """
        Analyze hiring query results and generate comprehensive analysis
        
        Args:
            query_results: Results from different queries
            company_details: Company-specific details
            unique_results: Unique results across all queries
            cik_to_company: Mapping of CIK to company names
            
        Returns:
            Comprehensive analysis with AI summaries
        """
        # For large datasets, use batch processing
        if len(company_details) > 50:
            logger.info(f"Large dataset detected ({len(company_details)} companies). Using batch processing.")
            
            # Split into batches of 10 companies each
            batch_size = 10
            company_batches = [company_details[i:i+batch_size] for i in range(0, len(company_details), batch_size)]
            
            # Generate batch summaries instead of individual company summaries
            batch_summaries = self.generate_batch_analysis(company_batches, cik_to_company)
            
            # Calculate hiring difficulty scores for all companies in batch mode
            for company_data in company_details:
                company_data['hiring_difficulty_score'] = self.calculate_hiring_difficulty_score(company_data)
            
            # Generate overall summary from top results only
            overall_summary = None
            if unique_results:
                overall_summary = self.generate_overall_summary(unique_results[:15], cik_to_company)
            
            return {
                'company_details': company_details,
                'ai_summary': overall_summary,
                'batch_summaries': batch_summaries,
                'processing_mode': 'batch',
                'total_companies': len(company_details),
                'batches_processed': len(batch_summaries)
            }
        
        else:
            # Standard processing for smaller datasets
            # Generate company-specific summaries
            for company_data in company_details:
                if self.is_available() and len(company_data['relevant_segments']) >= 1:
                    company_summary = self.generate_company_summary(company_data)
                    if company_summary:
                        company_data['ai_summary'] = company_summary
                
                # Calculate hiring difficulty score for each company
                company_data['hiring_difficulty_score'] = self.calculate_hiring_difficulty_score(company_data)
            
            # Generate overall summary
            overall_summary = None
            if self.is_available() and unique_results:
                overall_summary = self.generate_overall_summary(unique_results, cik_to_company)
            
            return {
                'company_details': company_details,
                'ai_summary': overall_summary,
                'processing_mode': 'standard'
            }
    
    def analyze_trends(self, segments: List[str]) -> Optional[str]:
        """
        Analyze trends across multiple segments
        
        Args:
            segments: List of text segments to analyze
            
        Returns:
            Trend analysis or None if not available
        """
        if not self.openai_client or not segments:
            return None
        
        # Smart segment selection within token limits
        available_tokens = self.max_context_tokens - 500 - self.max_completion_tokens  # Reserve for system prompt
        combined_text = self._smart_segment_selection(segments, available_tokens)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Identify common themes and trends in hiring challenges across different companies based on their SEC filings. Focus only on explicitly mentioned challenges."},
                    {"role": "user", "content": f"Identify common hiring and workforce trends from these SEC filing segments:\n\n{combined_text}"}
                ],
                max_tokens=min(500, self.max_completion_tokens),
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return None
    
    def calculate_hiring_difficulty_score(self, company_data: Dict) -> int:
        """
        Calculate hiring difficulty likelihood score on a 1-10 scale
        
        This method evaluates multiple factors from SEC 10-K filings to assess the likelihood
        that a company is facing or will face hiring difficulties:
        
        SCORING FACTORS:
        1. Number of relevant segments (0-3 points):
           - 10+ segments: +3 points (extensive hiring discussions)
           - 5-9 segments: +2 points (moderate hiring discussions) 
           - 2-4 segments: +1 point (some hiring discussions)
           - 0-1 segments: +0 points (minimal/no hiring discussions)
        
        2. Average similarity score (0-3 points):
           - 0.85+: +3 points (highly specific hiring challenges)
           - 0.75-0.84: +2 points (specific hiring challenges)
           - 0.65-0.74: +1 point (some hiring relevance)
           - <0.65: +0 points (low hiring relevance)
        
        3. Section diversity (0-2 points):
           - 4+ different sections: +2 points (hiring issues across multiple areas)
           - 2-3 sections: +1 point (hiring issues in some areas)
           - 0-1 sections: +0 points (limited scope)
        
        4. AI summary severity analysis (0-2 points):
           - 3+ high-severity keywords: +2 points (severe hiring challenges)
           - 1-2 high-severity keywords: +1 point (moderate challenges)
           - 3+ medium-severity keywords: +1 point (general hiring activity)
           - <1 high-severity or <3 medium-severity: +0 points
        
        SCORE INTERPRETATION:
        - 1-3: LOW likelihood of hiring difficulties
        - 4-6: MEDIUM likelihood of hiring difficulties  
        - 7-10: HIGH likelihood of hiring difficulties
        
        Args:
            company_data: Dictionary containing company analysis data including
                         relevant_segments, avg_similarity, sections_found, ai_summary
            
        Returns:
            Integer score from 1 (least likely to face hiring difficulties) to 10 (most likely)
        """
        if not company_data.get('relevant_segments'):
            return 1  # No hiring-related content found
        
        # Factors that contribute to hiring difficulty score
        base_score = 1
        
        # Factor 1: Number of relevant segments (more segments = more hiring challenges mentioned)
        segment_count = len(company_data['relevant_segments'])
        if segment_count >= 10:
            base_score += 3
        elif segment_count >= 5:
            base_score += 2
        elif segment_count >= 2:
            base_score += 1
        
        # Factor 2: Average similarity score (higher similarity = more specific hiring challenges)
        avg_similarity = company_data.get('avg_similarity', 0)
        if avg_similarity >= 0.85:
            base_score += 3
        elif avg_similarity >= 0.75:
            base_score += 2
        elif avg_similarity >= 0.65:
            base_score += 1
        
        # Factor 3: Diversity of sections mentioned (more sections = broader hiring issues)
        sections_count = len(company_data.get('sections_found', []))
        if sections_count >= 4:
            base_score += 2
        elif sections_count >= 2:
            base_score += 1
        
        # Factor 4: Presence of AI summary indicates substantial content
        if company_data.get('ai_summary'):
            # Analyze AI summary for specific keywords that indicate severity
            summary_text = company_data['ai_summary'].lower()
            high_severity_keywords = ['shortage', 'critical', 'severe', 'difficult', 'challenge', 'unable', 'crisis', 'competition for talent']
            medium_severity_keywords = ['hiring', 'recruiting', 'talent', 'workforce', 'employees', 'staffing']
            
            high_severity_matches = sum(1 for keyword in high_severity_keywords if keyword in summary_text)
            medium_severity_matches = sum(1 for keyword in medium_severity_keywords if keyword in summary_text)
            
            if high_severity_matches >= 3:
                base_score += 2
            elif high_severity_matches >= 1:
                base_score += 1
            elif medium_severity_matches >= 3:
                base_score += 1
        
        # Cap the score at 10
        return min(base_score, 10) 