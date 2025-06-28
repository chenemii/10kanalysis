import chromadb
import chromadb.config
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorStore:
    """Handles all ChromaDB vector database operations"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB with persistence
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        # Initialize ChromaDB with persistence - Back to 0.3.29 version
        self.chroma_client = chromadb.Client(chromadb.config.Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Dictionary to store company-specific collections
        self.company_collections = {}
    
    def get_company_collection(self, cik: str, company_name: str = None):
        """
        Get or create a ChromaDB collection for a specific company
        
        Args:
            cik: Company CIK
            company_name: Company display name for metadata
            
        Returns:
            ChromaDB collection for the company
        """
        collection_name = f"company_{cik}"
        
        if collection_name not in self.company_collections:
            display_name = company_name or cik
            
            self.company_collections[collection_name] = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": f"SEC 10-K filings for {display_name}", "company": display_name, "cik": cik}
            )
        
        return self.company_collections[collection_name]
    
    def get_available_companies(self) -> List[str]:
        """
        Get list of companies that have data in the database
        
        Returns:
            List of company CIKs
        """
        collections = self.chroma_client.list_collections()
        companies = []
        
        for collection in collections:
            if collection.name.startswith("company_"):
                cik = collection.name.replace("company_", "")
                companies.append(cik)
        
        return sorted(companies)
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about stored collections
        
        Returns:
            Dictionary with collection statistics
        """
        collections = self.chroma_client.list_collections()
        stats = {
            "total_companies": 0,
            "companies": {},
            "total_chunks_all_companies": 0
        }
        
        for collection in collections:
            if collection.name.startswith("company_"):
                cik = collection.name.replace("company_", "")
                try:
                    actual_collection = self.chroma_client.get_collection(collection.name)
                    count = actual_collection.count()
                    stats["companies"][cik] = {
                        "chunks": count,
                        "collection": collection.name
                    }
                    stats["total_chunks_all_companies"] += count
                    stats["total_companies"] += 1
                except Exception as e:
                    stats["companies"][cik] = {"error": str(e)}
        
        return stats
    
    def store_chunks(self, chunk_dicts: List[Dict], cik_to_company: Dict[str, str]) -> None:
        """
        Store chunk dictionaries in the vector database
        
        Args:
            chunk_dicts: List of chunk dictionaries with metadata
            cik_to_company: Mapping of CIK to company ticker
        """
        if not chunk_dicts:
            logger.warning("No chunks to store")
            return
        
        # Group chunks by company (CIK)
        chunks_by_company = {}
        for chunk_dict in chunk_dicts:
            cik = chunk_dict['filing_info']['cik']
            company_name = cik_to_company.get(cik, cik)
            
            if cik not in chunks_by_company:
                chunks_by_company[cik] = {
                    'chunks': [],
                    'company_name': company_name
                }
            chunks_by_company[cik]['chunks'].append(chunk_dict)
        
        # Store chunks for each company in its own collection
        for cik, company_data in chunks_by_company.items():
            collection = self.get_company_collection(cik, company_data['company_name'])
            company_chunks = company_data['chunks']
            
            # Extract texts for embedding
            texts = [chunk['text'] for chunk in company_chunks]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Prepare data for ChromaDB
            ids = []
            metadatas = []
            documents = []
            
            for i, chunk_dict in enumerate(company_chunks):
                filing_info = chunk_dict['filing_info']
                
                chunk_id = f"{filing_info['cik']}_{filing_info['accessionNumber']}_{chunk_dict['chunk_id']}"
                
                metadata = {
                    'cik': filing_info['cik'],
                    'ticker': company_data['company_name'],
                    'accession_number': filing_info['accessionNumber'],
                    'filing_date': filing_info['filingDate'],
                    'report_date': filing_info['reportDate'],
                    'section': chunk_dict['section'],
                    'priority': chunk_dict['priority'],
                    'chunk_index': chunk_dict['chunk_id'],
                    'chunk_length': len(chunk_dict['text'])
                }
                
                ids.append(chunk_id)
                metadatas.append(metadata)
                documents.append(chunk_dict['text'])
            
            # Store in company-specific ChromaDB collection
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Stored {len(company_chunks)} chunks for {company_data['company_name']} in collection '{collection.name}'")
        
        logger.info(f"Successfully stored chunks for {len(chunks_by_company)} companies")
    
    def query_by_cik(self, query: str, cik: str, company_name: str = None, n_results: int = 10) -> List[Dict]:
        """
        Query a specific company's collection by CIK
        
        Args:
            query: Search query
            cik: Company CIK
            company_name: Company name for display
            n_results: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            collection = self.get_company_collection(cik, company_name)
            
            # Check if collection has any data
            if collection.count() == 0:
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search company-specific collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'company': company_name or cik
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.warning(f"Error querying company {cik}: {str(e)}")
            return []
    
    def query_all_companies(self, query: str, n_results: int = 5, cik_to_company: Dict[str, str] = None) -> List[Dict]:
        """
        Query all company collections
        
        Args:
            query: Search query
            n_results: Number of results to return per company
            cik_to_company: Mapping of CIK to company names
            
        Returns:
            List of relevant chunks with metadata from all companies
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Get all available companies
        company_ciks = self.get_available_companies()
        all_results = []
        
        # Search each company's collection
        for cik in company_ciks:
            try:
                company_name = cik_to_company.get(cik, cik) if cik_to_company else cik
                collection = self.get_company_collection(cik, company_name)
                
                # Check if collection has any data
                if collection.count() == 0:
                    continue
                
                # Search company-specific collection
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Format results with company information
                for i in range(len(results['documents'][0])):
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'company': company_name
                    }
                    all_results.append(result)
                    
            except Exception as e:
                logger.warning(f"Error querying company {cik}: {str(e)}")
                continue
        
        # Sort all results by similarity score
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return all_results
    
    def is_filing_in_db(self, cik: str, accession_number: str) -> bool:
        """
        Check if a filing is already processed and stored in ChromaDB
        
        Args:
            cik: Company CIK
            accession_number: SEC accession number
            
        Returns:
            True if filing exists in database
        """
        try:
            collection = self.get_company_collection(cik)
            
            # Query for any chunk with this CIK and accession number
            results = collection.query(
                query_texts=["test"],  # Dummy query
                n_results=1,
                where={"cik": cik, "accession_number": accession_number}
            )
            return len(results['documents'][0]) > 0
        except Exception as e:
            logger.debug(f"Error checking if filing exists in DB: {e}")
            return False
    
    def get_processed_filings(self) -> Dict[str, set]:
        """
        Get summary of already processed filings by company
        
        Returns:
            Dictionary mapping CIK to set of processed accession numbers
        """
        processed = {}
        company_ciks = self.get_available_companies()
        
        for cik in company_ciks:
            try:
                collection = self.get_company_collection(cik)
                # Get all unique combinations of CIK and accession numbers
                all_results = collection.get()
                if all_results and all_results['metadatas']:
                    for metadata in all_results['metadatas']:
                        filing_cik = metadata.get('cik')
                        accession = metadata.get('accession_number')
                        if filing_cik and accession:
                            if filing_cik not in processed:
                                processed[filing_cik] = set()
                            processed[filing_cik].add(accession)
            except Exception as e:
                logger.debug(f"Error getting processed filings for {cik}: {e}")
        
        return processed
    
    def persist_database(self) -> None:
        """Persist ChromaDB data to disk"""
        try:
            self.chroma_client.persist()
            logger.info("ChromaDB data persisted to disk")
        except Exception as e:
            logger.error(f"Error persisting ChromaDB data: {e}")
    
    def migrate_old_collection(self, cik_to_company: Dict[str, str]) -> bool:
        """
        Migrate data from old single 'sec_filings' collection to company-specific collections
        
        Args:
            cik_to_company: Mapping of CIK to company names
            
        Returns:
            True if migration was performed, False if no migration needed
        """
        try:
            # Check if old collection exists
            collections = self.chroma_client.list_collections()
            old_collection_exists = any(col.name == "sec_filings" for col in collections)
            
            if not old_collection_exists:
                return False
            
            print("Found old 'sec_filings' collection. Migrating to company-specific structure...")
            
            # Get the old collection
            old_collection = self.chroma_client.get_collection("sec_filings")
            
            # Get all data from old collection
            all_data = old_collection.get(include=['embeddings', 'documents', 'metadatas'])
            
            if not all_data['documents']:
                print("Old collection is empty, no migration needed.")
                return False
            
            # Group data by company
            companies_data = {}
            for i, metadata in enumerate(all_data['metadatas']):
                cik = metadata.get('cik', 'unknown')
                company_name = cik_to_company.get(cik, cik)
                
                if cik not in companies_data:
                    companies_data[cik] = {
                        'documents': [],
                        'embeddings': [],
                        'metadatas': [],
                        'ids': [],
                        'company_name': company_name
                    }
                
                companies_data[cik]['documents'].append(all_data['documents'][i])
                companies_data[cik]['embeddings'].append(all_data['embeddings'][i])
                companies_data[cik]['metadatas'].append(metadata)
                companies_data[cik]['ids'].append(all_data['ids'][i])
            
            # Create new company-specific collections and migrate data
            migrated_count = 0
            for cik, data in companies_data.items():
                collection = self.get_company_collection(cik, data['company_name'])
                
                # Add data to new collection
                collection.add(
                    documents=data['documents'],
                    embeddings=data['embeddings'],
                    metadatas=data['metadatas'],
                    ids=data['ids']
                )
                
                migrated_count += len(data['documents'])
                print(f"Migrated {len(data['documents'])} chunks for {data['company_name']}")
            
            # Delete old collection
            self.chroma_client.delete_collection("sec_filings")
            print(f"Migration complete: {migrated_count} total chunks migrated to {len(companies_data)} companies")
            print("Old 'sec_filings' collection deleted.")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during migration: {str(e)}")
            print(f"Migration failed: {str(e)}")
            return False 