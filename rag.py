import streamlit as st
import os
import re
import json
import asyncio
from dotenv import load_dotenv
from urllib.parse import urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from scraper import WebScraper

# Load environment variables
load_dotenv()

class CompanyRAG:
    """RAG system for company data extraction and Q&A."""
    
    def __init__(self):
        self.setup_llms()
        self.setup_chromadb()
        self.vectorstore = None
        self.current_company = None
        
    def setup_llms(self):
        """Setup LLMs for different tasks."""
        # Google Gemini for embeddings only
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if self.gemini_api_key:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=self.gemini_api_key
            )
        
        # ChatGroq for all LLM tasks (URL generation and RAG Q&A)
        self.groq_enabled = False
        if os.getenv('GROQ_API_KEY'):
            try:
                self.groq_llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2
                    # API key will be automatically detected from GROQ_API_KEY env var
                )
                self.groq_enabled = True
            except Exception as e:
                st.warning(f"Groq setup failed: {e}")
    
    def setup_chromadb(self):
        """Initialize ChromaDB for persistent vector storage."""
        try:
            # Create a persistent ChromaDB directory
            self.chroma_persist_directory = "./chroma_db"
            
            # Ensure the directory exists
            os.makedirs(self.chroma_persist_directory, exist_ok=True)
            
            st.info("🗄️ ChromaDB initialized with persistent storage")
            
        except Exception as e:
            st.warning(f"ChromaDB initialization failed: {e}")
            self.chroma_persist_directory = None
    
    def company_name_to_url(self, company_name):
        """Convert company name to URL using Groq."""
        # Debug: Show what company name we received
        st.info(f"🔍 Converting company name: '{company_name}'")
        
        if not self.groq_enabled:
            # Fallback: simple URL generation
            clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', company_name.lower())
            clean_name = clean_name.replace(' ', '').strip()
            
            # Ensure we have a valid name
            if len(clean_name) < 2:
                clean_name = company_name.lower().replace(' ', '').replace('-', '')
            
            # Show the generated URL for debugging
            generated_url = f"https://www.{clean_name}.com"
            st.info(f"🔗 Generated URL (fallback): {generated_url}")
            return generated_url
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """You are a web search expert. Given a company name, generate the most likely official website URL.
                    
                    Rules:
                    1. Return the response in JSON format only
                    2. Use https://www. format for URLs
                    3. For well-known companies, use their exact domain
                    4. For generic names, try .com first
                    5. Remove spaces and special characters
                    
                    Response format:
                    {{"url": "https://www.company.com"}}
                    
                    Examples:
                    - "Apple" -> {{"url": "https://www.apple.com"}}
                    - "Microsoft Corporation" -> {{"url": "https://www.microsoft.com"}}
                    - "TradeIndia" -> {{"url": "https://www.tradeindia.com"}}
                    - "IndiaMART" -> {{"url": "https://www.indiamart.com"}}
                    """
                ),
                (
                    "human",
                    f"Generate the official website URL for company: {company_name}"
                )
            ])
            
            chain = prompt | self.groq_llm
            result = chain.invoke({"company_name": company_name})
            
            response_content = result.content.strip()
            st.info(f"🤖 Groq response: {response_content}")
            
            try:
                # Parse JSON response
                json_response = json.loads(response_content)
                url = json_response.get('url', '')
                
                if url:
                    st.info(f"🔗 Extracted URL: {url}")
                    
                    # Validate URL format
                    if url.startswith('http') and '.' in url and self.validate_url(url):
                        return url
                    else:
                        # Fallback with warning
                        st.warning(f"⚠️ Groq generated invalid URL: {url}, using fallback")
                        clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', company_name.lower())
                        clean_name = clean_name.replace(' ', '').strip()
                        fallback_url = f"https://www.{clean_name}.com"
                        st.info(f"🔗 Fallback URL: {fallback_url}")
                        return fallback_url
                else:
                    raise ValueError("No URL found in JSON response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                st.warning(f"⚠️ Failed to parse JSON response: {e}")
                # Try to extract URL from raw response as fallback
                if response_content.startswith('http') and '.' in response_content:
                    url = response_content
                    st.info(f"🔗 Using raw response as URL: {url}")
                    if self.validate_url(url):
                        return url
                
                # Final fallback
                st.warning("Using name-based fallback URL")
                clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', company_name.lower())
                clean_name = clean_name.replace(' ', '').strip()
                fallback_url = f"https://www.{clean_name}.com"
                st.info(f"🔗 Fallback URL: {fallback_url}")
                return fallback_url
                
        except Exception as e:
            st.warning(f"URL generation failed: {e}")
            # Fallback
            clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', company_name.lower())
            clean_name = clean_name.replace(' ', '')
            return f"https://www.{clean_name}.com"
    
    def validate_url(self, url):
        """Validate URL format and basic accessibility."""
        try:
            parsed = urlparse(url)
            
            # Check if URL has proper scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check if scheme is http or https
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Basic domain validation
            if '.' not in parsed.netloc:
                return False
                
            return True
        except Exception:
            return False
    
    def scrape_company_data(self, url):
        """Scrape company data from URL using WebBaseLoader with Selenium backup."""
        try:
            # Validate URL first
            if not self.validate_url(url):
                return None, "Invalid URL format. Please check the URL and try again."
            
            st.info(f"🔍 Loading data from: {url}")
            
            # Try WebBaseLoader first
            try:
                # Create WebBaseLoader
                loader = WebBaseLoader([url])
                loader.requests_per_second = 1
                
                # Try async loading first, fallback to sync
                try:
                    # Use async loading for better performance
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    docs = loop.run_until_complete(loader.aload())
                    loop.close()
                except Exception:
                    # Fallback to synchronous loading
                    docs = loader.load()
                
                if docs:
                    # Combine all document content
                    text = ""
                    for doc in docs:
                        text += doc.page_content + "\n"
                    
                    # Clean up text more thoroughly
                    # Remove excessive whitespace and clean up
                    import re
                    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
                    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space
                    text = text.strip()
                    
                    # Alternative cleaning method
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk and len(chunk) > 3)
                    
                    # Check if we got meaningful content
                    if len(text.strip()) > 100:
                        # Show preview of scraped content for debugging
                        preview = text[:500] + "..." if len(text) > 500 else text
                        st.info(f"📝 Content preview: {preview}")
                        
                        # Limit text size
                        if len(text) > 50000:
                            text = text[:50000]
                        
                        st.success(f"✅ Successfully loaded {len(text)} characters with WebBaseLoader")
                        return text, None
                    else:
                        raise Exception("No meaningful content extracted")
                else:
                    raise Exception("No documents loaded")
                    
            except Exception as webloader_error:
                error_msg = str(webloader_error)
                
                # Check if it's an SSL certificate error or connection issue
                if any(ssl_indicator in error_msg.lower() for ssl_indicator in 
                       ["ssl", "certificate", "cannot connect to host", "certificate_verify_failed"]):
                    
                    st.warning(f"⚠️ WebBaseLoader failed ({error_msg[:100]}...), trying Selenium backup...")
                    
                    # Use Selenium scraper as backup
                    scraper = WebScraper(delay=1.0, max_retries=2, timeout=30)
                    
                    try:
                        soup, raw_html = scraper.scrape_url(url)
                        
                        if soup:
                            # Extract text content from soup
                            # Remove script and style elements
                            for script in soup(["script", "style"]):
                                script.decompose()
                            
                            # Get text and clean it more thoroughly
                            text = soup.get_text(separator=' ', strip=True)
                            
                            # Remove excessive whitespace and clean up
                            import re
                            text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
                            text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space
                            text = text.strip()
                            
                            # Additional cleaning
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            text = ' '.join(chunk for chunk in chunks if chunk and len(chunk) > 3)
                            
                            # Check if we got meaningful content
                            if len(text.strip()) > 100:
                                # Show preview of scraped content for debugging
                                preview = text[:500] + "..." if len(text) > 500 else text
                                st.info(f"📝 Selenium content preview: {preview}")
                                
                                # Limit text size
                                if len(text) > 50000:
                                    text = text[:50000]
                                
                                st.success(f"✅ Successfully loaded {len(text)} characters with Selenium backup")
                                return text, None
                            else:
                                return None, f"❌ Selenium backup extracted insufficient content from {url}"
                        else:
                            return None, f"❌ Selenium backup failed to scrape {url}"
                            
                    except Exception as selenium_error:
                        return None, f"❌ Both WebBaseLoader and Selenium failed. WebLoader: {error_msg[:100]}... Selenium: {str(selenium_error)[:100]}..."
                    
                    finally:
                        # Clean up scraper resources
                        scraper.close()
                
                else:
                    # For non-SSL errors, return the original error
                    if "Cannot connect to host" in error_msg:
                        return None, f"❌ Cannot connect to {url}. Please check if the website is accessible and the URL is correct."
                    elif "DNS" in error_msg or "nodename nor servname provided" in error_msg:
                        return None, f"❌ DNS error: Unable to resolve {url}. Please verify the URL is correct."
                    else:
                        return None, f"❌ Error loading data: {error_msg}"
            
        except Exception as e:
            return None, f"❌ Unexpected error: {str(e)}"
    
    def create_vector_store(self, text, company_name):
        """Create vector store from company text data."""
        try:
            if not self.gemini_api_key:
                return "Google Gemini API key not found. Please add GEMINI_API_KEY to your .env file."
            
            st.info("📊 Processing and storing company data...")
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=400,
                length_function=len
            )
            
            chunks = text_splitter.split_text(text)
            
            if not chunks:
                return "No meaningful text found to process."
            
            # Debug: Show chunk information
            st.info(f"📊 Created {len(chunks)} text chunks")
            if chunks:
                first_chunk_preview = chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0]
                st.info(f"📝 First chunk preview: {first_chunk_preview}")
            
            # Create vector store with persistent storage
            collection_name = f"company_{company_name.lower().replace(' ', '_').replace('-', '_')}"
            
            if self.chroma_persist_directory:
                # Use persistent ChromaDB
                self.vectorstore = Chroma.from_texts(
                    chunks, 
                    embedding=self.embeddings,
                    collection_name=collection_name,
                    persist_directory=self.chroma_persist_directory
                )
                st.info(f"📊 Data stored in persistent ChromaDB collection: {collection_name}")
            else:
                # Fallback to in-memory ChromaDB
                self.vectorstore = Chroma.from_texts(
                    chunks, 
                    embedding=self.embeddings,
                    collection_name=collection_name
                )
                st.warning("⚠️ Using in-memory ChromaDB (data will be lost on restart)")
            
            self.current_company = company_name
            
            return f"✅ Successfully processed {len(chunks)} chunks of data for {company_name}"
            
        except Exception as e:
            return f"Error creating vector store: {str(e)}"
    
    def ask_question(self, query):
        """Ask question about the company data."""
        try:
            if not self.vectorstore:
                return "No company data loaded. Please scrape a company first."
            
            if not self.groq_enabled:
                return "Groq API key not found or setup failed."
            
            # Search for relevant documents
            docs = self.vectorstore.similarity_search(query=query, k=3)
            
            if not docs:
                return "No relevant information found for your query."
            
            # Debug: Show retrieved context
            st.info(f"🔍 Retrieved {len(docs)} relevant chunks for query")
            for i, doc in enumerate(docs):
                preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                st.info(f"📄 Chunk {i+1}: {preview}")
            
            # Create QA chain
            chain = load_qa_chain(llm=self.groq_llm, chain_type="stuff")
            
            # Get response
            response = chain.run(input_documents=docs, question=query)
            
            return response
            
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    def get_company_summary(self):
        """Get a summary of the loaded company."""
        if not self.vectorstore:
            return "No company data loaded."
        
        summary_query = f"Provide a comprehensive summary of {self.current_company} including their business, services, contact information, and key details."
        return self.ask_question(summary_query)
    
    def list_existing_collections(self):
        """List existing ChromaDB collections."""
        try:
            if self.chroma_persist_directory and os.path.exists(self.chroma_persist_directory):
                # This is a simple way to check for existing collections
                # In a more advanced setup, you'd query ChromaDB directly
                return True
            return False
        except Exception:
            return False
    
    def load_existing_collection(self, company_name):
        """Load an existing ChromaDB collection for a company."""
        try:
            if not self.gemini_api_key or not self.chroma_persist_directory:
                return False
            
            collection_name = f"company_{company_name.lower().replace(' ', '_').replace('-', '_')}"
            
            # Try to load existing collection
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_persist_directory
            )
            
            # Test if collection has data
            test_results = self.vectorstore.similarity_search("test", k=1)
            if test_results:
                self.current_company = company_name
                return True
            
        except Exception as e:
            st.warning(f"Could not load existing collection: {e}")
        
        return False
    
    def cleanup(self):
        """Cleanup resources."""
        if self.vectorstore:
            try:
                self.vectorstore.delete_collection()
            except:
                pass

def main():
    st.set_page_config(
        page_title="Company RAG System",
        page_icon="🏢",
        layout="wide"
    )
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = CompanyRAG()
    
    if 'company_loaded' not in st.session_state:
        st.session_state.company_loaded = False
    
    rag = st.session_state.rag_system
    
    # Sidebar
    with st.sidebar:
        st.title("🏢 Company RAG System")
        st.markdown('''
        **Powered by:**
        - LangChain & Google Gemini Pro
        - Web Scraping & Vector Search
        - Real-time Company Analysis
        
        ## 🔧 Features
        - Company name → URL conversion
        - Web scraping & data extraction
        - Vector storage & similarity search
        - Q&A with company data
        
        ## 👨‍💻 By Harshal Nelge
        - [LinkedIn](https://www.linkedin.com/in/harshal-nelge-3a35a6252/)
        ''')
        
        # API Key Status
        st.markdown("### 🔑 API Status")
        if rag.gemini_api_key:
            st.success("✅ Gemini API Connected (Embeddings)")
        else:
            st.error("❌ Gemini API Missing (Required for Embeddings)")
        
        if rag.groq_enabled:
            st.success("✅ Groq API Connected (LLM)")
        else:
            st.error("❌ Groq API Missing (Required for LLM)")
    
    # Main interface
    st.header("🏢 Company Intelligence RAG System", divider='rainbow')
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_type = st.radio(
            "Choose input type:",
            ["Company Name", "Direct URL"],
            horizontal=True
        )
        
        if input_type == "Company Name":
            company_input = st.text_input(
                "🏢 Enter company or directory name:",
                placeholder="e.g., IndiaMART, Microsoft, TradeIndia",
                help="Enter a well-known company name. Use the Quick Examples for guaranteed working companies."
            )
            if company_input:
                url = rag.company_name_to_url(company_input)
                # URL generation now includes debug info
        else:
            url = st.text_input(
                "🌐 Enter company website URL:",
                placeholder="https://www.example.com"
            )
            company_input = url.split('//')[1].split('.')[1] if '//' in url and '.' in url else "Unknown Company"
    
    with col2:
        st.markdown("### 🎯 Quick Examples")
        examples = [
            "IndiaMART",
            "TradeIndia", 
            "Microsoft",
            "Apple"
        ]
        
        # Add note about known working companies
        st.caption("💡 These are verified working companies")
        
        for example in examples:
            if st.button(f"📌 {example}", key=f"ex_{example}"):
                st.session_state.example_input = example
    
    # Handle example clicks
    if 'example_input' in st.session_state:
        company_input = st.session_state.example_input
        url = rag.company_name_to_url(company_input)
        del st.session_state.example_input
    
    # Scrape button
    if st.button("🔍 Load & Analyze Company", type="primary"):
        if 'url' in locals() and url:
            with st.spinner("Processing company data..."):
                # Scrape data
                text_data, error = rag.scrape_company_data(url)
                
                if error:
                    st.error(f"❌ {error}")
                elif text_data:
                    # Create vector store
                    result = rag.create_vector_store(text_data, company_input)
                    
                    if "✅" in result:
                        st.success(result)
                        st.session_state.company_loaded = True
                        
                        # Show company summary
                        with st.expander("📋 Company Summary", expanded=True):
                            summary = rag.get_company_summary()
                            st.write(summary)
                    else:
                        st.error(f"❌ {result}")
                else:
                    st.error("❌ No data extracted from the website.")
        else:
            st.error("❌ Please enter a company name or URL.")
    
    # Q&A Section
    if st.session_state.company_loaded:
        st.markdown("---")
        st.subheader(f"💬 Ask Questions about {rag.current_company}")
        
        # Predefined questions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 What services do they offer?"):
                st.session_state.selected_question = "What products and services does this company offer?"
        
        with col2:
            if st.button("📞 Contact information?"):
                st.session_state.selected_question = "What is the contact information including phone, email, and address?"
        
        with col3:
            if st.button("🏭 Company background?"):
                st.session_state.selected_question = "Tell me about the company's history, founding, and background."
        
        # Question input
        default_question = st.session_state.get('selected_question', '')
        question = st.text_input(
            "🤔 Your question:",
            value=default_question,
            placeholder="Ask anything about the company..."
        )
        
        if question:
            with st.spinner("🔍 Searching for answer..."):
                answer = rag.ask_question(question)
                
                # Display answer
                st.markdown("### 💡 Answer:")
                st.markdown(f"> **Question:** {question}")
                st.write(answer)
                
                # Clear selected question
                if 'selected_question' in st.session_state:
                    del st.session_state.selected_question
    
    else:
        st.info("👆 Load a company first to start asking questions!")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Clear Data"):
            rag.cleanup()
            st.session_state.company_loaded = False
            st.success("Data cleared!")
            st.rerun()
    
    with col2:
        if st.session_state.company_loaded:
            st.success(f"📊 Company Loaded: {rag.current_company}")
        else:
            st.info("📊 No company loaded")
    
    with col3:
        st.info("🔧 Built with Streamlit + LangChain")

if __name__ == "__main__":
    main()
