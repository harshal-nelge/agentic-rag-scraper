
import os
import sys
import csv
import json
import re
import logging
import argparse
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urlparse
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import gspread
from google.oauth2.service_account import Credentials
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup

# Import scraper
from scraper import WebScraper

# AI imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class CompanyData(BaseModel):
    """Schema for structured company data."""
    company_name: Optional[str] = Field(description="Clean company name")
    website: Optional[str] = Field(description="Official website URL")
    email: Optional[str] = Field(description="Primary contact email")
    phone: Optional[str] = Field(description="Primary phone number")
    address: Optional[str] = Field(description="Complete business address")
    description: Optional[str] = Field(description="Company description")
    category: Optional[str] = Field(description="Business category or industry")
    social_facebook: Optional[str] = Field(description="Facebook URL")
    social_twitter: Optional[str] = Field(description="Twitter URL")
    social_linkedin: Optional[str] = Field(description="LinkedIn URL")
    social_instagram: Optional[str] = Field(description="Instagram URL")
    services: Optional[List[str]] = Field(description="Products or services offered")
    founding_year: Optional[str] = Field(description="Year founded")
    employee_count: Optional[str] = Field(description="Number of employees")
    confidence_score: Optional[float] = Field(description="AI confidence (0-1)")

class ObserveNowScraper:
    """Main scraper class with integrated AI processing."""
    
    def __init__(self):
        self.setup_logging()
        self.scraper = WebScraper(delay=1.0, max_retries=3, timeout=30)
        self.setup_ai()
        self.setup_google_sheets()
        
        # Statistics
        self.stats = {
            'urls_processed': 0,
            'successful_scrapes': 0,
            'failed_scrapes': 0,
            'ai_extractions': 0,
            'manual_extractions': 0,
            'errors': []
        }
    
    def setup_logging(self):
        """Setup logging."""
        os.makedirs('logs', exist_ok=True)
        log_file = f"logs/scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_ai(self):
        """Setup ChatGroq AI model."""
        self.ai_enabled = False
        
        # Check if GROQ_API_KEY exists in environment
        if os.getenv('GROQ_API_KEY'):
            try:
                self.llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2
                    # API key will be automatically detected from GROQ_API_KEY env var
                )
                
                self.parser = JsonOutputParser(pydantic_object=CompanyData)
                
                # Create extraction prompt
                self.extraction_prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        """You are an expert data extraction specialist. Extract structured company information from the provided HTML content.

INSTRUCTIONS:
1. Extract clean, accurate company information
2. Standardize phone numbers to international format (+1-XXX-XXX-XXXX)
3. Clean and validate email addresses
4. Extract social media links for Facebook, Twitter, LinkedIn, Instagram
5. Identify products/services offered
6. Look for founding year and employee count
7. Provide a confidence score (0-1) based on data quality
8. Return ONLY valid JSON matching the schema
9. Use null for missing information

{format_instructions}"""
                    ),
                    (
                        "human",
                        """Extract company data from this website:

URL: {url}
HTML Content: {html_content}

Return structured JSON with the company information."""
                    )
                ])
                
                self.ai_enabled = True
                self.logger.info("AI extraction enabled with ChatGroq")
                
            except Exception as e:
                self.logger.warning(f"Failed to setup AI: {e}")
                self.ai_enabled = False
        else:
            self.logger.warning("GROQ_API_KEY not found. Using manual extraction only.")
    
    def setup_google_sheets(self):
        """Setup Google Sheets client."""
        self.sheets_enabled = False
        
        try:
            # Get credentials file and spreadsheet ID from environment
            credentials_file = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE', 'credentials.json')
            self.spreadsheet_id = os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID')
            
            if not self.spreadsheet_id:
                self.logger.warning("GOOGLE_SHEETS_SPREADSHEET_ID not found in environment variables")
                return
            
            if not os.path.exists(credentials_file):
                self.logger.warning(f"Google Sheets credentials file not found: {credentials_file}")
                return
            
            # Define the scope
            scope = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Authenticate and create the service
            creds = Credentials.from_service_account_file(credentials_file, scopes=scope)
            self.gc = gspread.authorize(creds)
            
            # Test connection by opening the spreadsheet
            self.spreadsheet = self.gc.open_by_key(self.spreadsheet_id)
            
            self.sheets_enabled = True
            self.logger.info("Google Sheets integration enabled successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup Google Sheets: {e}")
            self.sheets_enabled = False
    
    def extract_with_ai(self, url, html_content):
        """Extract data using ChatGroq AI."""
        try:
            # Limit HTML content to avoid token limits
            limited_html = html_content[:15000] if html_content else ""
            
            input_data = {
                "url": url,
                "html_content": limited_html,
                "format_instructions": self.parser.get_format_instructions()
            }
            
            chain = self.extraction_prompt | self.llm | self.parser
            result = chain.invoke(input_data)
            
            self.stats['ai_extractions'] += 1
            self.logger.info(f"AI extraction successful for {url}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"AI extraction failed for {url}: {e}")
            return None
    
    def extract_manually(self, soup, url):
        """Manual extraction as backup when AI fails."""
        try:
            data = {}
            
            # Extract company name
            data['company_name'] = self.extract_company_name(soup)
            data['website'] = self.clean_url(url)
            data['email'] = self.extract_email(soup)
            data['phone'] = self.extract_phone(soup)
            data['address'] = self.extract_address(soup)
            data['description'] = self.extract_description(soup)
            data['category'] = self.extract_category(soup)
            
            # Extract social links
            social_links = self.extract_social_links(soup)
            data['social_facebook'] = social_links.get('facebook')
            data['social_twitter'] = social_links.get('twitter')
            data['social_linkedin'] = social_links.get('linkedin')
            data['social_instagram'] = social_links.get('instagram')
            
            data['confidence_score'] = 0.7  # Manual extraction confidence
            
            self.stats['manual_extractions'] += 1
            return data
            
        except Exception as e:
            self.logger.error(f"Manual extraction failed for {url}: {e}")
            return None
    
    def extract_company_name(self, soup):
        """Extract company name from soup."""
        # Try title tag
        title = soup.find('title')
        if title:
            name = title.get_text().strip()
            # Clean common patterns
            name = re.sub(r'\s*[-|–—]\s*.*$', '', name)
            name = re.sub(r'^\s*Welcome to\s*', '', name, flags=re.IGNORECASE)
            if len(name) > 2 and len(name) < 100:
                return name
        
        # Try h1 tags
        h1_tags = soup.find_all('h1')
        for h1 in h1_tags[:3]:  # Check first 3 h1 tags
            text = h1.get_text().strip()
            if len(text) > 2 and len(text) < 100:
                return text
        
        return None
    
    def extract_email(self, soup):
        """Extract email from soup."""
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Check mailto links
        mailto_links = soup.find_all('a', href=re.compile(r'^mailto:'))
        for link in mailto_links:
            email = link.get('href').replace('mailto:', '').strip()
            if email_pattern.match(email):
                return email
        
        # Search in text
        text = soup.get_text()
        emails = email_pattern.findall(text)
        # Filter out common noise
        for email in emails:
            if not any(noise in email.lower() for noise in ['example.com', 'test.com', 'admin@admin']):
                return email
        
        return None
    
    def extract_phone(self, soup):
        """Extract phone number from soup."""
        phone_pattern = re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})|(\+?91[-.\s]?)?([0-9]{10})')
        
        # Check tel links
        tel_links = soup.find_all('a', href=re.compile(r'^tel:'))
        for link in tel_links:
            phone = link.get('href').replace('tel:', '').strip()
            match = phone_pattern.search(phone)
            if match:
                return phone
        
        # Search in text
        text = soup.get_text()
        match = phone_pattern.search(text)
        if match:
            return match.group()
        
        return None
    
    def extract_address(self, soup):
        """Extract address from soup."""
        # Look for address in common selectors
        address_selectors = ['.address', '.location', '.contact-address', '[itemtype*="PostalAddress"]']
        
        for selector in address_selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text().strip()
                if len(text) > 10 and any(indicator in text.lower() for indicator in ['street', 'ave', 'road', 'city']):
                    return text
        
        return None
    
    def extract_description(self, soup):
        """Extract description from soup."""
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            desc = meta_desc.get('content', '').strip()
            if len(desc) > 50:
                return desc
        
        # Try common description selectors
        desc_selectors = ['.description', '.about', '.company-description', '.overview']
        for selector in desc_selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text().strip()
                if len(text) > 50 and len(text) < 1000:
                    return text
        
        return None
    
    def extract_category(self, soup):
        """Extract category from soup."""
        # Try meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            keywords = meta_keywords.get('content', '')
            if keywords:
                return keywords.split(',')[0].strip()
        
        return None
    
    def extract_social_links(self, soup):
        """Extract social media links."""
        social_links = {}
        
        patterns = {
            'facebook': re.compile(r'(?:https?://)?(?:www\.)?facebook\.com/[^\s<>"\']+'),
            'twitter': re.compile(r'(?:https?://)?(?:www\.)?(?:twitter\.com|x\.com)/[^\s<>"\']+'),
            'linkedin': re.compile(r'(?:https?://)?(?:www\.)?linkedin\.com/[^\s<>"\']+'),
            'instagram': re.compile(r'(?:https?://)?(?:www\.)?instagram\.com/[^\s<>"\']+')
        }
        
        links = soup.find_all('a', href=True)
        for link in links:
            href = link.get('href', '')
            for platform, pattern in patterns.items():
                if pattern.search(href) and platform not in social_links:
                    social_links[platform] = href
                    break
        
        return social_links
    
    def clean_url(self, url):
        """Clean and normalize URL."""
        parsed = urlparse(url)
        return f"https://{parsed.netloc}"
    
    def load_urls_from_csv(self, csv_file):
        """Load URLs from CSV file."""
        urls = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Look for URL in common column names
                    url = None
                    for col in ['url', 'URL', 'website', 'link']:
                        if col in row and row[col].strip():
                            url = row[col].strip()
                            break
                    if url:
                        urls.append(url)
            
            self.logger.info(f"Loaded {len(urls)} URLs from {csv_file}")
            return urls
        except Exception as e:
            self.logger.error(f"Error loading URLs: {e}")
            return []
    
    def scrape_with_webbaseloader(self, url):
        """Scrape using WebBaseLoader (primary method)."""
        try:
            self.logger.info(f"Scraping {url} with WebBaseLoader")
            
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
                text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
                text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space
                text = text.strip()
                
                # Check if we got meaningful content
                if len(text.strip()) > 100:
                    self.logger.info(f"✅ WebBaseLoader extracted {len(text)} characters")
                    
                    # Print scraped content details
                    print("\n" + "="*80)
                    print("🔍 WEBBASELOADER SCRAPED CONTENT DETAILS")
                    print("="*80)
                    print(f"📏 Content Length: {len(text)} characters")
                    print(f"📄 First 500 characters:")
                    print("-" * 50)
                    print(text[:500])
                    print("-" * 50)
                    if len(text) > 1000:
                        print(f"📄 Last 500 characters:")
                        print("-" * 50)
                        print(text[-500:])
                        print("-" * 50)
                    
                    # Show some content analysis
                    word_count = len(text.split())
                    line_count = len(text.splitlines())
                    print(f"📊 Word Count: {word_count}")
                    print(f"📊 Line Count: {line_count}")
                    
                    # Check for social media links in the content
                    social_patterns = {
                        'Facebook': r'facebook\.com/[^\s<>"\']+',
                        'Twitter/X': r'(?:twitter\.com|x\.com)/[^\s<>"\']+',
                        'LinkedIn': r'linkedin\.com/[^\s<>"\']+',
                        'Instagram': r'instagram\.com/[^\s<>"\']+',
                        'YouTube': r'youtube\.com/[^\s<>"\']+',
                    }
                    
                    print("🔗 Social Media Links Found:")
                    for platform, pattern in social_patterns.items():
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            print(f"  {platform}: {len(matches)} links")
                            for match in matches[:3]:  # Show first 3 matches
                                print(f"    - {match}")
                            if len(matches) > 3:
                                print(f"    ... and {len(matches) - 3} more")
                        else:
                            print(f"  {platform}: No links found")
                    
                    # Check for contact information
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
                    
                    emails = re.findall(email_pattern, text)
                    phones = re.findall(phone_pattern, text)
                    
                    print("📧 Contact Information Found:")
                    if emails:
                        print(f"  Emails: {len(emails)} found")
                        for email in emails[:3]:
                            print(f"    - {email}")
                        if len(emails) > 3:
                            print(f"    ... and {len(emails) - 3} more")
                    else:
                        print("  Emails: None found")
                        
                    if phones:
                        print(f"  Phones: {len(phones)} found")
                        for phone in phones[:3]:
                            phone_str = ''.join(phone) if isinstance(phone, tuple) else phone
                            print(f"    - {phone_str}")
                        if len(phones) > 3:
                            print(f"    ... and {len(phones) - 3} more")
                    else:
                        print("  Phones: None found")
                    
                    print("="*80)
                    print()
                    
                    return text
                else:
                    raise Exception("No meaningful content extracted")
            else:
                raise Exception("No documents loaded")
                
        except Exception as e:
            self.logger.warning(f"WebBaseLoader failed for {url}: {str(e)[:100]}...")
            return None
    
    def process_url(self, url):
        """Process a single URL and extract data."""
        try:
            self.logger.info(f"Processing: {url}")
            
            # Try WebBaseLoader first
            raw_html = self.scrape_with_webbaseloader(url)
            soup = None
            
            if raw_html:
                # Create soup from WebBaseLoader content for manual extraction
                soup = BeautifulSoup(raw_html, 'html.parser')
                self.stats['successful_scrapes'] += 1
            else:
                # Fallback to traditional scraper
                self.logger.info(f"Falling back to traditional scraper for {url}")
                soup, raw_html = self.scraper.scrape_url(url)
                if not soup:
                    self.stats['failed_scrapes'] += 1
                    return None
                self.stats['successful_scrapes'] += 1
            
            # Try AI extraction first
            if self.ai_enabled and raw_html:
                data = self.extract_with_ai(url, raw_html)
                if data:
                    data['source_url'] = url
                    return data
            
            # Fallback to manual extraction
            if soup:
                self.logger.info(f"Using manual extraction for {url}")
                data = self.extract_manually(soup, url)
                if data:
                    data['source_url'] = url
                    return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing {url}: {e}")
            self.stats['errors'].append(f"Error processing {url}: {str(e)}")
            return None
    
    def export_to_google_sheets(self, companies):
        """Export companies data to Google Sheets."""
        if not companies:
            self.logger.warning("No data to export")
            return False
        
        if not self.sheets_enabled:
            self.logger.error("Google Sheets not enabled. Cannot export data.")
            return False
        
        try:
            # Debug: Print the raw companies data
            self.logger.info(f"Raw companies data: {companies}")
            
            # Convert to DataFrame
            df = pd.DataFrame(companies)
            
            # Debug: Print DataFrame info
            self.logger.info(f"DataFrame shape: {df.shape}")
            self.logger.info(f"DataFrame columns: {list(df.columns)}")
            
            # Reorder columns for better readability
            column_order = [
                'company_name', 'website', 'email', 'phone', 'address', 
                'description', 'category', 'social_facebook', 'social_twitter',
                'social_linkedin', 'social_instagram', 'services', 'founding_year',
                'employee_count', 'confidence_score', 'source_url'
            ]
            
            # Keep only existing columns and add missing ones with None
            for col in column_order:
                if col not in df.columns:
                    df[col] = None
            
            df = df[column_order]
            
            # Get or create the worksheet
            try:
                worksheet = self.spreadsheet.worksheet("Sheet1")
                self.logger.info("Found existing Sheet1")
            except gspread.WorksheetNotFound:
                worksheet = self.spreadsheet.add_worksheet(title="Sheet1", rows="1000", cols="20")
                self.logger.info("Created new Sheet1")
            
            # Clear existing data
            worksheet.clear()
            
            # Prepare data for upload
            # Convert DataFrame to list of lists with headers
            data_to_upload = [column_order]  # Headers
            
            for _, row in df.iterrows():
                row_data = []
                for col in column_order:
                    try:
                        value = row[col]
                        
                        # Handle None values
                        if value is None:
                            row_data.append("")
                            continue
                            
                        # Handle pandas NA values
                        if pd.isna(value):
                            row_data.append("")
                            continue
                            
                        # Handle lists/arrays
                        if isinstance(value, (list, tuple)):
                            if len(value) == 0:
                                row_data.append("")
                            else:
                                clean_items = [str(item) for item in value if item is not None and str(item).strip() != 'nan']
                                row_data.append(", ".join(clean_items))
                            continue
                            
                        # Handle numpy arrays
                        if hasattr(value, '__array__'):
                            try:
                                # Convert array to list first
                                array_list = value.tolist() if hasattr(value, 'tolist') else list(value)
                                if len(array_list) == 0:
                                    row_data.append("")
                                else:
                                    clean_items = [str(item) for item in array_list if item is not None and str(item).strip() != 'nan']
                                    row_data.append(", ".join(clean_items))
                            except:
                                row_data.append(str(value))
                            continue
                            
                        # Handle regular values
                        str_value = str(value).strip()
                        if str_value in ['nan', 'None', '']:
                            row_data.append("")
                        else:
                            row_data.append(str_value)
                            
                    except Exception as e:
                        self.logger.warning(f"Error processing value for column {col}: {e}")
                        row_data.append("")
                        
                data_to_upload.append(row_data)
            
            # Upload data to Google Sheets
            worksheet.update(data_to_upload, value_input_option='USER_ENTERED')
            
            # Format the header row
            worksheet.format('1:1', {
                'backgroundColor': {
                    'red': 0.2,
                    'green': 0.6,
                    'blue': 0.9
                },
                'textFormat': {
                    'bold': True,
                    'foregroundColor': {
                        'red': 1.0,
                        'green': 1.0,
                        'blue': 1.0
                    }
                }
            })
            
            # Auto-resize columns
            worksheet.columns_auto_resize(0, len(column_order))
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            worksheet.update(values=[[f"Last updated: {timestamp}"]], range_name='A1000')
            
            spreadsheet_url = f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}"
            self.logger.info(f"Exported {len(companies)} companies to Google Sheets: {spreadsheet_url}")
            
            return spreadsheet_url
            
        except Exception as e:
            self.logger.error(f"Error exporting to Google Sheets: {e}")
            return False
    
    def print_summary(self):
        """Print execution summary."""
        print("\n" + "="*60)
        print("               OBSERVENOW SCRAPING SUMMARY")
        print("="*60)
        
        print(f"URLs Processed:           {self.stats['urls_processed']}")
        print(f"Successful Scrapes:       {self.stats['successful_scrapes']}")
        print(f"Failed Scrapes:           {self.stats['failed_scrapes']}")
        if self.stats['urls_processed'] > 0:
            success_rate = (self.stats['successful_scrapes'] / self.stats['urls_processed']) * 100
            print(f"Success Rate:             {success_rate:.1f}%")
        
        print(f"\nAI Extractions:           {self.stats['ai_extractions']}")
        print(f"Manual Extractions:       {self.stats['manual_extractions']}")
        
        if self.stats['errors']:
            print(f"\nErrors Encountered:       {len(self.stats['errors'])}")
            for error in self.stats['errors'][:3]:
                print(f"  - {error}")
            if len(self.stats['errors']) > 3:
                print(f"  ... and {len(self.stats['errors']) - 3} more errors")
        
        print("="*60)
    
    def run(self, input_csv):
        """Main execution method."""
        try:
            # Load URLs
            urls = self.load_urls_from_csv(input_csv)
            if not urls:
                raise ValueError("No valid URLs found in CSV file")
            
            self.stats['urls_processed'] = len(urls)
            
            # Process URLs
            companies = []
            for url in tqdm(urls, desc="Scraping URLs"):
                data = self.process_url(url)
                if data:
                    companies.append(data)
            
            # Export results to Google Sheets
            spreadsheet_url = self.export_to_google_sheets(companies)
            
            # Print summary
            self.print_summary()
            
            print(f"\n✅ Scraping completed!")
            if spreadsheet_url:
                print(f"📊 Results exported to Google Sheets: {spreadsheet_url}")
                print(f"📋 Sheet name: observenow assign")
                print(f"📄 Worksheet: Sheet1")
            else:
                print("❌ Failed to export to Google Sheets")
            
            return spreadsheet_url
            
        except Exception as e:
            self.logger.error(f"Error in main execution: {e}")
            raise
        finally:
            self.scraper.close()

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='ObserveNow Web Data Extraction System')
    parser.add_argument('input_csv', help='CSV file containing URLs to scrape')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file '{args.input_csv}' not found")
        sys.exit(1)
    
    # Run scraper
    scraper = ObserveNowScraper()
    
    try:
        spreadsheet_url = scraper.run(args.input_csv)
        if spreadsheet_url:
            print(f"\n🎉 Success! Results exported to Google Sheets: {spreadsheet_url}")
        else:
            print(f"\n⚠️ Scraping completed but export to Google Sheets failed.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()