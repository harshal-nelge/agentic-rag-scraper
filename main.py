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
from scraper import WebScraper
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

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
            handlers=[logging.FileHandler(log_file)]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_ai(self):
        """Setup ChatGroq AI model."""
        self.ai_enabled = False
        
        if os.getenv('GROQ_API_KEY'):
            try:
                self.llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2
                )
                
                self.parser = JsonOutputParser(pydantic_object=CompanyData)
                
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
            credentials_file = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE', 'credentials.json')
            self.spreadsheet_id = os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID')
            
            if not self.spreadsheet_id or not os.path.exists(credentials_file):
                return
            
            scope = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            creds = Credentials.from_service_account_file(credentials_file, scopes=scope)
            self.gc = gspread.authorize(creds)
            self.spreadsheet = self.gc.open_by_key(self.spreadsheet_id)
            
            self.sheets_enabled = True
            self.logger.info("Google Sheets integration enabled")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup Google Sheets: {e}")
            self.sheets_enabled = False
    
    def extract_with_ai(self, url, html_content):
        """Extract data using ChatGroq AI."""
        try:
            limited_html = html_content[:15000] if html_content else ""
            
            input_data = {
                "url": url,
                "html_content": limited_html,
                "format_instructions": self.parser.get_format_instructions()
            }
            
            chain = self.extraction_prompt | self.llm | self.parser
            result = chain.invoke(input_data)
            
            self.stats['ai_extractions'] += 1
            return result
            
        except Exception as e:
            self.logger.error(f"AI extraction failed for {url}: {e}")
            return None
    
    def extract_manually(self, soup, url):
        """Manual extraction as backup when AI fails."""
        try:
            data = {}
            data['company_name'] = self.extract_company_name(soup)
            data['website'] = self.clean_url(url)
            data['email'] = self.extract_email(soup)
            data['phone'] = self.extract_phone(soup)
            data['address'] = self.extract_address(soup)
            data['description'] = self.extract_description(soup)
            data['category'] = self.extract_category(soup)
            
            social_links = self.extract_social_links(soup)
            data['social_facebook'] = social_links.get('facebook')
            data['social_twitter'] = social_links.get('twitter')
            data['social_linkedin'] = social_links.get('linkedin')
            data['social_instagram'] = social_links.get('instagram')
            
            data['confidence_score'] = 0.7
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
        for email in emails:
            if not any(noise in email.lower() for noise in ['example.com', 'test.com', 'admin@admin']):
                return email
        
        return None
    
    def extract_phone(self, soup):
        """Extract phone number from soup."""
        phone_pattern = re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})|(\+?91[-.\s]?)?([0-9]{10})')
        
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
                
                if len(text.strip()) > 100:
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
            
            raw_html = self.scrape_with_webbaseloader(url)
            soup = None
            
            if raw_html:
                soup = BeautifulSoup(raw_html, 'html.parser')
                self.stats['successful_scrapes'] += 1
            else:
                soup, raw_html = self.scraper.scrape_url(url)
                if not soup:
                    self.stats['failed_scrapes'] += 1
                    return None
                self.stats['successful_scrapes'] += 1
            
            if self.ai_enabled and raw_html:
                data = self.extract_with_ai(url, raw_html)
                if data:
                    data['source_url'] = url
                    return data
            
            if soup:
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
            df = pd.DataFrame(companies)
            
            column_order = [
                'company_name', 'website', 'email', 'phone', 'address', 
                'description', 'category', 'social_facebook', 'social_twitter',
                'social_linkedin', 'social_instagram', 'services', 'founding_year',
                'employee_count', 'confidence_score', 'source_url'
            ]
            
            for col in column_order:
                if col not in df.columns:
                    df[col] = None
            
            df = df[column_order]
            
            try:
                worksheet = self.spreadsheet.worksheet("Sheet1")
            except gspread.WorksheetNotFound:
                worksheet = self.spreadsheet.add_worksheet(title="Sheet1", rows="1000", cols="20")
            
            worksheet.clear()
            data_to_upload = [column_order]
            
            for _, row in df.iterrows():
                row_data = []
                for col in column_order:
                    try:
                        value = row[col]
                        
                        if value is None or pd.isna(value):
                            row_data.append("")
                            continue
                            
                        if isinstance(value, (list, tuple)):
                            if len(value) == 0:
                                row_data.append("")
                            else:
                                clean_items = [str(item) for item in value if item is not None and str(item).strip() != 'nan']
                                row_data.append(", ".join(clean_items))
                            continue
                            
                        if hasattr(value, '__array__'):
                            try:
                                array_list = value.tolist() if hasattr(value, 'tolist') else list(value)
                                if len(array_list) == 0:
                                    row_data.append("")
                                else:
                                    clean_items = [str(item) for item in array_list if item is not None and str(item).strip() != 'nan']
                                    row_data.append(", ".join(clean_items))
                            except:
                                row_data.append(str(value))
                            continue
                            
                        str_value = str(value).strip()
                        if str_value in ['nan', 'None', '']:
                            row_data.append("")
                        else:
                            row_data.append(str_value)
                            
                    except Exception as e:
                        self.logger.warning(f"Error processing value for column {col}: {e}")
                        row_data.append("")
                        
                data_to_upload.append(row_data)
            
            worksheet.update(data_to_upload, value_input_option='USER_ENTERED')
            
            worksheet.format('1:1', {
                'backgroundColor': {'red': 0.2, 'green': 0.6, 'blue': 0.9},
                'textFormat': {'bold': True, 'foregroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}
            })
            
            worksheet.columns_auto_resize(0, len(column_order))
            
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