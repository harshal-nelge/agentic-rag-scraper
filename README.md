# ObserveNow Assignment - Simplified Web Data Extraction

A streamlined system for scraping company data from websites and converting it to structured CSV using ChatGroq AI with manual extraction as backup.

## 🚀 Features

- **Smart Web Scraping**: BeautifulSoup + Selenium fallback for JavaScript sites
- **AI-Powered Extraction**: ChatGroq model converts raw HTML to structured JSON
- **Manual Backup**: Rule-based extraction when AI fails
- **CSV Export**: Clean, structured output ready for analysis
- **Simple Architecture**: Just 3 files - easy to understand and modify

## 📁 Project Structure

```
observenow/
├── main.py              # Main scraper with AI + manual extraction
├── rag.py              # RAG system with Streamlit UI for Q&A
├── scraper.py           # Web scraping (BeautifulSoup + Selenium)
├── requirements.txt     # Python dependencies
├── urls.csv            # Input URLs (sample provided)
└── README.md           # This file
```

## 🛠️ Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
Create a `.env` file:
```bash
# For main scraper
GROQ_API_KEY=your_groq_api_key_here

# For RAG system  
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your API keys:
- **Groq**: [Groq Console](https://console.groq.com/) (for data extraction)
- **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey) (for Q&A)

### 3. Prepare Input File
Your CSV should have a column named `url`, `URL`, `website`, or `link`:

```csv
url
https://www.indiamart.com/
https://www.tradeindia.com/
https://www.yellowpages.com/
```

## 🚀 Usage

### Option 1: Batch Scraping (main.py)
Extract data from multiple URLs and export to google sheets:

```bash
# Basic usage
python main.py urls.csv


### Option 2: Interactive RAG System (rag.py)
Chat with company data using Streamlit UI:

```bash
# Launch the RAG interface
streamlit run rag.py
```

**RAG Features:**
- 🏢 Enter company name → Auto-generates URL
- 🌐 Or enter direct website URL  
- 🔍 Scrapes & stores data in vector database
- 💬 Ask questions about the company
- 📊 Get instant AI-powered answers

## 📊 What Gets Extracted

The system extracts and structures the following data:

### Basic Information
- **company_name**: Clean company name
- **website**: Official website URL
- **email**: Primary contact email
- **phone**: Primary phone number (standardized)
- **address**: Business address
- **description**: Company description
- **category**: Business category/industry

### Social Media
- **social_facebook**: Facebook URL
- **social_twitter**: Twitter/X URL
- **social_linkedin**: LinkedIn URL
- **social_instagram**: Instagram URL

### Additional Data
- **services**: Products/services offered (list)
- **founding_year**: Year company was founded
- **employee_count**: Number of employees
- **confidence_score**: AI confidence level (0-1)
- **source_url**: Original scraped URL

## 🤖 How It Works

### 1. Web Scraping
- **Primary**: BeautifulSoup with requests (fast, lightweight)
- **Fallback**: Selenium with Chrome (handles JavaScript)

### 2. Data Extraction
- **AI-First**: ChatGroq model processes raw HTML → structured JSON
- **Manual Backup**: Rule-based extraction when AI fails
- **Smart Cleaning**: Automatic data normalization

### 3. Output
- Exports to CSV with all extracted fields
- Includes confidence scores and source URLs
- Ready for spreadsheet analysis

## 🔧 Configuration

### Environment Variables (.env)
```bash
GROQ_API_KEY=your_api_key_here    # Required for AI extraction
DEFAULT_DELAY=1                   # Delay between requests
MAX_RETRIES=3                     # Retry failed requests
TIMEOUT=30                        # Request timeout
```

### Customization
Both `main.py` and `scraper.py` are well-commented and easy to modify:

- **Extraction Rules**: Edit manual extraction methods in `main.py`
- **AI Prompts**: Modify the ChatGroq prompt for different extraction needs
- **Scraping Behavior**: Adjust delays, retries, headers in `scraper.py`

## 📈 Sample Output

```csv
company_name,website,email,phone,address,description,category,social_facebook,confidence_score,source_url
IndiaMART,https://www.indiamart.com,support@indiamart.com,+91-9999999999,"Delhi, India","Online marketplace for business",E-commerce,https://facebook.com/indiamart,0.95,https://www.indiamart.com/
TradeIndia,https://www.tradeindia.com,info@tradeindia.com,+91-8888888888,"Mumbai, India","B2B trading platform",B2B,https://facebook.com/tradeindia,0.90,https://www.tradeindia.com/
```

## 🚨 Error Handling

The system is robust with multiple fallbacks:

1. **Scraping Fails**: Retries with exponential backoff
2. **AI Extraction Fails**: Falls back to manual extraction
3. **Manual Extraction Fails**: Logs error and continues
4. **Invalid Data**: Skips and continues processing

All errors are logged to `logs/scraper_YYYYMMDD_HHMMSS.log`

## 📊 Statistics & Reporting

After completion, see a summary:
```
OBSERVENOW SCRAPING SUMMARY
============================
URLs Processed:           6
Successful Scrapes:        5
Failed Scrapes:           1
Success Rate:             83.3%

AI Extractions:           4
Manual Extractions:       1
```
