import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import warnings
import datetime
import os
from typing import Set, List, Dict, Tuple, Any
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import PyPDF2
from io import BytesIO
import pandas as pd
import random
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
import re
import pickle
from tqdm import tqdm
import zipfile
import camelot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Adjusted parameters for Colab stability
MAX_WORKERS = 30  # Reduced from 100 for stability
PDF_TIMEOUT = 30
HTML_TIMEOUT = 20
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.3
MAX_CONTENT_SIZE = 20 * 1024 * 1024  # 20MB
RATE_LIMIT_DELAY = 1.5  # Increased delay for a gentler pace
MAX_URLS = 5000
DOMAIN_DELAYS = {}

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36',
]

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

def create_session() -> requests.Session:
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

SESSION = create_session()

def load_existing_links(csv_file: str) -> Set[str]:
    """load existing links from csv link file with deduplication"""
    existing = set()
    if os.path.exists(csv_file):
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].strip():
                        existing.add(row[0].strip())
            logger.info(f"Loaded {len(existing)} existing links from {csv_file}")
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
    return existing

def domain_key(url: str) -> str:
    """Extract domain key for rate limiting"""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

def respect_domain_delay(url: str):
    key = domain_key(url)
    last_request = DOMAIN_DELAYS.get(key, 0)
    elapsed = time.time() - last_request
    if elapsed < RATE_LIMIT_DELAY:
        sleep_time = RATE_LIMIT_DELAY - elapsed
        time.sleep(sleep_time)
    DOMAIN_DELAYS[key] = time.time()

def extract_metadata(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract comprehensive metadata from HTML"""
    metadata = {}
    metadata["title"] = getattr(soup.title, "string", "").strip() if soup.title else ""
    charset = soup.find("meta", charset=True)
    metadata["charset"] = charset["charset"] if charset else ""
    html_tag = soup.find("html")
    metadata["language"] = html_tag.get("lang", "") if html_tag else ""
    desc_tag = soup.find("meta", attrs={"name": "description"})
    keywords_tag = soup.find("meta", attrs={"name": "keywords"})
    metadata["description"] = desc_tag.get("content", "").strip() if desc_tag else ""
    metadata["keywords"] = keywords_tag.get("content", "").strip() if keywords_tag else ""
    return metadata

def extract_valid_links(soup: BeautifulSoup, base_url: str) -> Set[str]:
    """Extract valid links with sophisticated filtering"""
    links = set()
    blocked_extensions = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".docx", ".js",
                          ".css", ".mp3", ".mp4", ".avi", ".mov", ".webm", ".ogg", ".wav",
                          ".ico", ".exe", ".bin", ".tar", ".gz", ".bz2",".txt")

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if (
            not href or
            "#" in href or
            href.startswith("javascript:") or
            href.startswith("mailto:") or
            href.startswith("tel:") or
            "forms.gle" in href or
            "docs.google.com/forms" in href
        ):
            continue

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        # filter by domain
        if "iiti.ac.in" not in parsed.netloc and "iiti.ac.in" not in parsed.path:
            continue

        #filter by file extension
        if any(full_url.lower().endswith(ext) for ext in blocked_extensions):
            continue

        #filter by filetype patterns
        if any(pattern in full_url.lower() for pattern in ["/cdn-cgi/", "/wp-admin/", "/wp-json/"]):
            continue

        # Remove fragments and query parameters
        clean_url = parsed._replace(fragment="", query="").geturl()

        links.add(clean_url)

    return links

def get_clean_body_text(soup: BeautifulSoup) -> str:
    """Extract clean body text with context preservation"""
    body = soup.find("body")
    if not body:
        return ""

    #remove unwanted things
    for tag in body(["script", "style", "noscript", "iframe", "svg",
                     "video", "audio", "footer", "nav", "aside", "form"]):
        tag.decompose()

    text = body.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_pdf_text_from_url(url: str) -> str:
    """
    Extracts text and tables from a PDF, falling back to PyPDF2 for non-tabular content.
    """
    try:
        respect_domain_delay(url)
        headers = {"User-Agent": random.choice(USER_AGENTS)}

        head_resp = SESSION.head(url, headers=headers, timeout=5, verify=False)
        content_type = head_resp.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type:
            logger.warning(f"URL {url} claims to be PDF but has Content-Type: {content_type}")
            return ""

        resp = SESSION.get(url, headers=headers, timeout=PDF_TIMEOUT, verify=False, stream=True)
        resp.raise_for_status()

        content = b""
        for chunk in resp.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > MAX_CONTENT_SIZE:
                logger.warning(f"PDF too large ({len(content)} bytes), skipping: {url}")
                return ""
            if not chunk:
                break

        full_text = ""
        with BytesIO(content) as pdf_file:
            try:
                # Use Camelot to find and extract tables
                tables = camelot.read_pdf(
                    pdf_file,
                    pages='all',
                    flavor='stream', # 'stream' is faster and more resource-friendly than 'lattice'
                    strip_text='\n\t'
                )

                if tables:
                    for table in tables:
                        # Convert table DataFrame to a string representation (e.g., CSV)
                        table_string = table.df.to_csv(index=False, header=True, sep='|')
                        full_text += f"\n\n---START OF TABLE---\n{table_string}\n---END OF TABLE---\n\n"
                    logger.info(f"Successfully extracted {len(tables)} tables from PDF: {url}")
            except Exception as e:
                logger.warning(f"Camelot table extraction failed for {url}: {str(e)}. Falling back to standard text extraction.")

            # Use PyPDF2 to get the rest of the text, including from image-based pages
            try:
                pdf_file.seek(0)
                reader = PyPDF2.PdfReader(pdf_file)
                other_text = " ".join(
                    page.extract_text().replace('\n', ' ').strip()
                    for page in reader.pages
                    if page.extract_text()
                )
                full_text += other_text
            except Exception as e:
                logger.error(f"PyPDF2 text extraction failed for {url}: {str(e)}")
                return ""

        return re.sub(r'\s+', ' ', full_text).strip()

    except Exception as e:
        logger.error(f"PDF extraction failed for {url}: {str(e)}")
        return ""

def scrape_single_url(url: str) -> Dict[str, Any]:
    """Scrape single URL with comprehensive error handling"""
    result = {
        "url": url,
        "title": "",
        "keywords": "",
        "description": "",
        "body_text": "",
        "new_links": set(),
        "error": "",
        "status": "success"
    }

    try:
        respect_domain_delay(url)
        head_resp = SESSION.head(url, timeout=5, allow_redirects=True)
        if head_resp.status_code == 404:
            result["error"] = "404 Dead Link"
            result["status"] = "dead"
            return result

        if url.lower().endswith(".pdf"):
            result["body_text"] = extract_pdf_text_from_url(url)
            filename = url.split("/")[-1]
            clean_filename = filename.split('?')[0].split('#')[0]
            result["title"] = clean_filename.rsplit('.', 1)[0] if '.' in clean_filename else clean_filename
            return result

        headers = {"User-Agent": random.choice(USER_AGENTS)}
        resp = SESSION.get(url, headers=headers, timeout=HTML_TIMEOUT, verify=False)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        if "html" not in content_type:
            result["error"] = f"Non-HTML content: {content_type}"
            result["status"] = "invalid_content"
            return result

        soup = BeautifulSoup(resp.content, "html.parser")
        metadata = extract_metadata(soup)
        h1 = soup.find("h1")
        result["title"] = h1.get_text(strip=True) if h1 else metadata.get("title", "")
        result["keywords"] = metadata.get("keywords", "")
        result["description"] = metadata.get("description", "")
        result["body_text"] = get_clean_body_text(soup)
        result["new_links"] = extract_valid_links(soup, url)

    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        result["error"] = str(e)
        result["status"] = "failed"

    return result

def scrape_urls_parallel(urls: List[str]) -> Tuple[List[Dict], Set[str]]:
    """Parallel scraping with intelligent task management"""
    results = []
    new_links = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scrape_single_url, url): url for url in urls}
        for future in tqdm(as_completed(futures), total=len(urls), desc="Scraping URLs"):
            url = futures[future]
            try:
                result = future.result()
                results.append(result)
                new_links.update(result["new_links"])
            except Exception as e:
                logger.error(f"Processing failed for {url}: {str(e)}")
                results.append({
                    "url": url,
                    "title": "",
                    "keywords": "",
                    "description": "",
                    "body_text": "",
                    "new_links": set(),
                    "error": str(e),
                    "status": "failed"
                })

    return results, new_links

def run_until_all_scraped(input_csv: str, output_csv: str):
    """Main scraping orchestration with state persistence and URL limit"""
    if not os.path.exists(input_csv):
        with open(input_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["url"])
        logger.info(f"Created input file: {input_csv}")

    if not os.path.exists(output_csv):
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["row_id", "url", "title", "keywords", "description", "body_text", "error", "status"])

    processed_urls = set()
    if os.path.exists(output_csv):
        try:
            for chunk in pd.read_csv(output_csv, usecols=['url'], chunksize=10000):
                processed_urls.update(chunk['url'].dropna().astype(str).unique())
            logger.info(f"Loaded {len(processed_urls)} processed URLs from output")
        except Exception as e:
            logger.error(f"Error loading processed URLs: {e}")

    state_file = "scraper_state.pkl"
    if os.path.exists(state_file):
        try:
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
                processed_urls.update(state.get('processed_urls', set()))
            logger.info(f"Loaded {len(state.get('processed_urls', set()))} URLs from state file")
        except Exception:
            logger.warning("Failed to load state file, starting fresh")

    run_count = 0
    total_processed = 0
    total_new_links = 0

    while True:
        run_count += 1
        start_time = time.time()

        if total_processed >= MAX_URLS:
            logger.info(f"🚫 Maximum URL limit reached ({MAX_URLS}). Stopping scraper.")
            break

        all_urls = load_existing_links(input_csv)
        urls_to_process = [url for url in all_urls if url not in processed_urls]

        remaining = MAX_URLS - total_processed
        if remaining <= 0:
            logger.info(f"🚫 Reached maximum URL limit ({MAX_URLS}). Stopping.")
            break

        if len(urls_to_process) > remaining:
            logger.info(f"⚠ Batch size {len(urls_to_process)} exceeds remaining capacity {remaining}. Processing partial batch.")
            urls_to_process = urls_to_process[:remaining]

        logger.info(f"\n{'='*50}")
        logger.info(f"🏁 STARTING RUN #{run_count}")
        logger.info(f"📊 URLs in queue: {len(all_urls)} | To process: {len(urls_to_process)}")
        logger.info(f"📈 Total processed: {total_processed}/{MAX_URLS}")
        logger.info(f"{'='*50}")

        if not urls_to_process:
            logger.info("✅ All URLs processed. Exiting.")
            break

        results, new_links = scrape_urls_parallel(urls_to_process)
        num_processed = len(urls_to_process)
        num_new_links = len(new_links)
        total_processed += num_processed
        total_new_links += num_new_links

        with open(output_csv, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            current_row_id = len(processed_urls)
            for idx, result in enumerate(results):
                def safe_str(s):
                    return s.encode('utf-8', 'replace').decode('utf-8') if isinstance(s, str) else s

                writer.writerow([
                    current_row_id + idx + 1,
                    safe_str(result['url']),
                    safe_str(result['title']),
                    safe_str(result['keywords']),
                    safe_str(result['description']),
                    safe_str(result['body_text']),
                    safe_str(result['error']),
                    safe_str(result['status'])
                ])

        processed_urls.update(urls_to_process)

        if total_processed < MAX_URLS:
            if new_links:
                existing = load_existing_links(input_csv)
                truly_new = new_links - existing
                if truly_new:
                    with open(input_csv, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        for link in truly_new:
                            writer.writerow([link])
                    logger.info(f"🔍 Added {len(truly_new)} new links to queue")
        else:
            logger.info("🚫 Skipping adding new links - reached URL limit")

        try:
            with open(state_file, 'wb') as f:
                pickle.dump({
                    'processed_urls': processed_urls,
                    'run_count': run_count,
                    'total_processed': total_processed
                }, f)
            logger.info("💾 Saved scraper state")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

        duration = time.time() - start_time
        logger.info(f"⏱ Run #{run_count} completed in {duration:.1f}s")
        logger.info(f"📊 Processed: {num_processed} | New links: {num_new_links}")
        logger.info(f"📈 Total processed: {total_processed}/{MAX_URLS} | Total new links: {total_new_links}")

        if total_processed >= MAX_URLS:
            logger.info(f"🚫 Reached maximum URL limit ({MAX_URLS}). Stopping scraper.")
            break

    try:
        os.remove(state_file)
        logger.info("🧹 Cleaned up state file")
    except Exception:
        pass

    logger.info(f"\n{'='*50}")
    logger.info(f"🏁 SCRAPING COMPLETE")
    logger.info(f"• Total runs: {run_count}")
    logger.info(f"• Total URLs processed: {total_processed}")
    logger.info(f"• Total unique links discovered: {total_new_links}")
    logger.info(f"{'='*50}")

def process_zip_file(zip_path: str, destination_dir: str):
    """
    Extracts all files from a ZIP archive and saves them to a destination directory.
    """
    if not os.path.exists(zip_path):
        print(f"Error: ZIP file not found at {zip_path}")
        return

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.filename.startswith('__MACOSX/'):
                zip_ref.extract(file_info, destination_dir)
                print(f"Extracted: {file_info.filename}")

def main():
    """
    Main function to orchestrate the entire data pipeline using pandas.
    """
    web_input_csv = "cleaned_urls_list.csv"
    web_output_csv = "output_data.csv"
    local_zip_path = "EMAIL_PDFs.zip"
    local_data_dir = "local_data"
    final_output_csv = "final_integrated_data.csv"

    print("Starting web crawling process...")
    run_until_all_scraped(web_input_csv, web_output_csv)
    print("Web crawling completed. Data saved to", web_output_csv)

    print("Extracting local files from ZIP...")
    process_zip_file(local_zip_path, local_data_dir)

    print("Starting pandas data integration...")

    if os.path.exists(web_output_csv):
        web_df = pd.read_csv(web_output_csv)
        web_df['source'] = 'web_page'
        print(f"Loaded {len(web_df)} records from web scraping.")
    else:
        print("Warning: Web output file not found. Creating empty DataFrame.")
        web_df = pd.DataFrame(columns=['row_id', 'url', 'title', 'keywords', 'description', 'body_text', 'error', 'status', 'source'])

    local_data = []
    if os.path.exists(local_data_dir):
        for filename in os.listdir(local_data_dir):
            file_path = os.path.join(local_data_dir, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        local_data.append({
                            'filename': filename,
                            'content': content,
                            'source': 'local_file'
                        })
                except Exception as e:
                    print(f"Error reading local file {filename}: {e}")
        local_df = pd.DataFrame(local_data)
        print(f"Loaded {len(local_df)} local files.")
    else:
        print("Warning: Local data directory not found. Creating empty DataFrame.")
        local_df = pd.DataFrame(columns=['filename', 'content', 'source'])

    web_df = web_df.rename(columns={'title': 'filename', 'body_text': 'content'})
    web_df_cleaned = web_df[['filename', 'content', 'source']].copy()
    final_df = pd.concat([local_df, web_df_cleaned], ignore_index=True)
    final_df.dropna(subset=['content'], inplace=True)
    final_df.drop_duplicates(subset=['content'], inplace=True)
    final_df.to_csv(final_output_csv, index=False)

    print(f"✅ Integration complete! Final combined data saved to {final_output_csv} with {len(final_df)} records.")

if __name__ == "__main__":
    main()