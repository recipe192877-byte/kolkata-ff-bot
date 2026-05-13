import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import datetime
import time

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def standardize_date(date_str):
    """Converts 'SUNDAY, 15 MARCH 2026' or similar to '15/03/2026'"""
    # Clean up common extraneous characters
    date_str = date_str.replace('"', '').replace('(', '').replace(')', '').strip()
    
    # If it's already in the target format, return it
    if re.match(r'^\d{2}/\d{2}/\d{4}$', date_str):
        return date_str
        
    try:
        # Extract the date part, handling cases with/without day of the week
        # Use regex to find and extract the most likely date part
        match = re.search(r'(\d+)\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\s+(\d{4})', date_str, re.IGNORECASE)
        if match:
            # Reconstruct the string for strptime
            date_part = f"{match.group(1)} {match.group(2)}" # Day and Year
            # Find the month name from the original string
            month_match = re.search(r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*', date_str, re.IGNORECASE)
            if month_match:
                date_part = f"{match.group(1)} {month_match.group(0)} {match.group(2)}"
        else:
            # Fallback to previous logic if regex fails
            if ',' in date_str:
                date_part = date_str.split(',', 1)[1].strip() # Split only on first comma
            else:
                date_part = date_str
            
            # Remove non-alphanumeric chars (except spaces) and ordinal suffixes (1ST, 2ND, 3RD, 4TH etc.)
            date_part = re.sub(r'[^\w\s]', '', date_part).strip()
            date_part = re.sub(r'(\d+)(?:ST|ND|RD|TH)', r'\1', date_part, flags=re.IGNORECASE)
        
        # Try parsing with various common date formats
        formats = [
            '%d %B %Y',  # 15 March 2026
            '%d %b %Y',  # 15 Mar 2026
            '%Y-%m-%d',  # 2026-03-15
            '%m/%d/%Y'   # 03/15/2026
        ]
        
        for fmt in formats:
            try:
                dt_obj = datetime.datetime.strptime(date_part, fmt)
                return dt_obj.strftime('%d/%m/%Y')
            except ValueError:
                continue # Try next format
        
        # If no format matches, raise an error to be caught by the outer try-except
        raise ValueError(f"No matching date format found for '{date_part}'")
        
    except Exception as e:
        print(f"Date parse error for '{date_str}': {e}")
        
    return date_str # Return original if all parsing fails


def _fetch_with_retry(url, headers, retries=MAX_RETRIES):
    """Fetch URL with retry logic and exponential backoff.
    Adds random jitter to retry delay and logs HTTP status codes.
    """
    import random 
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            return response
        except requests.exceptions.RequestException as e: # Catch all requests-related exceptions
            print(f"  Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                wait = RETRY_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1) # Exponential backoff with jitter
                print(f"  Retrying in {wait:.2f}s...")
                time.sleep(wait)
    return None


def scrape_kolkata_ff_in():
    url = "https://kolkataff.in/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}
    all_data = []

    try:
        response = _fetch_with_retry(url, headers)
        if not response:
            print("kolkataff.in: All retries failed.")
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 3: # Need at least 3 rows: date, patti, single
                continue
                
            # Date extraction from the first row's text
            date_col_raw = rows[0].get_text(strip=True)
            if not date_col_raw: # Skip tables without a date header
                continue
            
            # The date format on kolkataff.in might be simple (e.g., "15 MARCH 2026")
            # Try to standardize it using the existing function
            date_parsed = standardize_date(date_col_raw)
            if not re.match(r'^\d{2}/\d{2}/\d{4}$', date_parsed): # If standardization failed
                print(f"  kolkataff.in: Could not parse date from '{date_col_raw}', skipping table.")
                continue

            # Determine patti and single row indices
            # Check if there's a header indicating '1234' like bazi numbering
            has_bazi_header = False
            if len(rows) >= 2:
                header_text = rows[1].get_text(strip=True).replace(' ', '').upper()
                if '1234' in header_text or 'BAZI' in header_text or 'बाजि' in header_text: # Added Bengali "bazi"
                    has_bazi_header = True
            
            if has_bazi_header and len(rows) >= 4:
                patti_row_idx = 2
                single_row_idx = 3
            elif len(rows) >= 3:
                patti_row_idx = 1
                single_row_idx = 2
            else:
                continue # Not enough rows to parse results
            
            if len(rows) <= single_row_idx: # Ensure single row exists
                continue
            
            # Extract cells from patti and single rows
            pattis_cols = rows[patti_row_idx].find_all(['td', 'th'])
            singles_cols = rows[single_row_idx].find_all(['td', 'th'])

            # Assuming 8 bazis max (Kolkata FF standard)
            num_bazis = min(len(pattis_cols), len(singles_cols), 8)
            
            for bazi_idx in range(num_bazis):
                try:
                    p = pattis_cols[bazi_idx].get_text(strip=True)
                    s = singles_cols[bazi_idx].get_text(strip=True)
                except IndexError:
                    continue # Should not happen with min(..., 8) but as a safeguard

                # Clean and validate extracted values
                # Filter out obvious non-result cells like 'Tips', empty strings
                if not p or not s or 'Tips' in p or 'Tips' in s or p == '-' or s == '-':
                    continue
                
                # Extract numeric values
                p_val = re.sub(r'\D', '', p)
                s_val = re.sub(r'\D', '', s)

                if p_val and s_val:
                    # Basic validation for patti (3 digits) and single (1 digit) if possible
                    if len(p_val) == 3 and len(s_val) == 1:
                        # Combine to form the full result string (patti + single)
                        res = p_val + s_val 
                        all_data.append({
                            'Date': date_parsed, 'Bazi': bazi_idx + 1,
                            'Result_String': res, 'Patti': p_val, 'Single': s_val,
                            'Source': 'kolkataff.in'
                        })
                    else:
                        print(f"  kolkataff.in: Skipping malformed result - Patti: '{p_val}', Single: '{s_val}' for Bazi {bazi_idx + 1} on {date_parsed}")

        return all_data
    except Exception as e:
        print(f"Error scraping fallback kolkataff.in: {e}")
        return []


def scrape_kolkata_ff():
    url = "https://kolkataff.tv/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/', # Mimic a referrer
        'DNT': '1', # Do Not Track
        'Connection': 'keep-alive'
    }
    
    all_data = []
    scrape_ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        print(f"[{scrape_ts}] Fetching data from kolkataff.tv...")
        response = _fetch_with_retry(url, headers)
        if not response:
            print("kolkataff.tv: All retries failed.")
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) < 2: # Need at least 2 rows: date and results
                    continue
                    
                date_col = rows[0].get_text(strip=True)
                # Filter out header rows that are not actual dates
                if "Result Time" in date_col or "Time" in date_col or not date_col:
                    continue
                    
                date_col = standardize_date(date_col)
                # If date standardization fails, skip this table
                if not re.match(r'^\d{2}/\d{2}/\d{4}$', date_col):
                    print(f"  kolkataff.tv: Could not parse date from '{date_col}', skipping table.")
                    continue
                
                cols = rows[1].find_all(['td', 'th'])
                bazi_results = [c.get_text(strip=True) for c in cols]
                
                # Process up to 8 bazis
                for bazi_idx, result_str in enumerate(bazi_results[:8]):
                    bazi_num = bazi_idx + 1
                    
                    # Clean and validate potential result string
                    result_str_cleaned = result_str.strip()
                    if result_str_cleaned in ('--', '', '-', 'Refresh', 'Tips'):
                        patti, single = None, None
                        result_string_to_store = "" # Store empty string if no valid result
                    else:
                        # Extract digits using regex
                        digits = re.sub(r'\D', '', result_str_cleaned) # Remove all non-digits
                        
                        patti, single = None, None
                        result_string_to_store = result_str_cleaned # Default to original cleaned string
                        
                        if len(digits) == 4:
                            patti = digits[:3]
                            single = digits[3]
                            result_string_to_store = digits # Store just digits for 4-digit results
                        elif len(digits) == 1:
                            single = digits
                            result_string_to_store = digits # Store just digit for 1-digit results
                        # Cases like "123" or "12" or >4 digits are ambiguous or malformed on this site for P/S
                        # We prioritize 4-digit (Patti+Single) or 1-digit (Single only)
                        
                    all_data.append({
                        'Date': date_col, 'Bazi': bazi_num,
                        'Result_String': result_string_to_store, # Store processed string
                        'Patti': patti, 'Single': single,
                        'Source': 'kolkataff.tv'
                    })
    except Exception as e:
        print(f"Error scraping data from kolkataff.tv: {e}")
        
    try:
        print(f"[{scrape_ts}] Fetching fallback from kolkataff.in...")
        fallback_data = scrape_kolkata_ff_in()
        if fallback_data:
            all_data.extend(fallback_data)
    except Exception as e:
        print(f"Error scraping fallback data: {e}")
            
    if not all_data:
        print("Both sources failed to return data.")
        return None
            
    try:
        df_new = pd.DataFrame(all_data)
        
        # Ensure 'Date' column is in the correct format for sorting and comparison
        df_new['Date'] = df_new['Date'].apply(lambda x: standardize_date(str(x)))
        df_new = df_new[df_new['Date'].str.match(r'^\d{2}/\d{2}/\d{4}$', na=False)] # Filter out invalid dates
        
        # FIX #3: Better dedup — prefer records with both Patti AND Single (most complete)
        # Add completeness score: records with both patti+single rank higher
        df_new['_completeness'] = df_new.apply(
            lambda r: (2 if pd.notna(r.get('Patti')) and str(r.get('Patti', '')).strip() != '' else 0) +
                      (1 if pd.notna(r.get('Single')) and str(r.get('Single', '')).strip() != '' else 0),
            axis=1
        )
        # Sort by completeness (highest first) so drop_duplicates keeps the best
        df_new = df_new.sort_values(['Date', 'Bazi', '_completeness', 'Source'], ascending=[True, True, False, False]) # Prefer .tv over .in if completeness is same
        df_new = df_new.drop_duplicates(subset=['Date', 'Bazi'], keep='first')
        df_new = df_new.drop(columns=['_completeness'], errors='ignore')
        
        csv_filename = 'kolkata_ff_history_advanced.csv'
        
        try:
            df_old = pd.read_csv(csv_filename, dtype={'Patti': str, 'Single': str}) # Read Patti/Single as string
            # Handle potential empty CSV gracefully
            if df_old.empty:
                print("Existing CSV is empty. Initializing with new data.")
                df = df_new
            else:
                combined = pd.concat([df_old, df_new], ignore_index=True)
                
                # Ensure 'Date' column is in the correct format for sorting
                combined['Date'] = combined['Date'].apply(lambda x: standardize_date(str(x)))
                
                # Same completeness-based dedup for the combined dataset
                combined['_completeness'] = combined.apply(
                    lambda r: (2 if pd.notna(r.get('Patti')) and str(r.get('Patti', '')).strip() != '' else 0) +
                              (1 if pd.notna(r.get('Single')) and str(r.get('Single', '')).strip() != '' else 0),
                    axis=1
                )
                
                # Sort first by date and bazi to group, then by completeness to prioritize
                combined = combined.sort_values(['Date', 'Bazi', '_completeness', 'Source'], ascending=[True, True, False, False]) # Prefer .tv over .in
                combined = combined.drop_duplicates(subset=['Date', 'Bazi'], keep='first')
                combined = combined.drop(columns=['_completeness'], errors='ignore')
                
                # Final sort by date and bazi for chronological order
                combined['Date_Obj'] = pd.to_datetime(combined['Date'], format='%d/%m/%Y', errors='coerce')
                combined = combined.sort_values(by=['Date_Obj', 'Bazi'], ascending=[True, True]).drop(columns=['Date_Obj'])
                
                df = combined
            
            df.to_csv(csv_filename, index=False)
            print(f"[{scrape_ts}] Scraped {len(df_new)} new records. Total in DB: {len(df)}.")
        except FileNotFoundError:
            df_new.to_csv(csv_filename, index=False)
            print(f"[{scrape_ts}] Created new database. Scraped {len(df_new)} records.")
            df = df_new
        
        try:
            import github_sync
            from threading import Thread
            sync_thread = Thread(target=github_sync.upload_to_github, daemon=True)
            sync_thread.start()
            sync_thread.join(timeout=30)  # Don't block scraper for more than 30s
            if sync_thread.is_alive():
                print("GitHub sync timed out after 30s, continuing...")
        except ImportError:
            pass # github_sync.py might not exist or be needed
        except Exception as e:
            print(f"GitHub sync error: {e}")
            
        return df
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    scrape_kolkata_ff()