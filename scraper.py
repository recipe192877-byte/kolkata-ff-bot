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
    # Clean the input string first
    clean_date_str = date_str.replace('"', '').replace('(', '').replace(')', '').strip()

    # Check if it's already in the target format 'DD/MM/YYYY'
    if re.fullmatch(r'\d{2}/\d{2}/\d{4}', clean_date_str):
        return clean_date_str

    try:
        # Extract date part if comma is present (e.g., "SUNDAY, 15 MARCH 2026")
        if ',' in clean_date_str:
            date_part = clean_date_str.split(',', 1)[1].strip()
        else:
            date_part = clean_date_str.strip()

        # Remove common non-alphanumeric characters except space, and ordinal suffixes
        # This regex is more robust to variations like "15th March" or similar.
        date_part = re.sub(r'(?i)(\d+)(st|nd|rd|th)', r'\1', date_part).strip() # remove ordinal suffixes
        date_part = re.sub(r'[^a-zA-Z0-9\s]', '', date_part).strip() # remove other non-alphanumeric chars

        # Try parsing with various common formats
        formats = [
            '%d %B %Y',  # 15 March 2026
            '%d %b %Y',  # 15 Mar 2026
            '%d-%m-%Y',  # 15-03-2026
            '%Y-%m-%d',  # 2026-03-15
            '%d/%m/%Y'   # 15/03/2026 (should be caught by initial fullmatch, but good for robustness)
        ]
        for fmt in formats:
            try:
                dt_obj = datetime.datetime.strptime(date_part, fmt)
                return dt_obj.strftime('%d/%m/%Y')
            except ValueError:
                continue # Try next format

    except Exception as e:
        # Catch any unexpected errors during processing
        pass # Will return original or fall through if no format matches

    # If parsing fails, return the original (cleaned) string, indicating failure
    print(f"Date parse error for standardizing '{date_str}'. Returning original: '{clean_date_str}'")
    return clean_date_str


def _fetch_with_retry(url, headers, retries=MAX_RETRIES):
    """Fetch URL with retry logic and exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e: # Catch specific request exceptions
            print(f"  Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                wait = RETRY_DELAY * (2 ** (attempt - 1)) # Exponential backoff
                print(f"  Retrying in {wait:.2f}s...")
                time.sleep(wait)
    print(f"All {retries} attempts failed for {url}.")
    return None


def scrape_kolkata_ff_in():
    url = "https://kolkataff.in/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    all_data = []
    try:
        response = _fetch_with_retry(url, headers)
        if not response:
            print("kolkataff.in: All retries failed. Returning empty list.")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 3: # Need at least date, headers (or patti), and singles row
                continue

            # Extract date from the first row, which often contains date information
            date_col_raw = rows[0].get_text(strip=True)
            date_parsed = standardize_date(date_col_raw)
            if not re.fullmatch(r'\d{2}/\d{2}/\d{4}', date_parsed):
                # If date isn't properly parsed, it's likely not a result table
                continue

            # Determine row structure: Some tables have '1234' on the 2nd row (index 1), some don't.
            # '1234' indicates column headers for bazi numbers.
            patti_row_idx = -1
            single_row_idx = -1

            if len(rows) >= 2 and '1234' in rows[1].get_text(strip=True).replace(' ', ''):
                # Structure: Date, Bazi_Numbers, Patti, Single
                # The actual result rows start from index 2
                if len(rows) >= 3: patti_row_idx = 2
                if len(rows) >= 4: single_row_idx = 3
            else:
                # Structure: Date, Patti, Single (or similar, without explicit Bazi_Numbers row)
                # The actual result rows start from index 1 (or 2 if date is multiline)
                if len(rows) >= 2: patti_row_idx = 1
                if len(rows) >= 3: single_row_idx = 2

            if patti_row_idx == -1 or single_row_idx == -1 or len(rows) <= single_row_idx:
                continue # Not enough rows for results

            pattis_cols = [c.get_text(strip=True) for c in rows[patti_row_idx].find_all(['td', 'th'])]
            singles_cols = [c.get_text(strip=True) for c in rows[single_row_idx].find_all(['td', 'th'])]

            num_bazis = min(len(pattis_cols), len(singles_cols), 8) # Max 8 bazis per day

            for bazi_idx in range(num_bazis):
                try:
                    p = pattis_cols[bazi_idx]
                    s = singles_cols[bazi_idx]
                except IndexError:
                    continue # Should not happen if num_bazis is calculated correctly, but for safety

                # Basic validation and cleaning
                if not p or not s or 'Tips' in p or 'Tips' in s or p == '' or s == '':
                    continue

                p_val = re.sub(r'\D', '', p)
                s_val = re.sub(r'\D', '', s)

                if p_val and s_val:
                    # Construct result string as 'patti' + 'single'
                    result_string = p_val + s_val
                    all_data.append({
                        'Date': date_parsed, 'Bazi': bazi_idx + 1,
                        'Result_String': result_string, 'Patti': p_val, 'Single': s_val,
                        'Source': 'kolkataff.in'
                    })
        return all_data
    except Exception as e:
        print(f"Error scraping kolkataff.in: {e}")
        return []


def scrape_kolkata_ff():
    url = "https://kolkataff.tv/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    all_data = [] # To accumulate data from both sources
    scrape_ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # --- kolkataff.tv scraping ---
    print(f"[{scrape_ts}] Fetching data from kolkataff.tv...")
    try:
        response = _fetch_with_retry(url, headers)
        if not response:
            print("kolkataff.tv: All retries failed. Attempting fallback.")
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')

            for table in tables:
                rows = table.find_all('tr')
                if len(rows) < 2: # Need at least a date row and a results row
                    continue

                date_col_raw = rows[0].get_text(strip=True)
                if "Result Time" in date_col_raw or "Time" in date_col_raw: # Skip header rows
                    continue

                date_parsed = standardize_date(date_col_raw)
                if not re.fullmatch(r'\d{2}/\d{2}/\d{4}', date_parsed):
                     # If date isn't properly parsed, this might not be a result table
                     continue

                cols = rows[1].find_all(['td', 'th']) # Results are typically in the second row
                bazi_results = [c.get_text(strip=True) for c in cols]

                for bazi_idx, result_str in enumerate(bazi_results):
                    # Only process up to 8 bazis
                    if bazi_idx >= 8:
                        break

                    bazi_num = bazi_idx + 1

                    # Clean and parse the result string
                    cleaned_result_str = result_str.strip().replace(' ', '')
                    if not cleaned_result_str or cleaned_result_str in ['--', '-', 'Refresh', 'Tips']:
                        patti, single = None, None
                        final_result_str = result_str # Keep original for Result_String if no parse
                    else:
                        final_result_str = cleaned_result_str
                        # Try to extract numbers
                        match = re.search(r'(\d+)', cleaned_result_str)
                        if match:
                            digits = match.group(1)
                            if len(digits) == 4:
                                patti = digits[:3]
                                single = digits[3]
                            elif len(digits) == 1:
                                patti = None
                                single = digits
                            elif len(digits) >= 2 and len(digits) <= 3: # Handle cases like '123' or '23'
                                patti = digits
                                single = None
                            else: # e.g., numbers like '12345' or nothing parsable
                                patti, single = None, None
                        else:
                            patti, single = None, None

                    all_data.append({
                        'Date': date_parsed, 'Bazi': bazi_num,
                        'Result_String': final_result_str, 'Patti': patti, 'Single': single,
                        'Source': 'kolkataff.tv'
                    })
    except Exception as e:
        print(f"Error scraping data from kolkataff.tv: {e}")

    # --- kolkataff.in (fallback) scraping ---
    print(f"[{scrape_ts}] Fetching fallback from kolkataff.in...")
    try:
        fallback_data = scrape_kolkata_ff_in()
        if fallback_data:
            all_data.extend(fallback_data)
        else:
            print("kolkataff.in fallback returned no data.")
    except Exception as e:
        print(f"Error scraping fallback data from kolkataff.in: {e}")

    if not all_data:
        print("Both sources failed to return any data.")
        return None

    try:
        df_new = pd.DataFrame(all_data)

        # Enhance completeness metric for deduplication
        # A record is more complete if it has both Patti and Single, then just Patti, then just Single.
        df_new['_completeness'] = df_new.apply(
            lambda r: (2 if pd.notna(r.get('Patti')) and str(r.get('Patti', '')).strip() != '' and
                           pd.notna(r.get('Single')) and str(r.get('Single', '')).strip() != '' else
                      (1 if pd.notna(r.get('Patti')) and str(r.get('Patti', '')).strip() != '' else
                      (1 if pd.notna(r.get('Single')) and str(r.get('Single', '')).strip() != '' else 0))),
            axis=1
        )

        # Sort by completeness (highest first) to ensure drop_duplicates keeps the most complete record
        # Also sort by source preference: kolkataff.tv is primary
        df_new['Source_Rank'] = df_new['Source'].apply(lambda x: 1 if x == 'kolkataff.tv' else 2)
        df_new = df_new.sort_values(by=['Date', 'Bazi', '_completeness', 'Source_Rank'],
                                     ascending=[True, True, False, True]) # Higher completeness, lower source_rank
        df_new = df_new.drop_duplicates(subset=['Date', 'Bazi'], keep='first')
        df_new = df_new.drop(columns=['_completeness', 'Source_Rank'], errors='ignore')

        csv_filename = 'kolkata_ff_history_advanced.csv'

        try:
            df_old = pd.read_csv(csv_filename, dtype={'Patti': str, 'Single': str}) # Ensure numeric parsing doesn't drop leading zeros for str columns
            combined = pd.concat([df_old, df_new])

            # Apply the same enhanced completeness-based deduplication for the combined dataset
            combined['_completeness'] = combined.apply(
                lambda r: (2 if pd.notna(r.get('Patti')) and str(r.get('Patti', '')).strip() != '' and
                               pd.notna(r.get('Single')) and str(r.get('Single', '')).strip() != '' else
                          (1 if pd.notna(r.get('Patti')) and str(r.get('Patti', '')).strip() != '' else
                          (1 if pd.notna(r.get('Single')) and str(r.get('Single', '')).strip() != '' else 0))),
                axis=1
            )
            combined['Source_Rank'] = combined['Source'].apply(lambda x: 1 if x == 'kolkataff.tv' else 2)

            combined = combined.sort_values(by=['Date', 'Bazi', '_completeness', 'Source_Rank'],
                                             ascending=[True, True, False, True])
            combined = combined.drop_duplicates(subset=['Date', 'Bazi'], keep='first')
            combined = combined.drop(columns=['_completeness', 'Source_Rank'], errors='ignore')

            # Ensure 'Date' column is in the correct format before conversion to datetime object
            # This handles cases where 'standardize_date' might have failed for some old records
            combined['Date_Obj'] = pd.to_datetime(combined['Date'], format='%d/%m/%Y', errors='coerce')
            combined = combined.dropna(subset=['Date_Obj']) # Drop rows where date couldn't be parsed
            combined = combined.sort_values(by=['Date_Obj', 'Bazi'], ascending=[True, True]).drop(columns=['Date_Obj'])

            combined.to_csv(csv_filename, index=False)
            print(f"[{scrape_ts}] Scraped {len(df_new)} new records. Total in DB: {len(combined)}.")
            df = combined
        except FileNotFoundError:
            # If CSV doesn't exist, create it with the new data
            df_new = df_new.drop(columns=['_completeness', 'Source_Rank'], errors='ignore') # Drop temp cols before saving
            df_new['Date_Obj'] = pd.to_datetime(df_new['Date'], format='%d/%m/%Y', errors='coerce')
            df_new = df_new.dropna(subset=['Date_Obj'])
            df_new = df_new.sort_values(by=['Date_Obj', 'Bazi'], ascending=[True, True]).drop(columns=['Date_Obj'])
            df_new.to_csv(csv_filename, index=False)
            print(f"[{scrape_ts}] Created new database. Scraped {len(df_new)} records.")
            df = df_new

        try:
            import github_sync
            from threading import Thread
            sync_thread = Thread(target=github_sync.upload_to_github, daemon=True)
            sync_thread.start()
            sync_thread.join(timeout=45) # Increased timeout for GitHub sync to 45s
            if sync_thread.is_alive():
                print("GitHub sync timed out after 45s, continuing main process...")
        except ImportError:
            print("INFO: 'github_sync' module not found. Skipping GitHub synchronization.")
        except Exception as e:
            print(f"ERROR: GitHub sync failed with exception: {e}")

        return df
    except Exception as e:
        print(f"Error processing and saving data: {e}")
        return None

if __name__ == "__main__":
    scrape_kolkata_ff()