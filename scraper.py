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
    date_str = date_str.replace('"', '').replace('(', '').replace(')', '').strip()
    
    if re.match(r'\d{2}/\d{2}/\d{4}', date_str):
        return date_str
        
    try:
        if ',' in date_str:
            date_part = date_str.split(',')[1].strip()
        else:
            date_part = date_str
            
        # Remove non-alphanumeric chars and ordinal suffixes (1ST, 2ND, 3RD, 4TH etc.)
        date_part = re.sub(r'[^\w\s]', '', date_part).strip()
        date_part = re.sub(r'(\d+)(?:ST|ND|RD|TH)', r'\1', date_part, flags=re.IGNORECASE)
        
        dt_obj = datetime.datetime.strptime(date_part, '%d %B %Y')
        return dt_obj.strftime('%d/%m/%Y')
    except Exception as e:
        print(f"Date parse error for '{date_str}': {e}")
        
    return date_str


def _fetch_with_retry(url, headers, retries=MAX_RETRIES):
    """Fetch URL with retry logic and exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"  Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                wait = RETRY_DELAY * attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    return None


def scrape_kolkata_ff_in():
    url = "https://kolkataff.in/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
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
            if len(rows) < 3:
                continue
                
            date_col = rows[0].get_text(strip=True)
            try:
                date_str_clean = re.sub(r'[^\w\s]', '', date_col).strip()
                dt_obj = datetime.datetime.strptime(date_str_clean, '%d %B %Y')
                date_parsed = dt_obj.strftime('%d/%m/%Y')
            except Exception:
                continue
                
            if len(rows) >= 4 and '1234' in rows[1].get_text(strip=True).replace(' ', ''):
                patti_row = 2
                single_row = 3
            else:
                patti_row = 1
                single_row = 2
                
            if len(rows) <= single_row:
                continue
                
            pattis = [c.get_text(strip=True) for c in rows[patti_row].find_all(['td', 'th'])]
            singles = [c.get_text(strip=True) for c in rows[single_row].find_all(['td', 'th'])]
            
            num_bazis = min(len(pattis), len(singles), 8)
            for bazi_idx in range(num_bazis):
                try:
                    p = pattis[bazi_idx]
                    s = singles[bazi_idx]
                except IndexError:
                    continue
                
                if not p or not s or 'Tips' in p or 'Tips' in s or p == '' or s == '':
                    continue
                
                p_val = re.sub(r'\D', '', p)
                s_val = re.sub(r'\D', '', s)
                if p_val and s_val:
                    res = p_val + s_val
                    all_data.append({
                        'Date': date_parsed, 'Bazi': bazi_idx + 1,
                        'Result_String': res, 'Patti': p_val, 'Single': s_val,
                        'Source': 'kolkataff.in'
                    })
        return all_data
    except Exception as e:
        print(f"Error scraping fallback kolkataff.in: {e}")
        return []


def scrape_kolkata_ff():
    url = "https://kolkataff.tv/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
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
                if len(rows) < 2:
                    continue
                    
                date_col = rows[0].get_text(strip=True)
                if "Result Time" in date_col or "Time" in date_col:
                    continue
                    
                date_col = standardize_date(date_col)
                
                cols = rows[1].find_all(['td', 'th'])
                bazi_results = [c.get_text(strip=True) for c in cols]
                
                for bazi_idx, result in enumerate(bazi_results[:8]):
                    bazi_num = bazi_idx + 1
                    
                    if result == '--' or 'Refresh' in result or 'Tips' in result or result == '' or result == '-':
                        patti, single = None, None
                    else:
                        match = re.search(r'(\d+)', result)
                        if match:
                            digits = match.group(1)
                            if len(digits) >= 4:
                                digits = digits[:4]
                                patti, single = digits[:3], digits[3]
                            elif len(digits) == 1:
                                patti, single = None, digits
                            else:
                                patti, single = digits, None
                        else:
                            patti, single = None, None
                            
                    all_data.append({
                        'Date': date_col, 'Bazi': bazi_num,
                        'Result_String': result, 'Patti': patti, 'Single': single,
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
        
        # FIX #3: Better dedup — prefer records with both Patti AND Single (most complete)
        # Add completeness score: records with both patti+single rank higher
        df_new['_completeness'] = df_new.apply(
            lambda r: (2 if pd.notna(r.get('Patti')) and str(r.get('Patti', '')).strip() != '' else 0) +
                      (1 if pd.notna(r.get('Single')) and str(r.get('Single', '')).strip() != '' else 0),
            axis=1
        )
        # Sort by completeness (highest first) so drop_duplicates keeps the best
        df_new = df_new.sort_values(['Date', 'Bazi', '_completeness'], ascending=[True, True, False])
        df_new = df_new.drop_duplicates(subset=['Date', 'Bazi'], keep='first')
        df_new = df_new.drop(columns=['_completeness'], errors='ignore')
        
        csv_filename = 'kolkata_ff_history_advanced.csv'
        
        try:
            df_old = pd.read_csv(csv_filename)
            combined = pd.concat([df_old, df_new])
            
            # Same completeness-based dedup for the combined dataset
            combined['_completeness'] = combined.apply(
                lambda r: (2 if pd.notna(r.get('Patti')) and str(r.get('Patti', '')).strip() != '' else 0) +
                          (1 if pd.notna(r.get('Single')) and str(r.get('Single', '')).strip() != '' else 0),
                axis=1
            )
            combined = combined.sort_values(['Date', 'Bazi', '_completeness'], ascending=[True, True, False])
            combined = combined.drop_duplicates(subset=['Date', 'Bazi'], keep='first')
            combined = combined.drop(columns=['_completeness'], errors='ignore')
            
            combined['Date_Obj'] = pd.to_datetime(combined['Date'], format='%d/%m/%Y', errors='coerce')
            combined = combined.sort_values(by=['Date_Obj', 'Bazi'], ascending=[True, True]).drop(columns=['Date_Obj'])
            
            combined.to_csv(csv_filename, index=False)
            print(f"[{scrape_ts}] Scraped {len(df_new)} records. Total in DB: {len(combined)}.")
            df = combined
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
            pass
        except Exception as e:
            print(f"GitHub sync error: {e}")
            
        return df
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    scrape_kolkata_ff()
