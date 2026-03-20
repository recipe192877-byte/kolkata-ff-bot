import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import datetime
import time
from concurrent.futures import ThreadPoolExecutor

MONTHS = [
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]
YEARS = [2024, 2025, 2026]
URL_PATTERN = "https://kolkataff.tv/old-kolkata-ff-fatafat-result/monthly/index.php?month={}&year={}"
CSV_FILE = "kolkata_ff_history_advanced.csv"

def standardize_date(date_str):
    """Converts verbose dates to DD/MM/YYYY"""
    date_str = date_str.replace('"', '').replace('(', '').replace(')', '').strip()
    if re.match(r'\d{2}/\d{2}/\d{4}', date_str):
        return date_str
    try:
        if ',' in date_str:
            date_part = date_str.split(',')[1].strip()
        else:
            date_part = date_str
        date_part = re.sub(r'[^\w\s]', '', date_part).strip()
        dt_obj = datetime.datetime.strptime(date_part, '%d %B %Y')
        return dt_obj.strftime('%d/%m/%Y')
    except Exception:
        pass
    return date_str

def fetch_month_data(year, month):
    url = URL_PATTERN.format(month, year)
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')
        
        all_data = []
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2: continue
            
            date_col = rows[0].get_text(strip=True)
            if "Result Time" in date_col or "Time" in date_col: continue
            
            date_col = standardize_date(date_col)
            cols = rows[1].find_all(['td', 'th'])
            bazi_results = [c.get_text(strip=True) for c in cols]
            
            # Usually Bazi 1-8
            for bazi_idx, result in enumerate(bazi_results[:8]):
                bazi_num = bazi_idx + 1
                patti, single = None, None
                
                if result and result != '--' and result != '-' and 'Refresh' not in result and 'Tips' not in result:
                    match = re.search(r'(\d+)', result)
                    if match:
                        digits = match.group(1)
                        if len(digits) >= 4:
                            patti, single = digits[:3], digits[3]
                        elif len(digits) == 1:
                            single = digits
                        else:
                            patti = digits
                
                all_data.append({
                    'Date': date_col,
                    'Bazi': bazi_num,
                    'Result_String': result,
                    'Patti': patti,
                    'Single': single
                })
        return all_data
    except Exception as e:
        print(f"Error fetching {month} {year}: {e}")
        return []

def run_deep_scraper():
    print("Starting Deep Historical Scraper for Kolkata FF...")
    all_records = []
    
    # Use threads to speed up scraping 3 years * 12 months = 36 requests
    tasks = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for year in YEARS:
            for month in MONTHS:
                # Basic optimization: don't fetch future dates
                if year == 2026 and MONTHS.index(month) > 2: # Stop after March 2026
                    continue
                tasks.append(executor.submit(fetch_month_data, year, month))
                
        for i, future in enumerate(tasks):
            data = future.result()
            if data:
                all_records.extend(data)
            print(f"Processed {i+1}/{len(tasks)} requests...", end='\r')
            
    # Also scrape homepage to get the absolute latest if any
    print("\nFetching latest data from Homepage...")
    try:
        home_res = requests.get("https://kolkataff.tv/", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(home_res.content, 'html.parser')
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2: continue
            date_col = standardize_date(rows[0].get_text(strip=True))
            if "Time" in date_col: continue
            bazi_results = [c.get_text(strip=True) for c in rows[1].find_all(['td', 'th'])]
            for bazi_idx, result in enumerate(bazi_results[:8]):
                patti, single = None, None
                if result:
                    match = re.search(r'(\d+)', result)
                    if match:
                        digits = match.group(1)
                        if len(digits) >= 4:
                            patti, single = digits[:3], digits[3]
                        elif len(digits) == 1:
                            single = digits
                all_records.append({'Date': date_col, 'Bazi': bazi_idx+1, 'Result_String': result, 'Patti': patti, 'Single': single})
    except Exception as e:
        print(f"Homepage fetch error: {e}")

    df = pd.DataFrame(all_records)
    # Filter out empty records
    df = df.dropna(subset=['Single'])
    # Remove duplicates
    df = df.drop_duplicates(subset=['Date', 'Bazi'], keep='last')
    
    # Sort chronologically
    df['Date_Obj'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date_Obj'])
    df = df.sort_values(by=['Date_Obj', 'Bazi'], ascending=[True, True]).drop(columns=['Date_Obj'])
    
    df.to_csv(CSV_FILE, index=False)
    print(f"\nDeep Scraping Complete! Total valid records saved: {len(df)}")

if __name__ == "__main__":
    start_time = time.time()
    run_deep_scraper()
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")
