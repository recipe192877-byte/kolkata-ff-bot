import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import datetime

def standardize_date(date_str):
    """Converts 'SUNDAY, 15 MARCH 2026' or 'WEDNESDAY, 18 MARCH 2026' to '15/03/2026'"""
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
    except Exception as e:
        print(f"Date parse error for '{date_str}': {e}")
        pass
        
    return date_str

def scrape_kolkata_ff():
    url = "https://kolkataff.tv/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }
    
    try:
        print("Fetching data from kolkataff.tv...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')
        
        all_data = []
        
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
                            patti, single = digits[:3], digits[3]
                        elif len(digits) == 1:
                            patti, single = None, digits
                        else:
                            patti, single = digits, None
                    else:
                        patti, single = None, None
                        
                all_data.append({
                    'Date': date_col, 'Bazi': bazi_num,
                    'Result_String': result, 'Patti': patti, 'Single': single
                })
                
        df_new = pd.DataFrame(all_data)
        csv_filename = 'kolkata_ff_history_advanced.csv'
        
        try:
            df_old = pd.read_csv(csv_filename)
            combined = pd.concat([df_old, df_new]).drop_duplicates(subset=['Date', 'Bazi'], keep='last')
            # Sort chronologically
            combined['Date_Obj'] = pd.to_datetime(combined['Date'], format='%d/%m/%Y', errors='coerce')
            combined = combined.sort_values(by=['Date_Obj', 'Bazi'], ascending=[True, True]).drop(columns=['Date_Obj'])
            
            combined.to_csv(csv_filename, index=False)
            print(f"Successfully scraped {len(df_new)} records. Total in DB: {len(combined)}.")
            df = combined
        except FileNotFoundError:
            df_new.to_csv(csv_filename, index=False)
            print(f"Created new database. Scraped {len(df_new)} records.")
            df = df_new
        
        try:
            import github_sync
            github_sync.upload_to_github()
        except ImportError:
            pass
            
        return df

    except Exception as e:
        print(f"Error scraping data: {e}")
        return None

if __name__ == "__main__":
    scrape_kolkata_ff()
