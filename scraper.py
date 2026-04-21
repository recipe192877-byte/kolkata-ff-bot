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

def scrape_kolkata_ff_in():
    url = "https://kolkataff.in/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    all_data = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
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
                p = pattis[bazi_idx]
                s = singles[bazi_idx]
                
                if not p or not s or 'Tips' in p or 'Tips' in s or p == '' or s == '':
                    continue
                
                p_val = re.sub(r'\D', '', p)
                s_val = re.sub(r'\D', '', s)
                if p_val and s_val:
                    res = p_val + s_val
                    all_data.append({
                        'Date': date_parsed, 'Bazi': bazi_idx + 1,
                        'Result_String': res, 'Patti': p_val, 'Single': s_val
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
    
    try:
        print("Fetching data from kolkataff.tv...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
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
                            digits = digits[:4]  # Sanitize: take only first 4 digits
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
    except Exception as e:
        print(f"Error scraping data from kolkataff.tv: {e}")
        
    try:
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
        df_new['Has_Single'] = df_new['Single'].notna()
        df_new = df_new.sort_values(['Date', 'Bazi', 'Has_Single'])
        df_new = df_new.drop_duplicates(subset=['Date', 'Bazi'], keep='last').drop(columns=['Has_Single'])
        
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
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    scrape_kolkata_ff()
