import requests, re, datetime
from bs4 import BeautifulSoup
import pandas as pd

url = "https://kolkataff.in/"
headers = {"User-Agent": "Mozilla/5.0"}
soup = BeautifulSoup(requests.get(url, headers=headers).content, "html.parser")

all_data = []

for table in soup.find_all('table'):
    rows = table.find_all('tr')
    if len(rows) < 3: continue
    
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

df = pd.DataFrame(all_data)
if not df.empty:
    print(df.head(15))
else:
    print("NO DATA")
