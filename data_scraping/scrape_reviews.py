from google_play_scraper import reviews, Sort
import pandas as pd
import time

# Daftar aplikasi
apps = {
    'Netflix': 'com.netflix.mediaclient',
    'Vidio': 'com.vidio.android',
    'Disney+ Hotstar': 'in.startv.hotstar.dplus',
    'WeTV': 'com.tencent.qqlivei18n',
    'Viu': 'com.vuclip.viu'
}

all_reviews = []

for app_name, app_id in apps.items():
    print(f"Scraping {app_name}...")
    
    result, _ = reviews(
        app_id,
        lang='id',           # Bahasa Indonesia
        country='id',        # Indonesia
        sort=Sort.NEWEST,    # Review terbaru
        count=2000           # 2000 review per app
    )
    
    for r in result:
        all_reviews.append({
            'app_name': app_name,
            'username': r['userName'],
            'rating': r['score'],
            'review': r['content'],
            'date': r['at'],
            'thumbs_up': r['thumbsUpCount']
        })
    
    print(f"✓ {app_name}: {len(result)} reviews")
    time.sleep(2)  # Jeda biar sopan

# Simpan ke CSV
df = pd.DataFrame(all_reviews)
df.to_csv('streaming_reviews.csv', index=False)
print(f"\nTotal: {len(df)} reviews tersimpan!")