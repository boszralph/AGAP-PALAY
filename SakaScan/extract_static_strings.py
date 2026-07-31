# extract_static_strings.py
import os
import re
from deep_translator import GoogleTranslator
import mysql.connector  # or use your existing MySQL connection

# --- MySQL config (same as your app) ---
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'sakascan_db'
}

# --- Scan all templates ---
template_dir = 'templates'
pattern = re.compile(r"\{\{\s*'([^']*)'\s*\|\s*translate\s*\}\}")

strings_found = set()

for root, dirs, files in os.walk(template_dir):
    for file in files:
        if file.endswith('.html'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            matches = pattern.findall(content)
            for m in matches:
                # Skip empty or whitespace-only strings
                if m.strip():
                    strings_found.add(m.strip())

print(f"Found {len(strings_found)} unique strings.")

# --- Insert into translations table (if missing) ---
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

translator = GoogleTranslator(source='en', target='tl')

inserted = 0
for text in strings_found:
    # Check if already exists
    cursor.execute(
        "SELECT id FROM translations WHERE source_text = %s AND target_lang = 'tl'",
        (text,)
    )
    if not cursor.fetchone():
        # Auto-translate
        try:
            translated = translator.translate(text)
            cursor.execute(
                "INSERT INTO translations (source_text, target_lang, translated_text) VALUES (%s, %s, %s)",
                (text, 'tl', translated)
            )
            inserted += 1
            print(f"✅ Inserted: {text[:40]}... -> {translated[:40]}...")
        except Exception as e:
            print(f"❌ Failed: {text} - {e}")

conn.commit()
cursor.close()
conn.close()

print(f"Done. Inserted {inserted} new translations.")
