import requests
import os
import time

BASE_URL = "https://shikimori.one/animes/"
OUTPUT_DIR = "pages"
INDEX_FILE = "index.txt"
NUM_PAGES = 100
DELAY = 1

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

successful_pages = 0
current_id = 1

with open(INDEX_FILE, "w", encoding="utf-8") as index_file:
    while successful_pages < NUM_PAGES:
        url = f"{BASE_URL}{current_id}"

        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()

            file_name = f"{OUTPUT_DIR}/page_{current_id}.txt"
            with open(file_name, "w", encoding="utf-8") as file:
                file.write(response.text)

            index_file.write(f"{successful_pages + 1}: {url}\n")
            print(f"Страница {current_id} успешно сохранена: {url}")

            successful_pages += 1

        except requests.exceptions.RequestException as e:
            print(f"Ошибка при загрузке страницы {url}: {e}")

        current_id += 1
        time.sleep(DELAY)

print("Выкачивание завершено.")
