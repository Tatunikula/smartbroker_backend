from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import statistics
import time

def hae_hintatiedot_kaupunginosalla(haku: str, max_odotus=15):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get("https://asuntojen.hintatiedot.fi/haku")
        print("📥 Haetaan postinumerolla:", haku)

        # Avaa hakulomake mobiilissa tai jos ei muuten näy
        try:
            open_search = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable((By.ID, "openSearch"))
            )
            open_search.click()
            print("📂 Avattiin hakulomake")
        except:
            print("✅ Hakulomake jo auki")

        # Odotetaan että kenttä on klikattavissa
        input_field = WebDriverWait(driver, max_odotus).until(
            EC.element_to_be_clickable((By.ID, "postalField"))
        )
        input_field.clear()
        input_field.send_keys(haku)
        time.sleep(0.5)  # Odota että kentän syöttö aktivoi lomakkeen

        # Klikkaa hakunappia manuaalisesti
        search_button = WebDriverWait(driver, max_odotus).until(
            EC.element_to_be_clickable((By.ID, "search"))
        )
        search_button.click()
        time.sleep(0.5)

        # Lähetä vielä varmuudeksi lomake
        form = driver.find_element(By.TAG_NAME, "form")
        form.submit()

        print("🕒 Odotetaan tulostaulukkoa...")
        WebDriverWait(driver, max_odotus).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
        )
        time.sleep(2)

        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        hinnat = []

        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 6:
                teksti = cells[5].text.replace("€", "").replace(",", ".").replace(" ", "").strip()
                try:
                    arvo = float(teksti)
                    hinnat.append(arvo)
                except ValueError:
                    continue

        print("🔎 Löytyi", len(hinnat), "hintaa")
        print("➡️ Ensimmäiset:", hinnat[:5])

        driver.quit()
        if not hinnat:
            return None, 0
        return round(statistics.median(hinnat)), len(hinnat)

    except Exception as e:
        print("📦 Scraper error:", e)
        with open("page_source.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        driver.quit()
        return None, 0
