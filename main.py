from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List
from analyysi.kunto_arvio import arvioi_kunto
from laskenta.hintalaskuri import laske_kokonaishinta
from arviointi.gpt_perustelu import generoi_hinta_selitys
from pdf2image import convert_from_bytes
import pytesseract
from io import BytesIO
from laskenta.hintatiedot_scraper import hae_hintatiedot_kaupunginosalla
from kartta.postinumero_kaupunginosa_map import postinumero_to_kaupunginosa
import json
import uuid
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from fastapi.responses import StreamingResponse, JSONResponse
import io, os, json
from datetime import datetime
# tiedoston yläosaan muiden importtien jälkeen
from playwright.sync_api import sync_playwright

def html_to_pdf_bytes(html: str) -> bytes:
    # luo PDF Chromella; toimii Dockerissa
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="load")
        pdf = page.pdf(
            format="A4",
            margin={"top":"10mm","right":"10mm","bottom":"10mm","left":"10mm"},
            print_background=True
        )
        browser.close()
        return pdf

# -------------------- PROGRESS (latauspalkki) --------------------
import threading
PROGRESS = {}  # job_id -> {"percent":0,"stage":"", "done":False, "error":None, "result":None}
PROGRESS_LOCK = threading.Lock()
BG_EXECUTOR = ThreadPoolExecutor(max_workers=3)

def _set_progress(job_id, **kv):
    with PROGRESS_LOCK:
        state = PROGRESS.get(job_id, {"percent": 0, "stage": "", "done": False, "error": None, "result": None})
        state.update(kv)
        PROGRESS[job_id] = state

# -------------------- INIT --------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY puuttuu .env-tiedostosta.")

# Mallit
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "gpt-4o")                     # ensisijainen
SECONDARY_EXTRACT_MODEL = None
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-5")                              # generointi (ilmoitus, chat)

client = OpenAI(api_key=api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5176"],


    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- HTML -> PDF Playwrightilla --------------------
import asyncio


async def _html_to_pdf_bytes_async(html_str: str) -> bytes:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        # Ladataan HTML ja odotetaan, että sisältö on valmis
        await page.set_content(html_str, wait_until="load")
        pdf_bytes = await page.pdf(
            format="A4",
            print_background=True,
            margin={"top": "10mm", "right": "10mm", "bottom": "10mm", "left": "10mm"},
        )
        await browser.close()
        return pdf_bytes

def html_to_pdf_bytes(html_str: str) -> bytes:
    # Synkronoitu helperi; turvallinen kutsua FastAPIn sync-reiteistä
    return asyncio.run(_html_to_pdf_bytes_async(html_str))

# -------------------- HELPERS --------------------
def strip_html(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(separator="\n").strip()

def hae_aluehinta(sijainti: str):
    sijainti = sijainti.strip()
    if sijainti.isdigit() and len(sijainti) == 5:
        sijainti = postinumero_to_kaupunginosa(sijainti) or sijainti
    mediaani, maara = hae_hintatiedot_kaupunginosalla(sijainti)
    return (mediaani, maara) if mediaani else (None, 0)

def _normalize_text(s: str) -> str:
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)  # poista tavutus
    s = s.replace("\r", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = s.replace("\n", " ")
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def extract_text_from_pdf(file_bytes):
    try:
        import platform

        # Aseta Tesseractin polku vain Windowsissa (ei pilvessä/Linuxissa)
        if platform.system() == "Windows":
            os.environ["TESSDATA_PREFIX"] = r"C:\\Users\\User\\Downloads\\tesseract-ocr-w64-setup-5.5.0.20241111\\tessdata"

        # Lue sisään bytes
        content = file_bytes.read() if hasattr(file_bytes, "read") else file_bytes

        # 1) Suora tekstinpoiminta PDF:stä
        reader = PdfReader(BytesIO(content))
        texts = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                texts.append(t)
        text = "\n".join(texts)
        if text.strip():
            return _normalize_text(text)

        # 2) OCR-fallback (kuvamuotoiset PDF:t)
        images = convert_from_bytes(content, timeout=30)
        with ThreadPoolExecutor() as ex:
            ocr_texts = list(ex.map(lambda img: pytesseract.image_to_string(img, lang="fin"), images))
        return _normalize_text("\n".join(ocr_texts))

    except Exception as e:
        print("❌ PDF-tekstinluku epäonnistui:", e)
        return ""


def split_chunks(s: str, max_len: int = 12000):
    # Säilytä kaikki chunkit; älä pudota lyhyitä rippeitä (niissä on usein talousrivit)
    return [s[i:i+max_len] for i in range(0, len(s), max_len)] or []

# --- helper: pinta-alan parsiminen floatiksi (LISÄTTY) ---
def _parse_pinta_ala_to_float(pinta_ala):
    if pinta_ala is None:
        return None
    if isinstance(pinta_ala, (int, float)):
        return float(pinta_ala)
    m = re.search(r"[\d,.]+", str(pinta_ala))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return None

# --------- FACT-SCHEMA (laajennettu) ----------
FACT_SCHEMA_TEXT = """{
  "osoite": "",
  "huonejako": "",
  "pinta_ala_m2": "",
  "kerros": "",
  "hissi": "",
  "rakennusvuosi": "",
  "rakennustyyppi": "",
  "kunto": "",
  "erityispiirteet": [],

  "parveke": "",
  "sauna": "",
  "sailytystilat": "",
  "autopaikka": "",
  "lammitus": "",

  "kylpyhuone_varustus": [],
  "palvelut": [],
  "liikenneyhteydet": [],
  "vapaa_aika": [],

  "laajakaista_nopeus": "",
  "laajakaista_toimittaja": "",
  "kaapeli_tv": "",

  "autohalli": "",
  "invapaikat": "",
  "autopaikat_yht": "",
  "autopaikat_osakepaikkoja": "",

  "tontin_vuokra_paattyy": "",
  "tontinvuokravastike_e_m2kk": "",
  "tontinvuokravastike_e_kk": "",
  "vuokranantaja": "",

  "taloyhtion_tilat": [],
  "tehdyt_remontit": [],
  "tulevat_remontit": [],

  "talous": {
    "hoitovastike": "",
    "paomavastike": "",
    "yhtiovastike_yht": "",
    "lainaosuus": "",
    "vesimaksu": "",
    "vesimittari": "",
    "lunastuslauseke": "",
    "tontti": "",
    "autopaikkamaksu": "",
    "saunamaksu": "",
    "muut_maksut": []
  }
}"""

def build_extract_prompt(txt: str) -> str:
    return f"""
Poimi alla olevasta tekstistä VAIN yksittäistä asuntoa koskevat faktat JSON-muodossa.
- Käytä annettua schemaa. Jos jotakin kenttää ei löydy, jätä tyhjäksi ("" tai []).
- Älä keksi puuttuvia arvoja.
- Numerot suomalaisessa muodossa (esim. "40,5 m²", "738,45 €/kk", "180 762,48 €", "3/6", "2012").
- Erota hoito- ja pääomavastike. Jos vain "yhtiövastike", laita se yhtiovastike_yht-kenttään.
- Kerros voi olla "3/6" tai "3. kerros". Hissi: "kyllä"/"ei".
- Parveke/sauna/autopaikka: "kyllä"/"ei" tai lyhyt tarkenne.
- Tontti: "vuokrattu" (ja vuokrasopimuksen päättymisvuosi jos näkyy) tai "omistus".
- Huonejako: poimi mm. "Tyyppikuvaus" / "Huonejako" / "Huoneistoselitelmä" riviltä (esim. "2h+kt").
- Kunto: poimi, jos dokumentissa mainitaan (erinomainen, hyvä, tyydyttävä, heikko).

Schema:
{FACT_SCHEMA_TEXT}

Teksti:
{txt}
""".strip()

def call_extract_model(model: str, chunk_text: str) -> dict:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": build_extract_prompt(chunk_text)}]
        )
        out = (r.choices[0].message.content or "").strip().strip("`")
        m = re.search(r"\{.*\}", out, re.S)
        return json.loads(m.group(0)) if m else {}
    except Exception as e:
        print(f"⚠️ Extract-malli '{model}' antoi virheen: {e}")
        return {}

def score_facts(f: dict) -> int:
    if not isinstance(f, dict): return 0
    score = 0
    def non_empty_str(x): return isinstance(x, str) and x.strip() != ""
    add = lambda b: 1 if b else 0

    score += add(non_empty_str(f.get("osoite","")))
    score += add(non_empty_str(f.get("huonejako","")))
    score += add(non_empty_str(f.get("pinta_ala_m2","")))
    score += add(non_empty_str(f.get("kerros","")))
    score += add(non_empty_str(f.get("hissi","")))
    score += add(non_empty_str(f.get("rakennusvuosi","")))
    score += len(f.get("tehdyt_remontit") or [])
    score += len(f.get("tulevat_remontit") or [])

    tal = f.get("talous") or {}
    score += add(non_empty_str(tal.get("hoitovastike","")))
    score += add(non_empty_str(tal.get("paomavastike","")))
    score += add(non_empty_str(tal.get("yhtiovastike_yht","")))
    score += add(non_empty_str(tal.get("lainaosuus","")))
    score += add(non_empty_str(tal.get("vesimaksu","")))
    score += add(non_empty_str(tal.get("vesimittari","")))
    score += add(non_empty_str(tal.get("lunastuslauseke","")))
    score += add(non_empty_str(tal.get("tontti","")))
    return score

def merge_facts(a: dict, b: dict) -> dict:
    res = a.copy()
    for k, v in (b or {}).items():
        if isinstance(v, dict):
            res[k] = merge_facts(a.get(k, {}), v)
        elif isinstance(v, list):
            res[k] = (a.get(k) or []) + v
        else:
            res[k] = a.get(k) or v
    return res

# --- LISÄTYYPPISET PARSERIT RAAKATEKSTILLE ---
def parse_tontti_info(text: str) -> dict:
    t = text or ""
    out = {}
    m = re.search(r"Tontin\s+omistus\s+([A-Za-zÅÄÖåäö ]+)", t, re.I)
    if m: out["tontti_tyyppi"] = m.group(1).strip()
    m = re.search(r"Vuokranantaja\s+([^\n\r]+)", t, re.I)
    if m: out["vuokranantaja"] = m.group(1).strip()
    m = re.search(r"Vuokrauksen\s+päättymispäivä\s+([0-9.\-]+)", t, re.I)
    if m: out["tontin_vuokra_paattyy"] = m.group(1).strip()
    m = re.search(r"Tontinvuokravastike[^\n\r]*?\s([\d,.]+)\s*€\s*/?\s*m2", t, re.I)
    if m: out["tontinvuokravastike_e_m2kk"] = m.group(1).replace(",", ".")
    m = re.search(r"(?m)Tontinvuokravastike[^\n\r]*?\s([\d,.]+)\s*€\s*$", t, re.I)
    if m: out["tontinvuokravastike_e_kk"] = m.group(1).replace(",", ".")
    return out

def parse_laajakaista_ja_tv(text: str) -> dict:
    t = text or ""
    out = {}
    m = re.search(r"laajakaista[^\n\r]*?(\d{2,4}\s*/\s*\d{2,4}\s*M)", t, re.I)
    if m: out["laajakaista_nopeus"] = re.sub(r"\s+", "", m.group(1)).replace("M","M")
    m = re.search(r"Internetyhteyden\s+palveluntarjoaja\s+([^\n\r]+)", t, re.I)
    if m: out["laajakaista_toimittaja"] = m.group(1).strip()
    if re.search(r"Kaapeli-?TV", t, re.I):
        out["kaapeli_tv"] = "kyllä"
    return out

def parse_autopaikat(text: str) -> dict:
    t = text or ""
    out = {}
    m = re.search(r"Autotalli/hallipaikat\s+(\d+)", t, re.I)
    if m: out["autohalli"] = m.group(1)
    m = re.search(r"inva\s*(\d+)", t, re.I)
    if m: out["invapaikat"] = m.group(1)
    m = re.findall(r"Autopaikka[^\n\r]*\s(\d+)\s", t, re.I)
    if m:
        try:
            out["autopaikat_yht"] = str(sum(int(x) for x in m))
        except:
            pass
    if re.search(r"Autopaikat\s+osakepaikkoja", t, re.I):
        out["autopaikat_osakepaikkoja"] = "kyllä"
    return out

def parse_huonejako(text: str) -> str:
    t = text or ""
    m = re.search(r"(Tyyppikuvaus|Huoneistoselitelm[aä]|Huonejako)\s*[:\-]?\s*([0-9]+\s*h\s*\+\s*[a-zåäöA-ZÅÄÖ]+(?:\s*\+\s*[a-zåäöA-ZÅÄÖ]+)?)", t, re.I)
    if m: return re.sub(r"\s+", "", m.group(2))
    m = re.search(r"\b([1-5]\s*h\s*\+\s*[a-zåäöA-ZÅÄÖ]{1,4}(?:\s*\+\s*[a-zåäöA-ZÅÄÖ]{1,4})?)\b", t, re.I)
    return re.sub(r"\s+", "", m.group(1)) if m else ""

def parse_kunto(text: str) -> str:
    t = text or ""
    for word in ["erinomainen","hyvä","tyydyttävä","heikko"]:
        if re.search(fr"\b{word}\b", t, re.I):
            return word
    m = re.search(r"kunto\s*[:\-]\s*([a-zåäö]+)", t, re.I)
    return (m.group(1).lower() if m else "")

def parse_remontit(text: str) -> dict:
    t = text or ""
    out = {"tehdyt": [], "tulevat": []}
    m = re.search(r"(?is)(Tehdyt korjaukset|Korjaushistoria|Suoritetut korjaukset).*?(?=(Päätetyt|Tulevat|Yhteenveto|$))", t)
    if m:
        block = m.group(0)
        out["tehdyt"] = [s.strip(" -•\t") for s in re.findall(r"(?m)^\s*(?:\d{4}(?:[\-–]\d{4})?)\s*.*$", block)]
    blocks = re.split(r"(?i)\b(Tulevat remontit|Päätetyt ja käynnissä olevat korjaukset)\b", t)
    if len(blocks) >= 3:
        after = blocks[2]
        out["tulevat"] = [s.strip(" -•\t") for s in re.findall(r"(?m)^\s*(?:\d{4}(?:[\-–]\d{4})?)\s*.*$", after)]
    return out

def parse_sijainti_osio(text: str) -> dict:
    t = text or ""
    palvelut, liikenneyhteydet, vapaa = [], [], []
    if re.search(r"(metro|bussi|ratikka|raitiovaunu|juna)", t, re.I):
        liikenneyhteydet.append("Joukkoliikenne mainittu")
    if re.search(r"(kauppa|päivittäistavara|palvelu|keskus)", t, re.I):
        palvelut.append("Palvelut lähellä")
    if re.search(r"(meri|uimaranta|puisto|ulkoilu|reitit)", t, re.I):
        vapaa.append("Ulkoilumahdollisuuksia")
    return {"palvelut": palvelut, "liikenneyhteydet": liikenneyhteydet, "vapaa_aika": vapaa}

def compute_tontinvuokra_huoneisto_kk(facts: dict) -> str:
    rate = facts.get("tontinvuokravastike_e_m2kk")
    m2 = facts.get("pinta_ala_m2") or facts.get("kokonaispinta_ala_m2")
    if not rate or not m2:
        return ""
    try:
        r = float(str(rate).replace(",", "."))
        n = float(re.sub(r"[^\d.,]", "", m2).replace(",", "."))
        val = r * n
        return f"{val:,.2f} €".replace(",", "X").replace(".", ",").replace("X", " ")
    except Exception:
        return ""

def clean_remonttilista(items):
    out, seen = [], set()
    for raw in (items or []):
        s = str(raw or "").strip()
        if not s: continue
        s = re.sub(r"\s+", " ", s).strip(" -•\t,;")
        has_year = re.search(r"\b(19|20)\d{2}\b", s) is not None
        has_letters = re.search(r"[A-Za-zÅÄÖa-zåäö]", s) is not None
        if not has_year and not has_letters:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key); out.append(s)
    return out

# -------------------- YHTEINEN ANALYYSIMOOTTORI --------------------
def build_analysis_html_and_facts(full_text: str) -> dict:
    chunks = split_chunks(full_text, max_len=12000)
    if not chunks:
        return {"esite": "<p>Ei analysoitavaa.</p>"}

    # Valitse malli 1. chunkiin pisteytyksen perusteella
    first = chunks[0]
    f_primary = call_extract_model(EXTRACT_MODEL, first)
    f_secondary = call_extract_model(SECONDARY_EXTRACT_MODEL, first) if SECONDARY_EXTRACT_MODEL else {}
    s1, s2 = score_facts(f_primary), score_facts(f_secondary)
    chosen_model = SECONDARY_EXTRACT_MODEL if s2 > s1 else EXTRACT_MODEL
    facts = merge_facts({}, f_secondary if s2 > s1 else f_primary)

    # Loput chunkit valitulla mallilla
    rest = chunks[1:]
    if rest:
        def extract_with_chosen(c): return call_extract_model(chosen_model, c)
        with ThreadPoolExecutor(max_workers=min(8, len(rest))) as ex:
            for f in ex.map(extract_with_chosen, rest):
                facts = merge_facts(facts, f)

    # -------------------- Normalisoinnit & täydennykset --------------------
    def norm_bool(v: str):
        if not v: return ""
        s = str(v).strip().lower()
        if s in {"kylla","kyllä","k","yes","true","1"}: return "Kyllä"
        if s in {"ei","no","false","0"}: return "Ei"
        return v

    def norm_m2(val: str):
        if not val: return ""
        s = str(val).strip()
        s = s.replace(" m² m²", " m²").replace(" m2 m2", " m²")
        s = s.replace("m2", "m²")
        return s

    def clean_list(items):
        seen, out = set(), []
        for x in (items or []):
            t = (x or "").strip()
            if not t: continue
            t = t.replace("  ", " ").replace("Ulkoiluväline varasto","Ulkoiluvälinevarasto")
            if t not in seen:
                seen.add(t); out.append(t)
        return out

    def to_euros(s):
        if not s: return None
        x = re.sub(r"[^\d,.-]", "", str(s)).replace(",", ".")
        try: return float(x)
        except: return None

    # Kenttä-normalisoinnit
    facts["pinta_ala_m2"] = norm_m2(facts.get("pinta_ala_m2", ""))
    facts["hissi"] = norm_bool(facts.get("hissi", ""))
    facts["parveke"] = norm_bool(facts.get("parveke", ""))
    facts["sauna"] = norm_bool(facts.get("sauna", ""))
    facts["taloyhtion_tilat"] = clean_list(facts.get("taloyhtion_tilat"))
    facts["erityispiirteet"] = clean_list(facts.get("erityispiirteet"))

    tal = facts.get("talous") or {}
    if not tal.get("yhtiovastike_yht"):
        hv = to_euros(tal.get("hoitovastike"))
        pv = to_euros(tal.get("paomavastike"))
        if hv is not None and pv is not None:
            tal["yhtiovastike_yht"] = f"{hv + pv:.2f} €"
    facts["talous"] = tal

    # Regex‑täydennykset raakatekstistä
    tontti_add = parse_tontti_info(full_text)
    netti_add  = parse_laajakaista_ja_tv(full_text)
    auto_add   = parse_autopaikat(full_text)

    tal = facts.get("talous") or {}
    if tontti_add.get("tontti_tyyppi") and not tal.get("tontti"):
        tal["tontti"] = tontti_add["tontti_tyyppi"]
    facts["talous"] = tal

    for k in ["tontin_vuokra_paattyy","tontinvuokravastike_e_m2kk","tontinvuokravastike_e_kk","vuokranantaja"]:
        if tontti_add.get(k) and not facts.get(k):
            facts[k] = tontti_add[k]
    for k in ["laajakaista_nopeus","laajakaista_toimittaja","kaapeli_tv"]:
        if netti_add.get(k) and not facts.get(k):
            facts[k] = netti_add[k]
    for k in ["autohalli","invapaikat","autopaikat_yht","autopaikat_osakepaikkoja"]:
        if auto_add.get(k) and not facts.get(k):
            facts[k] = auto_add[k]

    # Fallbackit tärkeille kentille
    if not facts.get("huonejako"):
        hj = parse_huonejako(full_text)
        if hj: facts["huonejako"] = hj
    if not facts.get("kunto"):
        kn = parse_kunto(full_text)
        if kn: facts["kunto"] = kn

    rem = parse_remontit(full_text)
    if rem.get("tehdyt") and not (facts.get("tehdyt_remontit") or []):
        facts["tehdyt_remontit"] = clean_remonttilista(rem["tehdyt"])
    if rem.get("tulevat") and not (facts.get("tulevat_remontit") or []):
        facts["tulevat_remontit"] = clean_remonttilista(rem["tulevat"])

    # Laske tontinvuokravastike €/kk jos puuttuu
    if not facts.get("tontinvuokravastike_e_kk"):
        calc = compute_tontinvuokra_huoneisto_kk(facts)
        if calc:
            facts["tontinvuokravastike_e_kk"] = calc
            tal = facts.get("talous") or {}
            muut = tal.get("muut_maksut") or []
            if not any("Tontinvuokravastike" in (m or "") for m in muut):
                muut.append(f"Tontinvuokravastike: {calc}/kk")
            tal["muut_maksut"] = muut
            facts["talous"] = tal

    # ----- HTML rakentaminen -----
    def has_str(v): return isinstance(v, str) and v.strip() != ""
    def li(label, value): return f"<li>{label}: {value}</li>"
    def maybe_li(label, value): return li(label, value) if has_str(value) else ""
    def join_if(items):
        xs = [x for x in (items or []) if has_str(x)]
        return xs if xs else []

    osiot = []

    osa = ["<h2>Osoite ja koko</h2>", "<ul>"]
    osa.append(maybe_li("Osoite", facts.get("osoite")))
    osa.append(maybe_li("Huonejako", facts.get("huonejako")))
    osa.append(maybe_li("Pinta-ala", facts.get("pinta_ala_m2")))
    kerros = facts.get("kerros"); hissi = facts.get("hissi")
    if has_str(kerros) and has_str(hissi):
        osa.append(li("Kerros | Hissi", f"{kerros} | {hissi}"))
    else:
        osa.append(maybe_li("Kerros", kerros))
    osa.append("</ul>")
    if any("li>" in x for x in osa): osiot.append("\n".join(osa))

    osa = ["<h2>Rakennus ja kunto</h2>", "<ul>"]
    rt = facts.get("rakennustyyppi"); rv = facts.get("rakennusvuosi")
    if has_str(rt) and has_str(rv):
        osa.append(li("Rakennustyyppi | Rakennusvuosi", f"{rt} | {rv}"))
    else:
        osa.append(maybe_li("Rakennustyyppi", rt))
        osa.append(maybe_li("Rakennusvuosi", rv))
    osa.append(maybe_li("Kunto", facts.get("kunto")))
    osa.append("</ul>")
    if any("li>" in x for x in osa): osiot.append("\n".join(osa))

    osa = ["<h2>Varustelu</h2>", "<ul>"]
    osa.append(maybe_li("Parveke", facts.get("parveke")))
    osa.append(maybe_li("Sauna", facts.get("sauna")))
    osa.append(maybe_li("Säilytystilat", facts.get("sailytystilat")))
    osa.append(maybe_li("Autopaikka", facts.get("autopaikka")))
    osa.append(maybe_li("Lämmitys", facts.get("lammitus")))
    tark = []
    if has_str(facts.get("laajakaista_nopeus")): tark.append(facts["laajakaista_nopeus"])
    if has_str(facts.get("laajakaista_toimittaja")): tark.append(facts["laajakaista_toimittaja"])
    if facts.get("kaapeli_tv") == "kyllä": tark.append("Kaapeli-TV")
    if tark: osa.append(li("Laajakaista/TV", ", ".join(tark)))
    kvh = join_if(facts.get("kylpyhuone_varustus"))
    if kvh: osa.append(li("Kylpyhuoneen varusteet", ", ".join(kvh)))
    osa.append("</ul>")
    if any("li>" in x for x in osa): osiot.append("\n".join(osa))

    tilat = join_if(facts.get("taloyhtion_tilat"))
    if tilat: osiot.append("<h2>Taloyhtiö</h2>\n" + f"<p>Yhteiset tilat: {', '.join(tilat)}</p>")

    tehdyt = join_if(facts.get("tehdyt_remontit"))
    if tehdyt: osiot.append("<h2>Tehdyt remontit</h2>\n<ul>" + "".join(f"<li>{r}</li>" for r in tehdyt) + "</ul>")
    tulevat = join_if(facts.get("tulevat_remontit"))
    if tulevat: osiot.append("<h2>Tulevat remontit</h2>\n<ul>" + "".join(f"<li>{r}</li>" for r in tulevat) + "</ul>")

    tontti_rows, tal = [], (facts.get("talous") or {})
    if has_str(tal.get("tontti")):                   tontti_rows.append(li("Tontin omistus", tal["tontti"]))
    if has_str(facts.get("vuokranantaja")):          tontti_rows.append(li("Vuokranantaja", facts["vuokranantaja"]))
    if has_str(facts.get("tontin_vuokra_paattyy")):  tontti_rows.append(li("Vuokra päättyy", facts["tontin_vuokra_paattyy"]))
    if has_str(facts.get("tontinvuokravastike_e_m2kk")):
        tontti_rows.append(li("Tontinvuokravastike", f"{facts['tontinvuokravastike_e_m2kk']} €/m²/kk"))
    if has_str(facts.get("tontinvuokravastike_e_kk")):
        tontti_rows.append(li("Tontinvuokravastike (huoneisto)", f"{facts['tontinvuokravastike_e_kk']} €/kk"))
    if tontti_rows: osiot.append("<h2>Tontti</h2>\n<ul>" + "".join(tontti_rows) + "</ul>")

    paikat_rows = []
    if has_str(facts.get("autohalli")):                paikat_rows.append(li("Autohallipaikat", facts["autohalli"]))
    if has_str(facts.get("invapaikat")):               paikat_rows.append(li("Invapaikat", facts["invapaikat"]))
    if has_str(facts.get("autopaikat_yht")):           paikat_rows.append(li("Autopaikat yht.", facts["autopaikat_yht"]))
    if has_str(facts.get("autopaikat_osakepaikkoja")): paikat_rows.append("<li>Autopaikat ovat osakepaikkoja</li>")
    if paikat_rows: osiot.append("<h2>Autopaikat</h2>\n<ul>" + "".join(paikat_rows) + "</ul>")

    talous_rows = []
    if has_str(tal.get("hoitovastike")):    talous_rows.append(li("Hoitovastike", tal["hoitovastike"]))
    if has_str(tal.get("paomavastike")):    talous_rows.append(li("Pääomavastike", tal["paomavastike"]))
    if has_str(tal.get("yhtiovastike_yht")): talous_rows.append(li("Yhtiövastike yhteensä", tal["yhtiovastike_yht"]))
    if has_str(tal.get("lainaosuus")):      talous_rows.append(li("Lainaosuus", tal["lainaosuus"]))
    if has_str(tal.get("vesimaksu")):       talous_rows.append(li("Vesimaksu", tal["vesimaksu"]))
    if has_str(tal.get("vesimittari")):     talous_rows.append(li("Vesimittari", tal["vesimittari"]))
    if has_str(tal.get("autopaikkamaksu")): talous_rows.append(li("Autopaikkamaksu", tal["autopaikkamaksu"]))
    if has_str(tal.get("saunamaksu")):      talous_rows.append(li("Saunamaksu", tal["saunamaksu"]))
    mm = join_if(tal.get("muut_maksut"))
    if mm:                                  talous_rows.append(li("Muut maksut", ", ".join(mm)))
    if has_str(tal.get("lunastuslauseke")): talous_rows.append(li("Lunastuslauseke", tal["lunastuslauseke"]))
    if talous_rows: osiot.append("<h2>Talous</h2>\n<ul>" + "".join(talous_rows) + "</ul>")

    sij_rows = []
    ps = join_if(facts.get("palvelut")); liiks = join_if(facts.get("liikenneyhteydet")); vapaa = join_if(facts.get("vapaa_aika"))
    if ps:    sij_rows.append(li("Palvelut", ", ".join(ps)))
    if liiks: sij_rows.append(li("Liikenneyhteydet", ", ".join(liiks)))
    if vapaa: sij_rows.append(li("Vapaa-aika", ", ".join(vapaa)))
    if sij_rows: osiot.append("<h2>Sijainti ja ympäristö</h2>\n<ul>" + "".join(sij_rows) + "</ul>")

    html = "\n".join(osiot) if osiot else "<p>(Ei poimittuja tietoja)</p>"
    return {"esite": html, "facts": facts, "model": chosen_model}

# -------------------- ROUTES --------------------
@app.get("/")
def ping():
    return {"status": "ok"}

@app.get("/ilmoitukset")
def hae_ilmoitukset():
    if not os.path.exists("ilmoitukset.json"):
        return []
    with open("ilmoitukset.json", "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/tallenna_ilmoitus")
def tallenna_ilmoitus(data: dict = Body(...)):
    uusi_ilmo = {
        "id": str(uuid.uuid4()),
        "otsikko": data.get("otsikko", "(Otsikko puuttuu)"),
        "tila": data.get("tila", "Luonnos"),
        "nayttokerrat": 0,
        "ilmoitus": data.get("ilmoitus", ""),
        "kysymykset": data.get("kysymykset", []),
        "created_at": datetime.now().isoformat()
    }
    polku = "ilmoitukset.json"
    kaikki = []
    if os.path.exists(polku):
        with open(polku, "r", encoding="utf-8") as f:
            kaikki = json.load(f)
    kaikki.append(uusi_ilmo)
    with open(polku, "w", encoding="utf-8") as f:
        json.dump(kaikki, f, ensure_ascii=False, indent=2)
    return {"status": "ok", "id": uusi_ilmo["id"]}

@app.post("/analyze")
async def analyze_pdfs(pdfs: List[UploadFile] = File(...)):
    try:
        full_text = ""
        for idx, pdf in enumerate(pdfs):
            print(f"📥 Luetaan tiedosto: {pdf.filename}")
            pdf_bytes = await pdf.read()
            teksti = extract_text_from_pdf(BytesIO(pdf_bytes))
            full_text += f"\n\n[DOKUMENTTI {idx+1} – {pdf.filename}]\n{teksti}"
        if not full_text.strip():
            return {"error": "PDF-tiedostoista ei löytynyt luettavaa tekstiä."}
        result = build_analysis_html_and_facts(full_text)
        print("✅ Analyysi valmis valitulla mallilla:", result.get("model"))
        return {"esite": result["esite"]}
    except Exception as e:
        return {"error": f"Virhe analyysissä: {str(e)}"}

@app.post("/generate_ilmoitus")
async def generate_listing(data: dict = Body(...)):
    try:
        esite_html = data.get("esite", "")
        esite_teksti = strip_html(esite_html)
        prompt = f"""
Kirjoita SUOMEKSI erittäin ammattimainen ja houkutteleva asunnon myynti-ilmoitus alla olevan analyysin pohjalta.
Tyyli on kokeneen kiinteistönvälittäjän: lämmin, asiantunteva, vakuuttava ja täysin totuudenmukainen.
Käytä vain analyysissä olevia tietoja — älä arvaa tai lisää mitään.

RAKENNE JA TYYLI
- Pituus: 4–5 kappaletta, jokainen 3–5 virkettä.
- Käytä selkeitä <p>...</p> -tageja jokaiselle kappaleelle.
- Jokaisella kappaleella oma teema:
  1) Yleiskuva: sijainti + tärkein myyntivaltti
  2) Sisätilat ja pohjaratkaisu
  3) Taloyhtiön varustelu ja yhteiset tilat
  4) Sijainti, palvelut ja ympäristö
  5) Yhteenveto ja kutsu tutustumaan
- “Arki” sana saa esiintyä korkeintaan kerran.
- Sulauta faktat luontevasti tekstiin, vältä listamaisuutta ja toistoa.
- Älä käytä ylilatautuneita fraaseja kuten “täydellinen koti” tai “unelmakoti”.

FAKTAOSIO
- Tekstin jälkeen lisää <ul><li>...</li></ul>-osio, jossa tiedot seuraavassa järjestyksessä (vain jos analyysissä löytyy):
  Osoite; Huonejako; Pinta-ala; Kerros ja hissi; Rakennustyyppi / -vuosi; Lämmitystapa;
  TV / laajakaista; Taloyhtiön tilat; Vastikkeet; Vesimaksu; Lainaosuus; Tontti / lunastuslauseke; Autopaikat.

SÄÄNNÖT
- Numerot suomalaisessa muodossa (40,5 m²; 1 082,70 €)
- Jos tieto puuttuu analyysistä, jätä se mainitsematta.
- Palauta vain validia HTML:ää.


ANALYYSI:
{esite_teksti}
""".strip()
        r = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        html = (r.choices[0].message.content or "").replace("```html", "").replace("```", "").strip()
        html = re.sub(r"\s+\n", "\n", html).strip().replace("<ul></ul>", "").replace("<ul>\n</ul>", "")

        # Riskifraasien poisto
        FORBIDDEN_REGEXPS = [
            r"sijoittaj\w*",
            r"ensiasunnon\s*ostaj\w*",
            r"lapsiperhe\w*",
            r"täydellinen\s+kohde",
            r"ihanteellinen\s+kohde",
            r"unelma(kohde|koti)",
            r"täydellinen\s+valinta",
        ]
        for rexp in FORBIDDEN_REGEXPS:
            html = re.sub(rexp, "", html, flags=re.IGNORECASE)

        if not html or len(resub := re.sub(r"<[^>]*>", "", html).strip()) < 20:
            return {"error": "Myynti-ilmoituksen luonti epäonnistui: tyhjä tai vajaa sisältö."}

        return {"ilmoitus": html}
    except Exception as e:
        return {"error": f"Myynti-ilmoituksen luonti epäonnistui: {str(e)}"}

@app.post("/arvioi_hinta_tarkka")
async def arvioi_hinta_tarkka(data: dict = Body(...)):
    esite = data.get("esite", "")
    sijainti = data.get("sijainti", "")
    pinta_ala = data.get("pinta_ala")

    # Parsitaan pinta-ala numeroksi
    if isinstance(pinta_ala, str):
        m = re.search(r"[\d,.]+", pinta_ala)
        if m:
            try:
                pinta_ala = float(m.group(0).replace(",", "."))
            except ValueError:
                pass

    # Poimi postinumero osoitteesta
    sijainti_parsittu = sijainti.strip()
    zip_match = re.search(r"\b(\d{5})\b", sijainti_parsittu)
    if zip_match:
        sijainti_parsittu = zip_match.group(1)

    if not esite or not sijainti or not pinta_ala:  # noqa: E701 (pidetään alkuperäinen tyyli)
        return {"error": "Puuttuvia tietoja."}

    # Kunto
    try:
        kunto = arvioi_kunto(esite)
    except Exception as e:
        print("WARN: arvioi_kunto kaatui -> 'hyvä':", e)
        kunto = "hyvä"
    if kunto not in ["erinomainen", "hyvä", "tyydyttävä", "heikko"]:
        kunto = "hyvä"

    # Hinnat
    mediaani, vertailumaara = hae_aluehinta(sijainti_parsittu)
    if not mediaani:
        return {"error": "Ei löydetty markkinahintaa alueelta."}

    laskettu = laske_kokonaishinta(pinta_ala, mediaani, kunto)
    if not laskettu.get("arvio"):
        return {"error": "Laskenta epäonnistui."}

    selitys = generoi_hinta_selitys(
        sijainti, pinta_ala, kunto, mediaani,
        laskettu["arvio"], laskettu["haarukka"], vertailumaara
    )

    return {
        "arvio": laskettu["arvio"],
        "haarukka": laskettu["haarukka"],
        "mediaani_hinta_m2": mediaani,
        "kunto": kunto,
        "selitys": selitys
    }

@app.post("/chat")
async def chat(data: dict):
    q = data.get("question", "")
    ctx = data.get("context", "")
    if not q or not ctx:
        return {"vastaus": "Kysymys tai sisältö puuttuu."}
    prompt = f"""
Vastaa seuraavaan kysymykseen vain annetun analyysin perusteella.

--- ANALYYSI ---
{ctx}
---------------

KYSYMYS:
{q}
""".strip()
    try:
        r = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"vastaus": r.choices[0].message.content.strip()}
    except Exception:
        return {"vastaus": "Virhe vastauksessa."}

# ===================== ASYNC (latauspalkki) =====================
def _job_analyze(job_id: str, files: list[tuple[str, bytes]]):
    try:
        _set_progress(job_id, percent=5, stage="Luetaan tiedostoja")
        full_text = ""
        for idx, (fname, data) in enumerate(files, start=1):
            _set_progress(job_id, percent=5 + int(10 * idx / max(1, len(files))), stage=f"Tekstin poiminta: {fname}")
            teksti = extract_text_from_pdf(BytesIO(data))
            full_text += f"\n\n[DOKUMENTTI {idx} – {fname}]\n{teksti}"

        if not full_text.strip():
            _set_progress(job_id, done=True, error="PDF-tiedostoista ei löytynyt luettavaa tekstiä.")
            return

        _set_progress(job_id, percent=25, stage="Analysoidaan tekstiä")
        result = build_analysis_html_and_facts(full_text)

        _set_progress(job_id, percent=100, stage="Valmis", done=True, result={"esite": result["esite"]})
    except Exception as e:
        _set_progress(job_id, done=True, error=f"Virhe analyysissä: {e}")

def _job_generate_ilmoitus(job_id: str, esite_html: str):
    try:
        _set_progress(job_id, percent=10, stage="Valmistellaan analyysiä")
        esite_teksti = strip_html(esite_html)
        _set_progress(job_id, percent=35, stage="Generoidaan myynti-ilmoitusta")
        prompt = f"""
Kirjoita SUOMEKSI erittäin ammattimainen ja houkutteleva asunnon myynti-ilmoitus alla olevan analyysin pohjalta.
Tyyli on kokeneen kiinteistönvälittäjän: lämmin, asiantunteva, vakuuttava ja täysin totuudenmukainen.
Käytä vain analyysissä olevia tietoja — älä arvaa tai lisää mitään.

RAKENNE JA TYYLI
- Pituus: 4–5 kappaletta, jokainen 3–5 virkettä.
- Käytä selkeitä <p>...</p> -tageja jokaiselle kappaleelle.
- Jokaisella kappaleella oma teema:
  1) Yleiskuva: sijainti + tärkein myyntivaltti
  2) Sisätilat ja pohjaratkaisu
  3) Taloyhtiön varustelu ja yhteiset tilat
  4) Sijainti, palvelut ja ympäristö
  5) Yhteenveto ja kutsu tutustumaan
- “Arki” sana saa esiintyä korkeintaan kerran.
- Sulauta faktat luontevasti tekstiin, vältä listamaisuutta ja toistoa.
- Älä käytä ylilatautuneita fraaseja kuten “täydellinen koti” tai “unelmakoti”.

FAKTAOSIO
- Tekstin jälkeen lisää <ul><li>...</li></ul>-osio, jossa tiedot seuraavassa järjestyksessä (vain jos analyysissä löytyy):
  Osoite; Huonejako; Pinta-ala; Kerros ja hissi; Rakennustyyppi / -vuosi; Lämmitystapa;
  TV / laajakaista; Taloyhtiön tilat; Vastikkeet; Vesimaksu; Lainaosuus; Tontti / lunastuslauseke; Autopaikat.

SÄÄNNÖT
- Numerot suomalaisessa muodossa (40,5 m²; 1 082,70 €)
- Jos tieto puuttuu analyysistä, jätä se mainitsematta.
- Palauta vain validia HTML:ää.

ANALYYSI:
{esite_teksti}
""".strip()
        r = client.chat.completions.create(model=GEN_MODEL, messages=[{"role": "user", "content": prompt}])
        html = (r.choices[0].message.content or "").replace("```html", "").replace("```", "").strip()
        html = re.sub(r"\s+\n", "\n", html).strip().replace("<ul></ul>", "").replace("<ul>\n</ul>", "")
        _set_progress(job_id, percent=100, stage="Valmis", done=True, result={"ilmoitus": html})
    except Exception as e:
        _set_progress(job_id, done=True, error=f"Myynti-ilmoituksen luonti epäonnistui: {e}")

def _job_hinta_arvio(job_id: str, esite: str, sijainti: str, pinta_ala):
    try:
        _set_progress(job_id, percent=10, stage="Valmistellaan laskentaa")
        if isinstance(pinta_ala, str):
            m = re.search(r"[\d,.]+", pinta_ala)
            if m:
                try:
                    pinta_ala = float(m.group(0).replace(",", "."))
                except ValueError:
                    pass

        sijainti_parsittu = sijainti.strip()
        zip_match = re.search(r"\b(\d{5})\b", sijainti_parsittu)
        if zip_match:
            sijainti_parsittu = zip_match.group(1)

        if not esite or not sijainti or not pinta_ala:
            _set_progress(job_id, done=True, error="Puuttuvia tietoja.")
            return

        _set_progress(job_id, percent=30, stage="Arvioidaan kunto")
        try:
            kunto = arvioi_kunto(esite)
        except Exception as e:
            print("WARN: arvioi_kunto kaatui -> 'hyvä':", e)
            kunto = "hyvä"
        if kunto not in ["erinomainen", "hyvä", "tyydyttävä", "heikko"]:
            kunto = "hyvä"

        _set_progress(job_id, percent=50, stage="Haetaan aluehinta")
        mediaani, vertailumaara = hae_aluehinta(sijainti_parsittu)
        if not mediaani:
            _set_progress(job_id, done=True, error="Ei löydetty markkinahintaa alueelta.")
            return

        _set_progress(job_id, percent=70, stage="Lasketaan arvio")
        laskettu = laske_kokonaishinta(pinta_ala, mediaani, kunto)
        if not laskettu.get("arvio"):
            _set_progress(job_id, done=True, error="Laskenta epäonnistui.")
            return

        _set_progress(job_id, percent=85, stage="Kirjoitetaan selitys")
        selitys = generoi_hinta_selitys(sijainti, pinta_ala, kunto, mediaani, laskettu["arvio"], laskettu["haarukka"], vertailumaara)

        _set_progress(job_id, percent=100, stage="Valmis", done=True, result={
            "arvio": laskettu["arvio"],
            "haarukka": laskettu["haarukka"],
            "mediaani_hinta_m2": mediaani,
            "kunto": kunto,
            "selitys": selitys
        })
    except Exception as e:
        _set_progress(job_id, done=True, error=f"Hinta-arvion generointi epäonnistui: {e}")

# --------- KOKO PUTKI – yksi nappi (LISÄTTY) ---------
def _job_full_pipeline(job_id: str, files: list[tuple[str, bytes]]):
    try:
        _set_progress(job_id, percent=5, stage="Luetaan tiedostoja")
        full_text = ""
        for idx, (fname, data) in enumerate(files, start=1):
            _set_progress(job_id, percent=5 + int(10 * idx / max(1, len(files))), stage=f"Tekstin poiminta: {fname}")
            teksti = extract_text_from_pdf(BytesIO(data))
            full_text += f"\n\n[DOKUMENTTI {idx} – {fname}]\n{teksti}"

        if not full_text.strip():
            _set_progress(job_id, done=True, error="PDF-tiedostoista ei löytynyt luettavaa tekstiä.")
            return

        # 1) Analyysi
        _set_progress(job_id, percent=25, stage="Analysoidaan tekstiä")
        res = build_analysis_html_and_facts(full_text)
        esite_html = res.get("esite", "")
        facts = res.get("facts", {}) or {}

        # 2) Myynti-ilmoitus
        _set_progress(job_id, percent=55, stage="Generoidaan myynti-ilmoitus")
        esite_teksti = strip_html(esite_html)
        prompt = f"""
Kirjoita SUOMEKSI erittäin ammattimainen ja houkutteleva asunnon myynti-ilmoitus alla olevan analyysin pohjalta.
Tyyli on kokeneen kiinteistönvälittäjän: lämmin, asiantunteva, vakuuttava ja täysin totuudenmukainen.
Käytä vain analyysissä olevia tietoja — älä arvaa tai lisää mitään.

RAKENNE JA TYYLI
- Pituus: 4–5 kappaletta, jokainen 3–5 virkettä.
- Käytä selkeitä <p>...</p> -tageja jokaiselle kappaleelle.
- Jokaisella kappaleella oma teema:
  1) Yleiskuva: sijainti + tärkein myyntivaltti
  2) Sisätilat ja pohjaratkaisu
  3) Taloyhtiön varustelu ja yhteiset tilat
  4) Sijainti, palvelut ja ympäristö
  5) Yhteenveto ja kutsu tutustumaan
- “Arki” sana saa esiintyä korkeintaan kerran.
- Sulauta faktat luontevasti tekstiin, vältä listamaisuutta ja toistoa.
- Älä käytä ylilatautuneita fraaseja kuten “täydellinen koti” tai “unelmakoti”.

FAKTAOSIO
- Tekstin jälkeen lisää <ul><li>...</li></ul>-osio, jossa tiedot seuraavassa järjestyksessä (vain jos analyysissä löytyy):
  Osoite; Huonejako; Pinta-ala; Kerros ja hissi; Rakennustyyppi / -vuosi; Lämmitystapa;
  TV / laajakaista; Taloyhtiön tilat; Vastikkeet; Vesimaksu; Lainaosuus; Tontti / lunastuslauseke; Autopaikat.

SÄÄNNÖT
- Numerot suomalaisessa muodossa (40,5 m²; 1 082,70 €)
- Jos tieto puuttuu analyysistä, jätä se mainitsematta.
- Palauta vain validia HTML:ää.

ANALYYSI:
{esite_teksti}
""".strip()
        r = client.chat.completions.create(model=GEN_MODEL, messages=[{"role": "user", "content": prompt}])
        ilmoitus_html = (r.choices[0].message.content or "").replace("```html", "").replace("```", "").strip()
        ilmoitus_html = re.sub(r"\s+\n", "\n", ilmoitus_html).strip().replace("<ul></ul>", "").replace("<ul>\n</ul>", "")

        # 3) Hinta-arvio
        _set_progress(job_id, percent=75, stage="Lasketaan hinta-arvio")
        sijainti = (facts.get("osoite") or "").strip()
        pinta_ala_str = facts.get("pinta_ala_m2") or ""
        pinta_ala = _parse_pinta_ala_to_float(pinta_ala_str)

        sijainti_parsittu = sijainti
        m_zip = re.search(r"\b(\d{5})\b", sijainti_parsittu)
        if m_zip:
            sijainti_parsittu = m_zip.group(1)

        try:
            kunto = arvioi_kunto(esite_html or ilmoitus_html)
        except Exception:
            kunto = "hyvä"
        if kunto not in ["erinomainen", "hyvä", "tyydyttävä", "heikko"]:
            kunto = "hyvä"

        mediaani, vertailumaara = hae_aluehinta(sijainti_parsittu) if sijainti_parsittu else (None, 0)
        if mediaani and pinta_ala:
            laskettu = laske_kokonaishinta(pinta_ala, mediaani, kunto)
            if laskettu.get("arvio"):
                selitys = generoi_hinta_selitys(
                    sijainti or sijainti_parsittu, pinta_ala, kunto, mediaani,
                    laskettu["arvio"], laskettu["haarukka"], vertailumaara
                )
                hinta_payload = {
                    "arvio": laskettu["arvio"],
                    "haarukka": laskettu["haarukka"],
                    "mediaani_hinta_m2": mediaani,
                    "kunto": kunto,
                    "selitys": selitys
                }
            else:
                hinta_payload = {"error": "Laskenta epäonnistui."}
        else:
            hinta_payload = {"error": "Hinta-arvion edellyttämät tiedot puuttuvat (sijainti tai pinta-ala)."}

        _set_progress(job_id, percent=100, stage="Valmis", done=True, result={
            "esite": esite_html,
            "ilmoitus": ilmoitus_html,
            "hinta": hinta_payload,
            "facts": facts
        })
    except Exception as e:
        _set_progress(job_id, done=True, error=f"Koko putki epäonnistui: {e}")

# --------- Async endpointit ---------
@app.post("/analyze_async")
async def analyze_async(pdfs: List[UploadFile] = File(...)):
    job_id = str(uuid.uuid4())
    files = []
    for f in pdfs:
        files.append((f.filename, await f.read()))
    _set_progress(job_id, percent=1, stage="Jonossa")
    BG_EXECUTOR.submit(_job_analyze, job_id, files)
    return {"job_id": job_id}

@app.post("/generate_ilmoitus_async")
async def generate_listing_async(data: dict = Body(...)):
    job_id = str(uuid.uuid4())
    esite_html = data.get("esite", "")
    if not esite_html:
        return {"error": "Puuttuva esite."}
    _set_progress(job_id, percent=1, stage="Jonossa")
    BG_EXECUTOR.submit(_job_generate_ilmoitus, job_id, esite_html)
    return {"job_id": job_id}

@app.post("/arvioi_hinta_tarkka_async")
async def arvioi_hinta_tarkka_async(data: dict = Body(...)):
    job_id = str(uuid.uuid4())
    _set_progress(job_id, percent=1, stage="Jonossa")
    BG_EXECUTOR.submit(_job_hinta_arvio, job_id, data.get("esite", ""), data.get("sijainti", ""), data.get("pinta_ala"))
    return {"job_id": job_id}

# --------- Yksi nappi koko putkelle (LISÄTTY) ---------
@app.post("/run_all_async")
async def run_all_async(pdfs: List[UploadFile] = File(...)):
    job_id = str(uuid.uuid4())
    files = []
    for f in pdfs:
        files.append((f.filename, await f.read()))
    _set_progress(job_id, percent=1, stage="Jonossa")
    BG_EXECUTOR.submit(_job_full_pipeline, job_id, files)
    return {"job_id": job_id}

@app.get("/job_status/{job_id}")
def job_status(job_id: str):
    with PROGRESS_LOCK:
        st = PROGRESS.get(job_id)
    if not st:
        return {"error": "Tuntematon job_id"}
    return st

@app.get("/job_result/{job_id}")
def job_result(job_id: str):
    with PROGRESS_LOCK:
        st = PROGRESS.get(job_id)
    if not st:
        return {"error": "Tuntematon job_id"}
    if not st.get("done"):
        return {"error": "Ei vielä valmis"}
    result = st.get("result")
    PROGRESS.pop(job_id, None)
    if st.get("error"):
        return {"error": st["error"]}
    return result or {"error": "Tulosta ei ole"}

# --- PDF export + tallennus levyille ---

EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

@app.post("/export_pdf")
def export_pdf(payload: dict = Body(...)):
    esite_html = payload.get("esite", "")
    ilmoitus_html = payload.get("ilmoitus", "")
    hinta_html = payload.get("hintaArvio", "")
    otsikko = (payload.get("otsikko") or "Myyntiapuri").strip()

    html = f"""
    <html><head><meta charset="utf-8" />
      <style>
        @page {{ size: A4; margin: 10mm; }}
        body {{ font-family: -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif; color:#111; }}
        h1 {{ font-size: 20px; margin: 0 0 12px; }}
        h2 {{ font-size: 16px; margin: 16px 0 8px; }}
        .section {{ page-break-inside: avoid; margin: 8px 0 16px; }}
        ul {{ padding-left: 18px; }} li {{ margin: 4px 0; }}
        .meta {{ color:#555; font-size:12px; margin-bottom:12px; }}
        hr {{ border:0; border-top:1px solid #ddd; margin: 12px 0; }}
      </style>
    </head><body>
      <h1>Myyntiapuri – Myyntiaineisto</h1>
      <div class="meta">{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
      <div class="section"><h2>📋 Analyysi</h2>{esite_html}</div><hr/>
      <div class="section"><h2>📝 Myynti‑ilmoitus</h2>{ilmoitus_html}</div><hr/>
      <div class="section"><h2>💰 Hinta‑arvio</h2>{hinta_html}</div>
    </body></html>
    """

    # Playwright: generoi PDF tavubytes-muodossa
    pdf_bytes = html_to_pdf_bytes(html)

    # 1) palauta ladattavaksi
    safe = otsikko.replace(" ", "_")
    filename = f"{safe}_{datetime.now().strftime('%Y-%m-%d')}.pdf"
    # 2) tallenna myös palvelimelle
    save_path = os.path.join(EXPORT_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(pdf_bytes)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
# --- Pilveen tallennus (S3-optio) ---
def _maybe_upload_to_s3(filename: str, data: bytes) -> str | None:
    bucket = os.getenv("AWS_S3_BUCKET")
    region = os.getenv("AWS_REGION")
    access = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not all([bucket, region, access, secret]):
        return None
    try:
        import boto3
        s3 = boto3.client("s3", region_name=region,
                          aws_access_key_id=access,
                          aws_secret_access_key=secret)
        key = f"smartbroker/{filename}"
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="application/pdf")
        return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    except Exception as e:
        print("⚠️ S3 upload failed:", e)
        return None

@app.post("/tallenna_myyntiaineisto")
def tallenna_myyntiaineisto(payload: dict = Body(...)):
    esite_html = payload.get("esite", "")
    ilmoitus_html = payload.get("ilmoitus", "")
    hinta_html = payload.get("hintaArvio", "")
    otsikko = (payload.get("otsikko") or "Myyntiapuri").strip()

    html = f"""
    <html><head><meta charset="utf-8" />
      <style>
        @page {{ size: A4; margin: 10mm; }}
        body {{ font-family: -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif; color:#111; }}
        h1 {{ font-size: 20px; margin: 0 0 12px; }}
        h2 {{ font-size: 16px; margin: 16px 0 8px; }}
        .section {{ page-break-inside: avoid; margin: 8px 0 16px; }}
        ul {{ padding-left: 18px; }} li {{ margin: 4px 0; }}
        .meta {{ color:#555; font-size:12px; margin-bottom:12px; }}
        hr {{ border:0; border-top:1px solid #ddd; margin: 12px 0; }}
      </style>
    </head><body>
      <h1>Myyntiapuri – Myyntiaineisto</h1>
      <div class="meta">{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
      <div class="section"><h2>📋 Analyysi</h2>{esite_html}</div><hr/>
      <div class="section"><h2>📝 Myynti‑ilmoitus</h2>{ilmoitus_html}</div><hr/>
      <div class="section"><h2>💰 Hinta‑arvio</h2>{hinta_html}</div>
    </body></html>
    """

    pdf_bytes = html_to_pdf_bytes(html)

    safe = otsikko.replace(" ", "_")
    filename = f"{safe}_{datetime.now().strftime('%Y-%m-%d')}.pdf"

    # tallenna palvelimelle
    save_path = os.path.join(EXPORT_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(pdf_bytes)

    # yritä myös S3:een (jos .env on asetettu)
    cloud_url = _maybe_upload_to_s3(filename, pdf_bytes)

    return {"status": "ok",
            "filename": filename,
            "local_url": f"/download_export/{filename}",
            "cloud_url": cloud_url}  # None jos S3 ei konffattu

@app.get("/exports")
def list_exports():
    """Palauta palvelimelle tallennetut PDF:t listana."""
    files = []
    for name in sorted(os.listdir(EXPORT_DIR)):
        if name.lower().endswith(".pdf"):
            files.append({"filename": name, "url": f"/download_export/{name}"})
    return files

from fastapi.responses import FileResponse

@app.get("/download_export/{filename}")
def download_export(filename: str):
    path = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path, media_type="application/pdf", filename=filename)
''
