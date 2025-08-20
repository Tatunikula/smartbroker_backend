# laskenta/hintatiedot_scraper.py  — PILVI-SAFE VERSIO (ei seleniumia)
import os, csv, re
from typing import Tuple, Optional
from kartta.postinumero_kaupunginosa_map import postinumero_to_kaupunginosa

# CSV oletetaan repo-juuren tiedostoksi: aluehinnat.csv
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "aluehinnat.csv")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _to_float(val: str) -> Optional[float]:
    if val is None:
        return None
    s = str(val).replace(" ", "").replace("\u00A0", "").replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None

def _to_int(val: str) -> int:
    if val is None:
        return 0
    s = re.sub(r"[^\d]", "", str(val))
    try:
        return int(s) if s else 0
    except ValueError:
        return 0

def _read_rows():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def hae_hintatiedot_kaupunginosalla(sijainti: str) -> Tuple[Optional[float], int]:
    """
    Palauttaa (mediaani_hinta_m2, vertailujen_maara) annetulle kaupunginosalle tai postinumerolle.
    Data luetaan paikallisesta CSV:stä -> ei mitään selainajureita.
    """
    if not sijainti:
        return (None, 0)

    s = sijainti.strip()
    if s.isdigit() and len(s) == 5:
        target = postinumero_to_kaupunginosa(s) or s
    else:
        target = s
    norm_target = _norm(target)

    rows = _read_rows()
    if not rows:
        return (None, 0)

    # yritetään ensin tarkka osuma "kaupunginosa"/"alue"
    best_mediaani, best_maara = None, 0
    for r in rows:
        nimi = _norm(r.get("kaupunginosa") or r.get("alue") or r.get("district") or "")
        if not nimi:
            continue
        if nimi == norm_target:
            best_mediaani = _to_float(r.get("mediaani") or r.get("median") or r.get("€/m2") or r.get("price_per_m2"))
            best_maara = _to_int(r.get("maara") or r.get("count") or r.get("n"))
            if best_mediaani is not None:
                return (best_mediaani, best_maara)

    # muuten löyhempi osuma (substring)
    for r in rows:
        nimi = _norm(r.get("kaupunginosa") or r.get("alue") or r.get("district") or "")
        if not nimi or norm_target not in nimi:
            continue
        m = _to_float(r.get("mediaani") or r.get("median") or r.get("€/m2") or r.get("price_per_m2"))
        if m is not None:
            best_mediaani = m
            best_maara = _to_int(r.get("maara") or r.get("count") or r.get("n"))
            break

    return (best_mediaani, best_maara)
