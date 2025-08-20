from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY puuttuu .env-tiedostosta")

client = OpenAI(api_key=api_key)

def generoi_hinta_selitys(sijainti, pinta_ala, kunto, mediaani, arvio, haarukka, vertailumaara):
    """
    Palauttaa selkeän, ammattimaisen ja neutraalin HTML-selityksen hinta-arviolle.
    Tuottaa 3–4 kappaletta <p>-tageilla. Ei keksi uusia lukuja, ei spekuloi markkinatilanteesta yli annetun datan.
    """
    prompt = f"""
Laadi SUOMEKSI selkeä, asiantunteva ja tiiviisti jäsennelty HTML-selitys alla annettujen tietojen pohjalta.
Käytä vain annettuja arvoja; älä lisää lukuja tai vahvoja väitteitä, joita ei ole datassa.

TYYLI JA RAKENNE
- Palauta VAIN validia HTML:ää, jossa on 3–4 erillistä <p>…</p>-kappaletta (ei otsikoita).
- Avaus: ilmoita arvioitu hinta ja hintahaarukka napakasti, käytä <strong>-lihavointeja arvolle ja haarukalle.
- Perustelu: kerro mihin arvio tukeutuu (vertailukohteiden määrä {vertailumaara}, alueen mediaanihinta {mediaani} €/m², pinta-ala {pinta_ala} m², sijainti {sijainti}).
- Kunto: selitä neutraalisti, miten kunto "{kunto}" vaikuttaa suhteessa mediaanitasoon (nostaa/laskee/neutralisoi) ilman numeerista arvausta.
- Vaihteluväli: kerro lyhyesti mistä haarukan sisäinen vaihtelu voi johtua yleisellä tasolla (esim. taloyhtiön ja rakennuksen kunto/ikä, pohjaratkaisu ja varustelu kuten parveke/hissi, näkymät/melutaso). Älä väitä näiden olemassaoloa ellei niitä ole annettu; puhu yleisellä tasolla.
- Vältä toisteisuutta ja markkinahypeä. Kirjoita neutraalilla, ammattilaisen äänellä.

ANNETUT TIEDOT (käytä tekstissä):
- Arvioitu hinta: {arvio} €
- Hintahaarukka: {haarukka}
- Pinta-ala: {pinta_ala} m²
- Kunto: {kunto}
- Alueen mediaanihinta: {mediaani} €/m²
- Vertailukohtien määrä: {vertailumaara}
- Sijainti (kaupunginosa tai postinumero): {sijainti}
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Selityksen generointi epäonnistui:", e)
        return (
            "<p><strong>Arvioitu hinta:</strong> "
            f"{arvio} € — <strong>Hintahaarukka:</strong> {haarukka}.</p>"
            "<p>Selityksen tuottaminen epäonnistui. Tiedot perustuvat annettuun mediaanihintaan, "
            f"vertailukohtien määrään ({vertailumaara}) ja kohteen kokoon ({pinta_ala} m²) sijainnissa {sijainti}.</p>"
        )
