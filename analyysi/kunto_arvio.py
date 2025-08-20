from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY puuttuu .env-tiedostosta")

client = OpenAI(api_key=api_key)

def arvioi_kunto(esite: str) -> str:
    """Arvioi asunnon kunto myyntiesitteen perusteella (4-portainen luokitus)."""
    prompt = f"""
Analysoi seuraavan myyntiesitteen perusteella, mikä on asunnon kunto. Vastaa yhdellä sanalla.

Mahdolliset arvot ovat:
- erinomainen
- hyvä
- tyydyttävä
- heikko

Myyntiesite:
{esite}

Palauta vain yksi sana ilman lisäselityksiä.
"""
    try:
        vastaus = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}]
        )
        kunto = vastaus.choices[0].message.content.strip().lower()
        sallitut = ["erinomainen", "hyvä", "tyydyttävä", "heikko"]
        return kunto if kunto in sallitut else "hyvä"
    except Exception as e:
        print("❌ Kunnon arviointi epäonnistui:", e)
        return "hyvä"
