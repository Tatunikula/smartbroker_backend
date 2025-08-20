def laske_kokonaishinta(pinta_ala: float, mediaani_m2: float, kunto: str):
    """Laskee arviohinnan ja haarukan kunnon perusteella."""

    kunto_kertoimet = {
        "heikko": 0.85,
        "tyydyttävä": 0.95,
        "hyvä": 1.0,
        "erinomainen": 1.05
    }

    if kunto not in kunto_kertoimet:
        print(f"⚠️ Tuntematon kunto: {kunto}, käytetään neutraalia kerrointa 1.0")
        kerroin = 1.0
    else:
        kerroin = kunto_kertoimet[kunto]

    try:
        hinta_per_m2 = mediaani_m2 * kerroin
        arvio = int(round(pinta_ala * hinta_per_m2))

        vaihteluväli = 0.08  # 8 % suuntaansa
        min_hinta = int(round(arvio * (1 - vaihteluväli)))
        max_hinta = int(round(arvio * (1 + vaihteluväli)))

        return {
            "arvio": arvio,
            "haarukka": f"{min_hinta}–{max_hinta} €"
        }

    except Exception as e:
        print(f"❌ Hinnanlaskenta epäonnistui: {e}")
        return {
            "arvio": None,
            "haarukka": ""
        }
