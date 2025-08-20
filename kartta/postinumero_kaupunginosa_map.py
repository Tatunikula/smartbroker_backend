POSTINUMERO_KAUPUNGINOSA = {
    "00500": "Kallio",
    "00200": "Taka-Töölö",
    "00810": "Herttoniemi",
    "00350": "Haaga",
    "00610": "Oulunkylä",
    "00100": "Kamppi",
    "00530": "Alppiharju",
    "00400": "Kannelmäki",
    "00710": "Malmi",
    "00910": "Vuosaari",
    # Lisää lisää tarvittaessa
}

def postinumero_to_kaupunginosa(postinumero: str) -> str | None:
    return POSTINUMERO_KAUPUNGINOSA.get(postinumero)
