LANG_MAP = {
    "en": "English",
    "si": "Sinhala",
    "ta": "Tamil",
}

DEFAULT_LANGUAGE_CODE = "en"
DEFAULT_LANGUAGE_NAME = "English"


def get_selected_language(language_code: str | None) -> str:
    if not language_code:
        return DEFAULT_LANGUAGE_NAME
    return LANG_MAP.get(language_code, DEFAULT_LANGUAGE_NAME)