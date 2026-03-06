import os
import requests

from surya.settings import settings


def get_font_path() -> str:
    font_path = settings.RECOGNITION_RENDER_FONTS["all"]

    if not os.path.exists(font_path):
        os.makedirs(os.path.dirname(font_path), exist_ok=True)
        font_dl_path = f"{settings.RECOGNITION_FONT_DL_BASE}/{os.path.basename(font_path)}"
        with requests.get(font_dl_path, stream=True) as r, open(font_path, 'wb') as f:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return font_path
