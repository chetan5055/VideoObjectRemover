from pathlib import Path
from re import T


ROOT = Path(__file__).parent.parent


RESOUCRES_DIR = ROOT / "resources"
WATER_MARK_TEMPLATE_IMAGE_PATH = RESOUCRES_DIR / "watermark_template.png"

OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
