"""Debug script: run OMR engine on test image and print results."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.entry import _build_tuning_config_inmemory, entry_point_for_image

cfg = json.load(open("webui/storage/batches/4303aca72e01/config.json"))
print("Config MIN_JUMP:", cfg["threshold_params"]["MIN_JUMP"])

tuning_config = _build_tuning_config_inmemory(cfg)
print("tuning_config.threshold_params.MIN_JUMP:", tuning_config.threshold_params.MIN_JUMP)

image_path = Path("C:/Users/samir.mohammed/Downloads/prefilled_sheet_moderate_medium_pencil.png")
output_dir = Path(tempfile.mkdtemp(prefix="omr_debug_"))
template_dir = Path("webui/storage/batches/4303aca72e01")
template_path = template_dir / "template.json"

print("Image exists:", image_path.exists())
print("Template exists:", template_path.exists())
print("Output dir:", output_dir)

entry_point_for_image(
    image_path=image_path,
    output_dir=output_dir,
    template_payload=json.load(open(template_path)),
    config_payload=cfg,
    template_dir=template_dir,
)

csvs = list(output_dir.glob("Results/Results_*.csv"))
print("CSVs found:", csvs)
if csvs:
    print(open(csvs[0]).read())
else:
    print("No CSV found!")
    print("Files in output_dir:", list(output_dir.rglob("*")))
