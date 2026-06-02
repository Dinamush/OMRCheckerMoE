"""Debug: show exact threshold values for each question."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import src.core as core_module

# Monkey-patch get_local_threshold to log its decisions
_orig_get_local_threshold = core_module.ImageInstanceOps.get_local_threshold

def _debug_get_local_threshold(self, q_vals, global_thr, no_outliers, plot_title=None, plot_show=True):
    result = _orig_get_local_threshold(self, q_vals, global_thr, no_outliers, plot_title, plot_show)
    q_arr = sorted(q_vals)
    local_jumps = np.array(q_arr[2:]) - np.array(q_arr[:-2])
    max_jump = float(np.max(local_jumps)) if len(local_jumps) > 0 else 0
    min_jump = self.tuning_config.threshold_params.MIN_JUMP
    conf_jump = min_jump + self.tuning_config.threshold_params.CONFIDENT_SURPLUS
    label = (plot_title or "?").split(".")[-2] if "." in (plot_title or "") else (plot_title or "?")
    # Only log NR-likely questions
    print(f"  [{label}] vals={[round(v,1) for v in q_arr]} max_jump={round(max_jump,1)} no_outliers={no_outliers} thr={round(result,1)} global_thr={round(global_thr,1)}")
    return result

core_module.ImageInstanceOps.get_local_threshold = _debug_get_local_threshold

from src.entry import entry_point_for_image

cfg = json.load(open("webui/storage/batches/4303aca72e01/config.json"))
print(f"MIN_JUMP={cfg['threshold_params']['MIN_JUMP']}")

image_path = Path("C:/Users/samir.mohammed/Downloads/prefilled_sheet_moderate_medium_pencil.png")
output_dir = Path(tempfile.mkdtemp(prefix="omr_dbg2_"))
template_dir = Path("webui/storage/batches/4303aca72e01")

entry_point_for_image(
    image_path=image_path,
    output_dir=output_dir,
    template_payload=json.load(open(template_dir / "template.json")),
    config_payload=cfg,
    template_dir=template_dir,
)
