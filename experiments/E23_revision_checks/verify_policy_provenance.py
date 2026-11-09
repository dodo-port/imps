"""JKSS revision check: for the E22 inputs, which policy values came from doctrine text extraction
and which from defaults/rules. Calls the module's own extraction functions on the same retrieval hits.
No module code is changed. Output: policy_provenance.json"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
APP = HERE.parents[1] / "mission_plan-new_plan_250519"
sys.path.insert(0, str(APP))
from modules.doctrine_policy import DoctrinePolicyEngine
from modules.config import MUMT_RATIO
eng = DoctrinePolicyEngine(doctrine_dir=str(APP / "data" / "doctrine"), fallback_doc=str(APP / "doctrine_basis.md"))
seq_in = ["SEAD", "STRIKE"]
threats = [{"type": t} for t in ("SAM", "RADAR", "SAM", "RADAR", "SAM")]
pol = eng.build_policy(mission_types=seq_in, threats=threats, current_margin_km=8.0)
seq = eng._normalize_mission_sequence(seq_in)
hits = eng.rag.search(query=" ".join(seq), top_k=8, min_score=0.01)
ratio_ex = eng._extract_ratio_from_hits(hits)
margin_ex = eng._extract_margin_floor_from_hits(hits)
out = {
  "policy": {k: getattr(pol, k) for k in ("mission_sequence","min_fighter","min_recon_uav","min_attack_uav","mumt_ratio",
             "utilization_rate","max_recon_to_strike_ratio","recon_bias_allowance","safety_margin_floor_km")},
  "n_hits": len(hits), "hit_sources": [(h.get("source"), h.get("page")) for h in hits],
  "extracted_mumt_ratio_from_text": ratio_ex, "config_MUMT_RATIO_default": MUMT_RATIO,
  "extracted_margin_floor_from_text": margin_ex, "input_current_margin_km": 8.0,
}
(HERE / "policy_provenance.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
