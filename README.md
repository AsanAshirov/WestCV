# WestCV — WIUT Hackathon 2026, Computer Vision Track

Traffic event detection (Part A) and accident anticipation (Part B) for a fixed road camera.

> Status (2026-09-25): analysis phase. The solution is not implemented yet; `solution.py` is still the organizers' template.

## Contents

| Path | What it is |
|---|---|
| `solution.py` | Interface we implement (currently the organizers' template) |
| `run_submission.py`, `evaluate.py` | Organizers' harness and official metric. **Do not modify** |
| `requirements.txt`, `examples/` | From the organizers' starter kit |
| `docs/starter_kit_README.md` | Organizers' starter-kit README |
| `WIUT Hackathon _ CV Track Elimination Task.pdf` | Task description |
| `Videos.pdf` | Google Drive links to the 4 sample videos (not in the repo, 2-6 GB each) |
| `WIUT_CV_Track_analysis_RU.md` | Full task analysis and strategy (Russian) |
| `WIUT_toolkit_RU.md` | Annotation tools, open-source models and datasets catalog (Russian) |
| `tools/annotation/` | Annotation converters: Label Studio → ground_truth.json, annotator disagreement report, labelme → scene geometry |
| `research/reports/` | Raw research and verification reports behind the analysis (English) |
| `research/scripts/` | Experimental scripts (metric simulations, decode benchmarks, scene registration). Not part of the solution |
| `CLAUDE.md` | Working context for Claude Code sessions |

## Quickstart (starter kit)

```bash
pip install -r requirements.txt
python run_submission.py --videos samples --out predictions_samples.json --team westcv
python evaluate.py --pred predictions_samples.json --validate-only
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
```

The final README must also list: install/run steps including how weights are obtained, the approach (models, datasets with licences, learned vs rule-based parts), seeds and non-determinism, team members and who did what.
