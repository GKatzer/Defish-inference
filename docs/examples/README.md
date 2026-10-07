# Examples

Scripts, photos and recorded output used by the documentation. Everything runs against a service started as in the [quick start](../../README.md#quick-start). The recordings were made with the **real** detector and classifier files, which are kept outside the repository (the quick start uses stand-ins, so its boxes and classes differ from the recordings); the one exception is `fixes-before-after.txt`, recorded with the real detector and the **stand-in** classifier (its classes are meaningless), because it compares the original code with the current code.

| File | What it is |
|---|---|
| [make_standin_models.py](make_standin_models.py) | writes stand-in classifier files, and with `--detector` a stand-in detector with three fixed boxes (the tests build their own), with the same names and contract as the real ones |
| [walkthrough.sh](walkthrough.sh), [pretty.py](pretty.py) | every route once with curl; `pretty.py` cuts base64, rounds floats and keeps the first detections |
| [probe_service.py](probe_service.py) | black-box checks of a running service in groups (health, bad input, colours, photo, limiter); the `limiter` group sends `--limit` + 2 requests (default 60 + 2) and expects the last two to be refused |
| [measure_latency.py](measure_latency.py) | wall-clock time of `POST /detect` per photo |
| [analyze_summary.py](analyze_summary.py) | what `POST /analyze` says about each photo: fish, time, classes, how many flagged `uncertain` |
| [measure_classifier.py](measure_classifier.py) | needs the **real** files, in-process: the current one-call-per-crop classifier against the previous one-call-for-all-crops way (`--threads`, `--scan`), and comparison with the `Defish-ML-train` reference runtime (`--reference`; `--opencv-resize` puts the previous resize back) |
| [health_under_load.py](health_under_load.py) | `GET /health` latency while 8 requests of `/detect` or `/analyze` run |
| [annotate.py](annotate.py) | draws the boxes returned by `/detect` (scores) or, with `--classes`, by `/analyze` (classes, orange when `uncertain`) on a photo |
| [photos/](photos/) | four Creative Commons / public-domain photographs, sources and licences in [../media/CREDITS.md](../media/CREDITS.md) |
| [transcripts/](transcripts/) | recorded output: `walkthrough.txt`, `analyze-summary.txt` (and `analyze-summary-before-classifier-changes.txt`), `classifier-measurements.txt`, `latency.txt`, `health-under-load.txt` (real files, 2026-10-05; `latency.txt` 2026-10-04), `fixes-before-after.txt` (original code against the current code, stand-in classifier, 2026-10-04), `docker-compose.txt`, `tests.txt` |

The recordings were made on 2026-10-04 and 2026-10-05; each transcript starts with the date, the machine and the versions.
