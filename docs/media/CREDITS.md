# Media credits

Every photograph in this repository comes from Wikimedia Commons and is public domain, CC0 or CC BY (attribution below); nothing is CC BY-SA, nothing comes from a forum or from the project's own training data. The licences were read from the Commons file pages on 2026-10-04, when the files were downloaded. All other media are screenshots or pictures made by the scripts in [`../examples/`](../examples/).

## Photographs (`docs/examples/photos/`)

Each file was resized to at most 1600 pixels on the longer side and re-encoded as JPEG (quality 88), except `cardinal-tetra-school.jpg`, which is the unchanged original (801×686).

| File here | Commons file | Author | Licence |
|---|---|---|---|
| `planted-tank-tetras.jpg` | [File:Amaterske akvarium.jpg](https://commons.wikimedia.org/wiki/File:Amaterske_akvarium.jpg) (a small amateur aquarium, 100 litres) | User Aleš Tošovský | public domain |
| `guppies-planted-tank.jpg` | [File:Маленький заросший аквариум.jpg](https://commons.wikimedia.org/wiki/File:%D0%9C%D0%B0%D0%BB%D0%B5%D0%BD%D1%8C%D0%BA%D0%B8%D0%B9_%D0%B7%D0%B0%D1%80%D0%BE%D1%81%D1%88%D0%B8%D0%B9_%D0%B0%D0%BA%D0%B2%D0%B0%D1%80%D0%B8%D1%83%D0%BC.jpg) | MarDe | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0). Changes: resized and re-compressed |
| `cardinal-tetra-school.jpg` | [File:Paracheirodon axelrodi school.jpg](https://commons.wikimedia.org/wiki/File:Paracheirodon_axelrodi_school.jpg) | Tkinias (English Wikipedia) | public domain |
| `endler-guppies-dense-shoal.jpg` | [File:Poecilia wingei Cologne Zoo 2.jpg](https://commons.wikimedia.org/wiki/File:Poecilia_wingei_Cologne_Zoo_2.jpg) (Endler's guppies in an aquarium of the Cologne Zoo) | TomCatX | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |

## Pictures made from them (`docs/media/`)

| File | What it is |
|---|---|
| `detect-planted-tank-tetras.jpg`, `detect-guppies-planted-tank.jpg`, `detect-cardinal-tetra-school.jpg`, `detect-endler-guppies-dense-shoal.jpg` | the photo above with the boxes the **real detector** (`models/best.onnx`, a file that is not in the repository) returned from `POST /detect`, drawn by [`../examples/annotate.py`](../examples/annotate.py) on 2026-10-04; the number is the detector score. Shown at most 1200 pixels wide. They carry the licence of the photo they are made from (for `detect-guppies-planted-tank.jpg`: CC BY 4.0, MarDe, resized, with boxes drawn on it) |
| `analyze-planted-tank-tetras.jpg`, `analyze-cardinal-tetra-school.jpg` | the same photos (`planted-tank-tetras.jpg`: public domain, Aleš Tošovský; `cardinal-tetra-school.jpg`: public domain, Tkinias) with the boxes **and classes** the real detector and the real classifier returned from `POST /analyze` on 2026-10-05, drawn by [`../examples/annotate.py`](../examples/annotate.py) `--classes`: green = confident, orange = flagged `uncertain`. The true condition of these fish is not known; the pictures show what the model says, not what is true |

The `detect-*` pictures show no class names. The classes on the two `analyze-*` pictures are real model output; none of the example outputs recorded with the stand-in classifier (the `transcripts/fixes-before-after.txt` recording) is a prediction.

## Not media, but worth knowing

The detector and classifier files (not in this repository) were trained on third-party photographs that are not included in this repository and are not covered by these credits; see `Defish-ML-train` for how the data were assembled.
