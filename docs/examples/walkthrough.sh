#!/usr/bin/env bash
# Every endpoint once, with curl, against a running service. The answers go through pretty.py: base64 cut with `...`,
# floats rounded to 3 decimals, only the first detections shown.
# usage: BASE=http://127.0.0.1:8000 bash docs/examples/walkthrough.sh [photo.jpg]
# The default limit is 60 requests per 60 s per client address; the walkthrough sends 5 limited requests.
set -u
BASE=${BASE:-http://127.0.0.1:8000}
PHOTO=${1:-docs/examples/photos/planted-tank-tetras.jpg}


echo '$ curl -s '"$BASE"'/health'
curl -s "$BASE/health" | python3 docs/examples/pretty.py; echo
echo '$ curl -s '"$BASE"'/model-status'
curl -s "$BASE/model-status" | python3 docs/examples/pretty.py; echo
echo '$ curl -s -F image=@'"$PHOTO"' '"$BASE"'/detect'
curl -s -F "image=@$PHOTO" "$BASE/detect" | python3 docs/examples/pretty.py 2; echo
echo '$ printf '"'"'{"image_bytes":"%s"}'"'"' "$(base64 -w0 '"$PHOTO"')" | curl -s -H "Content-Type: application/json" -d @- '"$BASE"'/analyze'
printf '{"image_bytes":"%s"}' "$(base64 -w0 "$PHOTO")" | curl -s -H "Content-Type: application/json" -d @- "$BASE/analyze" | python3 docs/examples/pretty.py 1; echo
echo '$ curl -s -w "  [HTTP %{http_code}]\n" -H "Content-Type: application/json" -d "{}" '"$BASE"'/analyze'
curl -s -w "  [HTTP %{http_code}]\n" -H "Content-Type: application/json" -d '{}' "$BASE/analyze"; echo
echo '$ curl -s -w "  [HTTP %{http_code}]\n" -F image=@README.md '"$BASE"'/detect'
curl -s -w "  [HTTP %{http_code}]\n" -F "image=@README.md;type=text/markdown" "$BASE/detect"; echo
echo '$ curl -s -w "  [HTTP %{http_code}]\n" -X POST '"$BASE"'/detect'
curl -s -w "  [HTTP %{http_code}]\n" -X POST "$BASE/detect" | cut -c1-200
