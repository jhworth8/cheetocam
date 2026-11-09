#!/usr/bin/env bash
# Porch camera: real audio from the Bulb Cam, and fewer bytes per frame so the
# tunnel can carry more of them.
#
# Run ON THE PI, after copying the new live_stream.py into ~/cheetocam/.
# Idempotent. Backs up what it touches.

set -uo pipefail
cd "$HOME/cheetocam" || { echo "no ~/cheetocam — wrong host?"; exit 1; }
STAMP="$(date +%Y%m%d-%H%M%S)"

[ -f live_stream.py ] || { echo "live_stream.py missing"; exit 1; }
python3 -c "import ast,sys;ast.parse(open('live_stream.py').read())" \
  || { echo "live_stream.py does not parse — not restarting anything."; exit 1; }
echo "live_stream.py parses OK"

cp -a .env.porch-cam ".env.porch-cam.bak.$STAMP"
echo "Backed up .env.porch-cam -> .env.porch-cam.bak.$STAMP"

set_key() {
  local f="$1" k="$2" v="$3"
  if grep -qE "^${k}=" "$f"; then sed -i "s|^${k}=.*|${k}=${v}|" "$f"
  else printf '%s=%s\n' "$k" "$v" >> "$f"; fi
}

echo
echo "== Porch: audio from the camera's own stream =="
set_key .env.porch-cam STREAM_AUDIO_ENABLED '"1"'
set_key .env.porch-cam STREAM_AUDIO_SOURCE  '"rtsp://192.168.86.113:18554/porch_cam"'

echo "== Porch: smaller frames so more of them survive the tunnel =="
# 640x360 at q70 is ~28KB; at q45 and 480 wide it is nearer 9KB. Same frames,
# roughly a third of the bytes, which is what the uplink is actually short of.
set_key .env.porch-cam STREAM_JPEG_QUALITY '"45"'
set_key .env.porch-cam STREAM_MAX_WIDTH    '"480"'

# The front door is untouched on purpose: it is on the LAN-side USB camera with
# its own microphone, and the new defaults reproduce its old behaviour exactly.

echo
echo "== Restarting =="
sudo systemctl restart cat_detector cat_detector@porch-cam
sleep 20
for u in cat_detector cat_detector@porch-cam; do
  printf '  %-26s %s\n' "$u" "$(systemctl is-active "$u")"
done

TOKEN=$(grep -E '^STREAM_TOKEN=' .env | cut -d= -f2- | tr -d '"')

echo
echo "== Frame rate and bytes, measured locally =="
for p in 8088 8089; do
  timeout 10 curl -s "http://127.0.0.1:$p/stream.mjpg?token=$TOKEN" > "/tmp/s$p.mjpg" 2>/dev/null
  python3 - "$p" <<'PY'
import sys, os
p = sys.argv[1]
try:
    d = open(f"/tmp/s{p}.mjpg","rb").read()
    n = d.count(b"\xff\xd8\xff")
    kb = (len(d)/n/1024) if n else 0
    print(f"  port {p}: {n} frames / 10s  ~{round(n/10,1)} fps  ~{kb:.1f} KB per frame")
except Exception as e:
    print(f"  port {p}: no stream ({e})")
PY
done

echo
echo "== Audio: is the porch endpoint actually producing bytes? =="
BYTES=$(timeout 8 curl -s "http://127.0.0.1:8089/audio.mp3?token=$TOKEN" | wc -c)
echo "  porch /audio.mp3 returned $BYTES bytes in 8s"
if [ "$BYTES" -gt 4000 ]; then
  echo "  -> audio is flowing from the Bulb Cam's stream"
else
  echo "  -> little or nothing. Check: journalctl -u cat_detector@porch-cam -n 40"
fi

echo
uptime | sed 's/^/  /'
free -h | sed 's/^/  /'

cat <<'NOTES'

Rollback:
  cp ~/cheetocam/live_stream.py.bak.<stamp> ~/cheetocam/live_stream.py
  cp ~/cheetocam/.env.porch-cam.bak.<stamp> ~/cheetocam/.env.porch-cam
  sudo systemctl restart cat_detector cat_detector@porch-cam

Front door is deliberately unchanged — STREAM_JPEG_QUALITY and STREAM_MAX_WIDTH
default to the values that were hardcoded before, so it streams exactly as it did.
NOTES
