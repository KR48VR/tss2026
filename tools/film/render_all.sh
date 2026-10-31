#!/bin/bash
# Render every frame of a film in parallel chunks, then encode with its score.
#   tools/film/render_all.sh FILM SECONDS OUTDIR [WORKERS] [AUDIO.wav] [LUFS]
set -e
FILM=$1; SEC=$2; OUT=$3; WK=${4:-3}; AUD=$5; LUFS=${6:--14}
HERE=$(cd "$(dirname "$0")" && pwd)
N=$(python3 -c "print(round($SEC*24))"); mkdir -p "$OUT/frames"
CH=$(( (N + 5) / 6 ))
for i in 0 1 2 3 4 5; do echo "$((i*CH)) $(( (i+1)*CH < N ? (i+1)*CH : N ))"; done | \
  xargs -P "$WK" -L 1 bash -c 'python3 '"$HERE"'/render.py frames '"$FILM"' '"$OUT"'/frames $0 $1'
ls "$OUT/frames" | grep -c jpg
ffmpeg -hide_banner -loglevel error -y -framerate 24 -i "$OUT/frames/f%05d.jpg" -vf "hqdn3d=4:3:6:4.5,format=yuv420p" \
  -c:v libx264 -preset slow -crf 21 -movflags +faststart "$OUT/${FILM}_video.mp4"
if [ -n "$AUD" ]; then
  J=$(ffmpeg -hide_banner -i "$AUD" -af loudnorm=I=$LUFS:TP=-2:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
  g() { echo "$J" | python3 -c "import json,sys; print(json.load(sys.stdin)['$1'])"; }
  ffmpeg -hide_banner -loglevel error -y -i "$OUT/${FILM}_video.mp4" -i "$AUD" -map 0:v -map 1:a -c:v copy \
    -af "loudnorm=I=$LUFS:TP=-2:LRA=11:measured_I=$(g input_i):measured_TP=$(g input_tp):measured_LRA=$(g input_lra):measured_thresh=$(g input_thresh):offset=$(g target_offset):linear=true,aresample=48000" \
    -c:a aac -b:a 192k -shortest "$OUT/${FILM}.mp4"
fi
echo "done $OUT/${FILM}.mp4"
