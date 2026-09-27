#!/usr/bin/env bash
# make_proxies.sh — frame-exact H.264 4:2:0 proxies of the 4K 4:2:2 10-bit originals.
#   tools/video/make_proxies.sh DataSets/C3896.MP4 [...]   -> DataSets/proxies/<stem>_{1080p,720p}.mp4
# One decode, two encodes. Every frame is kept (-fps_mode passthrough, no -r), so frame i of
# the proxy is frame i of the original and labels made on a proxy are valid on the original.
# 1080p: CVAT and pipeline experiments; 720p: web/demo. Colour tags are left as-is on purpose.
set -euo pipefail
out="${OUT:-DataSets/proxies}"
mkdir -p "$out"
common=(-an -c:v libx264 -g 30 -keyint_min 30 -sc_threshold 0 -fps_mode passthrough
        -video_track_timescale 30000 -movflags +faststart)
for src in "$@"; do
  stem="$(basename "${src%.*}")"
  ffmpeg -hide_banner -loglevel error -stats -y -i "$src" -filter_complex \
    "[0:v:0]split=2[a][b];[a]scale=1920:-2:flags=bicubic,format=yuv420p[v1];[b]scale=1280:-2:flags=bicubic,format=yuv420p[v2]" \
    -map "[v1]" "${common[@]}" -preset medium -crf 20 -bf 0 "$out/${stem}_1080p.mp4" \
    -map "[v2]" "${common[@]}" -preset medium -crf 23 -maxrate 2.5M -bufsize 5M -profile:v high -level:v 4.0 -bf 2 "$out/${stem}_720p.mp4"
  for p in "$out/${stem}_1080p.mp4" "$out/${stem}_720p.mp4"; do
    n_src=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$src")
    n_p=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets,r_frame_rate -of csv=p=0 "$p")
    echo "$p: packets,rate=$n_p (original frames $n_src)"
  done
done
