#!/bin/sh
set -eu

app_dir=/home/jesse/cheetocam
camera_env="$app_dir/.env.xiao-camera"

cp "$app_dir/.env" "$camera_env"
sed -i '/^CAMERA_ID=/d;/^CAMERA_NAME=/d;/^CAMERA_SOURCE=/d;/^CAMERA_SOURCE_KIND=/d;/^STREAM_PORT=/d;/^STREAM_PUBLIC_BASE=/d;/^STREAM_TOKEN=/d;/^STREAM_AUDIO_DEVICE=/d;/^STREAM_AUDIO_ENABLED=/d;/^LEARNED_PROFILE_CACHE_PATH=/d' "$camera_env"

stream_token=$(openssl rand -hex 24)
printf '%s\n' \
  'CAMERA_ID=xiao-camera' \
  'CAMERA_NAME=Wireless Cheeto Cam' \
  'CAMERA_SOURCE=http://cheeto-xiao-camera.local:8080/' \
  'CAMERA_SOURCE_KIND=wireless' \
  'STREAM_PORT=8089' \
  'STREAM_PUBLIC_BASE=https://cam2.cheeto.cam' \
  "STREAM_TOKEN=$stream_token" \
  'STREAM_AUDIO_DEVICE=' \
  'STREAM_AUDIO_ENABLED=0' \
  'LEARNED_PROFILE_CACHE_PATH=learned_profiles.xiao-camera.npz' \
  >> "$camera_env"

chmod 600 "$camera_env"
sudo install -m 0644 "$app_dir/cat_detector@.service" /etc/systemd/system/cat_detector@.service
sudo systemctl daemon-reload
sudo systemctl enable --now cat_detector@xiao-camera.service
