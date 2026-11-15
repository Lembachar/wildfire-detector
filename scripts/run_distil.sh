#!/usr/bin/env bash
# photo-only students learning from the before+now teachers, then the 3-way comparison
cd "$(dirname "$0")/../src"
P="$USERPROFILE/.venvs/wildfire/Scripts/python"
L=../data/train_distil.log
: > $L
log() { echo "$@" >> $L; }
for pair in "pretrained pretrained_hist" "unet unet_hist" "tile pretrained_hist"; do
  set -- $pair
  log "== student $1 (teacher $2) $(date)"
  $P -u train_v2.py --model $1 --teacher $2 >> $L 2>&1 || log "FAILED $1"
done
log "== final compare $(date)"
$P -u eval_distil.py >> $L 2>&1 || log "FAILED compare"
log "== all done $(date)"
