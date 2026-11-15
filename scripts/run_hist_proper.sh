#!/usr/bin/env bash
# 1) let the flawed U-Net (incomplete history) finish, then stop that chain before its pretrained run
# 2) wait for the history download, retry failed photos until none are missing (max 5 passes)
# 3) keep the flawed run's files, rebuild history inputs, train all 3 before+now models, compare
cd "$(dirname "$0")/../src"
P="$USERPROFILE/.venvs/wildfire/Scripts/python"
L=../data/train_hist2.log
: > $L
log() { echo "$@" >> $L; }



log "== download finished $(date): errors $(grep -c error ../data/history_fill.log)"
for pass in 1 2 3 4 5; do
  $P -u fetch_history.py > ../data/history_retry_p$pass.log 2>&1
  n=$(grep -c error ../data/history_retry_p$pass.log)
  log "  retry pass $pass: $n still failing"
  [ "$n" -eq 0 ] && break
  sleep 60
done
log "  photos with history: $(ls ../data/history/*.npz | grep -vc tmp) of $(ls ../data/chips/*.npz | grep -vc tmp)"

log "== keep the incomplete-history run $(date)"
D=../outputs/incomplete_history; mkdir -p $D
mv ../outputs/model_*_hist.pt ../outputs/scores_*_hist_test.npy $D/ 2>/dev/null
cp ../data/train_hist.log ../outputs/metrics_v2.json $D/
log "  kept: $(ls $D | tr '\n' ' ')"

log "== history inputs $(date)"
$P -u prepare_history.py >> $L 2>&1 || { log "FAILED prepare_history"; exit 1; }
for m in tile unet pretrained; do
  log "== model ${m}_hist $(date)"
  $P -u train_v2.py --model $m --history >> $L 2>&1 || log "FAILED $m"
done
log "== final compare $(date)"
$P -u eval_history.py >> $L 2>&1
log "== all done $(date)"
