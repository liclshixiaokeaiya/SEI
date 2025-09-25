#!/usr/bin/env bash
set -euo pipefail


Datadir="${Datadir:-data/DRiD/all}"
Dataset="${Dataset:-drid}"
Seed="${Seed:-1}"
PercMislabeled="${PercMislabeled:-0.1}"
NoiseType="${NoiseType:-confusion}"
ResultDir="${ResultDir:-outputs/dridentropy_signed_resnet}"
NetType="${NetType:-resnet}"
Depth="${Depth:-50}"
JsonPath="${JsonPath:-data/DRiD/confusion_10.json}"
NumClasses="${NumClasses:-5}"
Gpu="${Gpu:-0}"


Args=(--data "$Datadir"
      --dataset "$Dataset"
      --net_type "$NetType"
      --depth "$Depth"
      --perc_mislabeled "$PercMislabeled"
      --noise_type "$NoiseType"
      --seed "$Seed"
      --num_classes "$NumClasses"
      --use_threshold_samples
      --jsondir "$JsonPath")   

TrainArgs=(--num_epochs 150
           --lr 0.001
           --wd 1e-4
           --batch_size 128
           --num_workers 16)

SaveDir1="$ResultDir/results/${Dataset}_${NetType}${Depth}_percmislabeled${PercMislabeled}_${NoiseType}_threshold1_seed${Seed}"
mkdir -p "$SaveDir1"

CUDA_VISIBLE_DEVICES="$Gpu" python runnerclipentropysigned.py "${Args[@]}" \
  --save "$SaveDir1" \
  --threshold_samples_set_idx 1 \
  - train_for_sei_computation "${TrainArgs[@]}" \
  - done

[[ -f "$SaveDir1/train_data.pth" ]] || { echo "✖ Missing train_data.pth in $SaveDir1" >&2; exit 1; }

SaveDir2="$ResultDir/results/${Dataset}_${NetType}${Depth}_percmislabeled${PercMislabeled}_${NoiseType}_threshold2_seed${Seed}"
mkdir -p "$SaveDir2"

CUDA_VISIBLE_DEVICES="$Gpu" python runnerclipentropysigned.py "${Args[@]}" \
  --save "$SaveDir2" \
  --threshold_samples_set_idx 2 \
  - train_for_sei_computation "${TrainArgs[@]}" \
  - done

[[ -f "$SaveDir2/train_data.pth" ]] || { echo "✖ Missing train_data.pth in $SaveDir2" >&2; exit 1; }

CUDA_VISIBLE_DEVICES="$Gpu" python runnerclipentropysigned.py "${Args[@]}" \
  --save "$SaveDir1" \
  --threshold_samples_set_idx 1 \
  - generate_sei_details \
  - done

CUDA_VISIBLE_DEVICES="$Gpu" python runnerclipentropysigned.py "${Args[@]}" \
  --save "$SaveDir2" \
  --threshold_samples_set_idx 2 \
  - generate_sei_details \
  - done

echo "All done."
