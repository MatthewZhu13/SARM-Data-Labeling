# Results: SARM2 on `lerobot/high_quality_folding`

Three questions from the brief:
- How accurate are the stage and progress labels?
- Does progress separate accepted demos from rejected ones?
- How does this compare with the `folding_sarm_reward` baseline?

All numbers are on held-out episodes: the 45-episode test split in `splits/hqf_v1.json`, or the separation set
below. Hand labels and how they were made: `docs/GOLD.md`.

## Models
| Name | What |
|---|---|
| `sarm_hand` | SARM2 (stage estimator + MoE reward model) trained on hand-ensemble stage labels: `labels/pseudo_hand_ens.json`, i.e. the `es_hand_ens` labeler's output on all 406 train/val episodes, hand labels where they exist |
| `hqf_pilot` | The same SARM2, trained on the dataset's own `dense_subtask` stage labels (mostly templates, 3–5 s early) |
| `folding_sarm_reward` | The HF baseline: SARM1 (CLIP), single stage, top camera only; run with `scripts/baseline_sarm1.py` |

## 1. Label quality on the test split
Stage accuracy is over all 45 test episodes: hand labels on 40, dataset labels on the 5 marked `unclear`. Progress
rows use the 40 hand-labeled episodes only.

| | `sarm_hand` | `hqf_pilot` | `folding_sarm_reward` |
|---|---|---|---|
| Stage accuracy, per frame (SARM2's causal estimator, ordered decoding) | **89.0%** | 75.3% | (no stages) |
| Progress MAE vs hand-label progress | **0.024** | 0.038 | 0.066 |
| Progress Spearman vs hand-label progress | **0.995** | 0.982 | 0.969 |

- Hand-label progress is stage-aware: each stage gets its average share of the episode, and progress moves linearly through each stage.
- The baseline was trained on a linear-in-time target. Against plain time it scores MAE 0.079, so its error is not a matter of convention.
- The offline whole-episode stage labeler (`episode-stage`, `es_hand_ens`) is still the more accurate stage source: 91.4% vs 89.0% for SARM2's 5-second causal window.

## 2. Accepted vs rejected
**Data.** `high_quality_folding` is a hand-picked subset of `lerobot/full_folding`.
- Episodes match one-to-one on (length, per-episode state mean): 1,162 of 1,200.
- **Accepted:** the 45 test episodes, found in `full_folding`.
- **Rejected:** 90 `full_folding` episodes from the same recording range (0–1522) and with the same task that were *not* picked.
  - 56 of them are stored in the same video files as labeled episodes, i.e. the same sessions; 44 share a file with an accepted test episode.
  - The other 34 are a seeded random draw.
- "Not picked" is the HF team's filter. We have not seen their per-episode reasons.
- Episode list: `splits/sep_full_folding.json`.

**Scores.** Each progress curve becomes one number per episode (`sarm_pipeline/separation.py`, at one value per second):

| Score | Meaning |
|---|---|
| `backward` | total downward movement |
| `rank_time` | Spearman of progress vs time, i.e. how steadily it rises |
| `final` | progress over the last 3 s |
| `drops` | number of 1-s drops larger than 0.05 |

AUROC is the chance that a random accepted episode scores better than a random rejected one; 0.5 is chance. Brackets are 95% bootstrap intervals.

| Score | `sarm_hand` | `hqf_pilot` | `folding_sarm_reward` |
|---|---|---|---|
| `backward` (less is better) | **0.91** [0.85, 0.95] | **0.92** [0.88, 0.96] | 0.70 [0.61, 0.78] |
| `rank_time` | **0.91** [0.85, 0.96] | 0.89 [0.82, 0.95] | 0.66 [0.57, 0.76] |
| `final` | 0.82 [0.75, 0.89] | 0.79 [0.72, 0.86] | 0.55 [0.45, 0.66] |
| `drops` (less is better) | 0.84 | 0.85 | 0.60 |
| episode length alone (shorter is better) | 0.59 | 0.59 | 0.59 |

**Checks on the result:**
- **Length.** Rejected episodes run longer (median 63 s vs 58 s), but length alone gives only 0.59. If rejected episodes longer than every accepted one are dropped (75 remain), SARM2's `backward` AUROC is 0.89–0.91.
- **Same sessions.** Against only the 44 rejected episodes stored in the same files as accepted ones, `backward` AUROC is 0.94 for `sarm_hand` and 0.98 for `hqf_pilot`, vs 0.66 for the baseline. So the gap is not a camera or session difference.

**What the curves show** (`reports/separation/curves.png`):
- On accepted episodes, SARM2 rises smoothly to about 1.
- On rejected episodes it often stalls below 0.9, and it also **flickers**: single frames drop to about 0, then come back.
  - That flicker is the model being unstable on footage unlike its training demos, not a measured undoing of the fold.
  - It still separates the two groups, but it means `backward` partly measures "unfamiliar-looking", not only "went backwards".
  - `final` (0.82) does not depend on the flicker.
- The baseline's curves are smoother but look about the same for both groups.

**Hand labels didn't change separation.** `sarm_hand` and `hqf_pilot` separate equally well. Better labels made stage and progress more accurate (section 1); telling good demos from bad comes from SARM2 having learned only from accepted demos.

## 3. What this means
- For labeling the frames of good demos, use `sarm_hand`, or `es_hand_ens` for stages.
- For flagging bad demos, any SARM2 run's `rank_time` or `backward` score works, at an AUROC of about 0.9 vs 0.66–0.70 for `folding_sarm_reward`. A threshold should be set on the pilot data once it arrives with its own accept/reject labels.
- Before using `backward` to point at *where* an episode goes wrong, smooth the curve or require drops to last more than about 1 s, because of the flicker.

## Reproduce
```bash
# hand-ensemble labels -> SARM2 retrain (GPU box, ~1 h)
python -m sarm_pipeline.pseudo --config configs/hqf_pilot.yaml --runs es_hand_s0 es_hand_s1 es_hand_s2
for s in train-stage train-reward eval; do
  python -m sarm_pipeline $s --config configs/hqf_pilot.yaml \
    --set run_name=sarm_hand labels.segments_override=labels/hand_ens_all.json
done
# accepted vs rejected + baseline (GPU box, ~1.5 h; needs ~/lr_venv, see scripts/baseline_sarm1.py)
scripts/separation_all.sh sarm_hand hqf_pilot      # -> reports/separation/
```
`labels/hand_ens_all.json` also holds the blind test labels, so `eval` scores against them. Training reads only train
episodes, so the test labels never reach a model.
