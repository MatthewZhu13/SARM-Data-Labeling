# Client data: stage and progress labels on the client t-shirt folding sets

Client data stays on the GPU box (`~/data/cloth-folding`). Client-derived files are kept out of git
(`.gitignore`): the hand labels, fold splits, and the report with per-episode plots and flags, all on the box.
This page holds aggregate numbers only.

## What's there
| Set | Episodes | Format | Usable? |
|---|---|---|---|
| "Dataset" (`Dataset-20261014T172733Z-1-00{1,2}`) | 17, 50 fps, 3 cameras, 42-dim state | LeRobot v2.1, one dataset split across two folders | **No.** See below |
| `nurvai2/lerobot-dataset2` | 19, 30 fps, 100–270 s each | LeRobot v2.1 | Yes: all results below |

Problems with the "Dataset" set:
- **Split download.** Each folder holds a different subset of the episode files: data parquets in one, videos in the other, metadata in both. Merging both folders gives the complete set; the merge uses hard links into `~/data/client/fold_clothes`.
- **No usable view.** `cam_high` points at the robot base for the whole episode, and the wrist cameras show close-ups of a black shirt against a reflective backdrop. The fold state can't be read from any camera, so the episodes can't be stage-labeled. Its own `annotation` column says "no visible robot activity" for long stretches.
- **Cut off at 160 s.** Episodes are mostly exactly 8000 frames (160 s), which looks like a recording time limit.

## Stages (defined from the video)
Rule as on `high_quality_folding`: stage *k* starts on the first frame where stage *k−1*'s result is in place.

| Stage | Starts when |
|---|---|
| 1 Spread the shirt flat | episode start (arm brings the shirt in, grabs and spreads it) |
| 2 Fold the sides into a long strip | the shirt lies flat |
| 3 Fold the strip into a bundle | the long strip is formed (all side folds done) |
| 4 Clear the folded shirt | the bundle is complete (pushed aside; table empty) |

- Episodes 8 and 11–13 start with the shirt already flat, so they have no stage 1.
- Hand labels cover all 19 episodes: top camera, one frame every 2 s, timestamps only, one labeler (`labels/client_hand.json` on the box).
- The black-shirt episodes (1, 11–13) are low contrast, and 3, 7 and 16 have long retries or tidying, so their boundaries are the least certain.

## How it was evaluated
- Only 19 episodes, so **5-fold cross-validation**: every episode is labeled by models trained on the other ~15 (13 train + 2 val), never on itself.
- Per fold:
  - the whole-episode stage labeler (`episode-stage`, 200 epochs)
  - SARM2: stage estimator 8k steps, reward model 15k steps
- `scripts/client_cv.sh` runs everything (~1 h on the box).

## Results: held-out predictions vs hand labels (19 episodes)
| | Frame accuracy | Excl. ±1 s of a boundary |
|---|---|---|
| Whole-episode stage labeler | **87.9%** | 88.9% |
| SARM2 stage estimator, ordered decoding | 85.9% | 86.9% |
| SARM2 stage estimator, per frame | 81.3% | 82.2% |

Per-stage recall (stage labeler): spread 92%, side folds 89%, bundle 78%, clear 91%.

**Boundary error** (median absolute; positive = predicted late):

| Stage start | Stage labeler | SARM2 (ordered) |
|---|---|---|
| side folds | 5.0 s (+0.3) | 3.9 s (0.0) |
| bundle | 6.0 s (+1.7) | 6.3 s (+2.4) |
| clear | 1.0 s (+0.7) | 1.0 s (+0.9) |

The labels themselves are only accurate to 2 s.

**Hardest episodes:** 15 (64%, a long sleeve-folding retry), 7 and 16 (long tidying after the strip), 14 and 5.

**Progress** (SARM2 vs hand-label stage-aware progress):
- MAE **0.085**; Spearman 0.885.
- MAE 0.058 when given the true stage, so about a third of the error comes from stage mistakes.
- The per-frame curve **flickers**, with brief jumps of ±0.3. It follows the right trend but is noisy, which is expected from ~15 training episodes. Compare `high_quality_folding`: MAE 0.024 from 406 training episodes.
- On the final empty-table frames, progress can drop to about 0, because the empty table at the end looks the same as the empty table at the start (seen in episode 1).

## Review flags
There is no accept/reject list for the client data, so episodes are flagged for a person to look at instead.

- **idle:** the top camera barely changes for 15 s or more before the clear stage.
  - Measured as a 5-s rolling median of the 1-s embedding change, below 0.015, which is about the 20th percentile of all changes.
  - The threshold was chosen on this same data, against the three stalls noted while hand-labeling.
- **unfinished:** the predicted stages never reach "clear".

| Episode | Flag | Note |
|---|---|---|
| 14 | idle, 17 s from 101 s | a real stall: the flat shirt sits untouched for about 40 s (80–118 s) |
| 12 | idle, 23 s from start | shirt already flat; the arm takes about 20 s to begin |
| 11 | idle, 15 s from start | same |
| 15, 17 | not flagged | stalls noted while labeling (72–102 s, 82–108 s) measure 12–13 s idle at the right time, just under the threshold |

- No episode is "unfinished": all 19 reach the clear stage.
- A "progress went backwards" flag isn't used: with this little training data, SARM2's flicker would flag nearly every episode.
- Thresholds should be set properly once some client episodes have accept/reject decisions.

## Not done, and why
- **`folding_sarm_reward` baseline on client data:** it expects a 32-dim state (OpenArms) and the client robot has 42, so it can't run as is.
- **Our `high_quality_folding` SARM2 applied zero-shot:** same reason (16-dim state).
- **Accepted vs rejected:** no labels exist yet.

## Next steps that would help most
1. More labeled client episodes. Stage accuracy went from 88% here (15 training episodes) to 91% on `high_quality_folding` with 31 hand-labeled ones. Progress needs more data still.
2. A few accept/reject decisions on client episodes, to set the flag thresholds and measure separation as on `high_quality_folding` (AUROC ≈ 0.9 there).
3. Fix the "Dataset" camera setup (`cam_high` should see the table) before collecting more data with that rig.
