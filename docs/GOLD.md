# Hand stage labels (`labels/test_gold_v2.json`, `labels/hand_train.json`)

The dataset's `dense_subtask` stage labels are mostly templates (see `docs/SNAGS.md`). Against the
video they are a median 3–5 s early at every boundary. A test score against them measures agreement
with the template, not with the video. This file records how the hand labels were made, so they can
be audited, extended or redone by someone else.

## Stage boundaries (rule)
Stage *k* starts on the first frame where stage *k−1*'s result is in place. Idle time and the reach
toward the next grasp belong to the next stage.

| Stage | Starts when | Looks like (top camera) |
|---|---|---|
| 1 first fold (both arms) | episode start | arms come in, top edge folded down |
| 2 slide closer | the first fold is placed: fold line visible, flap down | shirt pulled up toward the robot |
| 3 second fold (both arms) | the slide is done: shirt up against the robot edge | arms reach the bottom edge and fold it up |
| 4 third fold (one arm) | the long strip is formed | one arm folds the strip to about half its length |
| 5 final fold (one arm) | the strip is halved | final fold into a square, until the end (last frame stays `dummy`) |

Retries stay inside the stage being retried: a redone first fold (777, 884, 667) or a redone halving
(548, 778, 941) moves the next boundary to the end of the retry.

Where the rule comes from: episode 566 is one of the few with a non-template annotation
(12, 16, 25, 38 s). Its boundaries fall where slide, strip and halving are complete, and labeling it
blind with this rule gives 13, 16, 24, 37 s.

## How the labels were made
- Frames from the top camera (`observation.images.base`), one per second, with only the timestamp
  printed: no dataset label, no model prediction. Boundaries are whole seconds.
- `labels/test_gold_v2.json`: every test episode in `splits/hqf_v1.json`; 40 labeled, 5 `unclear`.
- `labels/hand_train.json`: 50 episodes drawn at random (seed 7): 40 from train, 10 from val. 40 of
  them are labeled (31 train, 9 val); 10 are `unclear` (`_unclear`).
- `unclear` is almost always the off-angle camera setup with the grey shirt (e.g. 578, 584, 596,
  803, 818), where the fold state cannot be read at 1 s. Those episodes are not scored, and the hand
  set contains none of them, so accuracy on that setup is unknown.
- To check or redo a label, view the episode at 1 fps and apply the rule. The test file stores
  [start, end) seconds per stage; the train file stores frames in the cache's `segments.json` format.

## Consistency
- Labeler vs themselves: 23 test episodes were labeled twice. The first pass (v1, below) had a
  model's prediction visible; the second was blind. Median difference 0 s at all four boundaries;
  mean absolute difference 1.7, 0.9, 0.7, 2.2 s. The stage 5 start (when the halving counts as done)
  is the least repeatable: 65% within 2 s.
- Dataset labels vs v2: median 3, 3, 3, 5 s early; mean absolute error 3.7, 4.7, 6.1, 9.1 s.
- One labeler made both the training and the test labels, so the model learns that person's reading
  of the rule. A second labeler on a subset would measure how much of the score is convention.

## Results (frame accuracy on `test_gold_v2.json`, 40 episodes)
| Model | Training labels | Accuracy | Excl. ±1 s of a boundary |
|---|---|---|---|
| `es_v1` | dataset | 75.5% | |
| `es_pseudo80` | dataset + out-of-fold pseudo-labels | 81.1% | 84.6% |
| `es_hand_s0/1/2` | hand labels only (31 train eps) | 91.0–91.3% | |
| `es_hand_ens` (3 seeds) | hand labels only | **91.4%** | 94.7% |
| `es_self_s0/1/2` | self-training: `es_hand_ens` labels for all 406 train/val eps, hand labels where they exist (`labels/pseudo_hand_ens.json`) | 91.0–91.4% | 94.6–94.7% |
| `es_self_ens` (3 seeds) | same | 91.3% | 94.7% |
| `es_self` + `es_hand` (6 models) | | 91.5% | 94.7% |

Self-training adds nothing: the same episodes stay worst (841 at ~65%, 710, 543, 716), so the remaining
errors are hard episodes and boundary timing, not too few training episodes.

`python -m sarm_pipeline.gold runs/<run>/episode_stage/predictions_test.parquet` scores against v2.

## History: v1 (`labels/test_corrections.json`)
The first pass reviewed only the 28 test episodes where `es_v1` disagreed with the dataset labels,
with labels and predictions printed on the frames. It then "calibrated" to the dataset by moving
boundaries 2, 3, 2, 6 s earlier, measured on 5 episodes whose labels looked right. Those 5 turned out
to carry the template too (e.g. 955 and 688: exactly 10/15/25/35 s), so the shift measured the
template rather than a convention, and v1 scores (84–85%) are not comparable to v2. v1 is kept for
reference; `--shift` still works with it.
