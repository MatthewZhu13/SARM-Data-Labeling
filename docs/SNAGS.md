# Snag log: SARM2 (opensarm) on LeRobot v3.0 folding data

Everything that got in the way, or that differs from the paper/repo, and what we did about it.
Code: `xdofai/opensarm` pinned at `9978198` (`third_party/opensarm`).

## Availability and license
| Snag | Detail | Handling |
|---|---|---|
| No SARM2 checkpoints | No releases; authors said in issue #8 (Aug 2026) they will "try our best to open-weight upon paper acceptance" | Train both models from scratch. The frozen SigLIP-2 encoder is the only pretrained part. |
| License | GitHub shows "Other", but `LICENSE` is MIT (© XDOF, Inc.) | **Confirm with Parth** before using it on company data. |
| SARM1 baseline is not a starting point | `lerobot-data-collection/folding_sarm_reward` is SARM1 (CLIP), `single_stage`, top camera only: progress but no stages | Use it only as a progress baseline on the same test episodes. |
| Baseline checkpoint doesn't load in current LeRobot | Saved when SARM was a policy: `config.json` has `use_amp` and `use_peft`, which `lerobot` 0.7's `SARMConfig` rejects (`DecodingError`) | Download it to `checkpoints/folding_sarm_reward` and drop those two fields; the weights are unchanged. |
| Baseline script uploads by default | `lerobot/rewards/sarm/compute_rabc_weights.py` has `--push-to-hub` with `default=True`, so running it as documented pushes a parquet to the dataset repo | `scripts/baseline_sarm1.py` does the same inference (sparse head, eval preprocessor, centre frame of the window) without uploading. |
| No accepted/rejected labels in `high_quality_folding` | Every episode in it is "accepted"; the brief's pilot data isn't here yet | `high_quality_folding` was selected from `lerobot/full_folding`. Episodes match 1:1 on (length, per-episode state mean), 1,162 of 1,200. Unselected `full_folding` episodes from the same recording range and task are the rejected set (`sarm_pipeline/separation.py`). "Not selected" is the HF team's filter, not a per-episode verdict we have seen. |

## Data format
| Snag | Detail | Handling |
|---|---|---|
| opensarm cannot read v3.0 | It vendors LeRobot v2.1 (`episodes.jsonl`, one parquet/mp4 per episode, renamed columns) | Our own loader reads v3.0 metadata directly (`sarm_pipeline/v3.py`). |
| Stock primitives don't fit folding | Of SARM2's 21 action primitives, folding maps almost entirely to `fold_flatten`, so the stage estimator would learn "folding vs not" | Stage classes replaced with the dataset's 5 folding stages + `dummy`; MoE gates grouped as two-arm fold / one-arm fold / reposition / dummy. |
| Only 451 / 1,200 episodes have stage labels | `dense_subtask_*` columns exist for episodes 537–987 only; `sparse_subtask_*` is a single "Fold t-shirt" stage | Train on the 451. The labels look VLM-generated (boundaries on whole seconds); spot-check needed. |
| **Stage labels are mostly templates** | Only 139 distinct boundary patterns across 451 episodes; 60 episodes have exactly (8, 12, 20, 30) s and 48 exactly (8, 12, 20, 28) s, regardless of the video. Hand review of 25 readable test episodes: labels clearly wrong (compressed, shifted 5–15 s, or ignoring a retry) in ~19. | Hand labels for 40 train/val episodes (`labels/hand_train.json`) and all readable test episodes (`labels/test_gold_v2.json`), labeled blind with one rule (`docs/GOLD.md`). The labeler trained on the hand labels alone scores 91.4% vs 81% for the best model trained on dataset/pseudo labels. Earlier fixes (out-of-fold relabeling, `labels/test_corrections.json`) are kept for reference. |
| Client data is v2.1, not v3.0 | Both client sets are LeRobot v2.1 (one parquet and one mp4 per camera per episode), at 50 and 30 fps, with a 42-dim state | `sarm_pipeline/v21.py` reads v2.1 in place and resamples to 30 fps, so `frame_gap` means 1 s everywhere. `embed` decodes an explicit frame list at each video's own fps. |
| Client "Dataset" set is split and unusable | Downloaded as two folders, each holding part of the episode files and part of the metadata. `cam_high` points at the robot base, and the wrist cameras can't show the fold state. | Merged with hard links (`~/data/client/fold_clothes`); left out of training and labeling (`docs/CLIENT.md`). |
| Empty table at start and end look the same | Client episodes start and end with an empty table, so SARM2 can put progress near 0 on the final frames | Flags use the predicted stage timeline (reached "clear"), not the last progress value. |
| SARM2 progress flickers on small data | With ~15 training episodes per fold, per-frame progress jumps ±0.3, so a drawdown-based "backward" flag fired on 17 of 19 client episodes | Client review flags use top-camera stillness (idle) and the stage timeline instead; drawdown is reported only. |
| Annotated subset may be biased | Annotated episodes average 55 s vs ~90 s overall | Check before trusting results as representative. |
| Videos are AV1 at 720p | Random-access decoding during training would starve the GPU | Decode every file once and cache frozen SigLIP embeddings (`sarm_pipeline/embed.py`), as the paper does. |

## Code issues in opensarm (fixed in our loop, not patched in their repo)
| Snag | Detail | Handling |
|---|---|---|
| Normalizer hardcodes 14 state dims | `get_normalizer_from_calculated` slices `[:14]` (YAM 2×7); OpenArms has 16 (incl. both grippers) | Own normalization over all dims, stats from the train split. |
| Validation once per epoch, 1 epoch configured | Validation runs only at the very end; "best" checkpoint is just the final one | Validate every `train.val_every` steps; keep `best.pt` by validation loss. |
| An epoch can silently cut `num_steps` | Loop stops at min(num_steps, one pass over the data) | Step budget alone decides when training stops. |
| act_pri LR schedule mismatch | Cosine decays over `total_steps: 80000` but training stops at 16k, so the LR barely decays | Decay length always equals the step budget. |
| Random rewind in validation | Validation set built with training settings, so val loss is noisy and checkpoint choice unreliable | Validation on a fixed, seeded set of windows with no rewind. |
| act_pri window mismatch | `act_pri_ws` trains with the episode start frame (stage_model not set); `sarm2_ws` feeds it windows without it | Stage estimator always gets the 6-frame layout used inside the reward model. |
| act_pri trained on clips | Paper trains the stage estimator on single-primitive clips, but at inference windows cross stage boundaries | Default `stage_model.window: episode` (windows over whole episodes, labeled by anchor stage); `segment` reproduces the clip setup. |
| Pinned `wandb==0.13.3`, Python 3.10 only, `decord` | Environment friction, no macOS wheels for decord | We import only their model files; our env is torch + transformers + PyAV. Metrics go to `metrics.jsonl`. |

## Platform
| Snag | Detail | Handling |
|---|---|---|
| Template labels are early, not just noisy | The templates squeeze the stages toward the start: against blind hand labels the dataset boundaries are a median 3–5 s early (mean error 4–9 s). Models trained on them inherit the bias, and out-of-fold pseudo-labels copy it. The first hand-label calibration (median 2, 3, 2, 6 s) was measured on episodes that turned out to carry the template too, so it measured the template, not a convention. | Dropped the calibration; the convention is set from a non-template annotation (566) and applied blind. |
| `pkill -f` over ssh killed its own shell | `ssh box 'pkill -f "bash hand_runs.sh"; ...'`: the remote command line contained the pattern, so pkill killed the ssh shell and the tmux job's parent, which took the training run down with it | Launch and stop jobs from a script file on the box; kill by tmux session name. |
| MPS lacks a nested-tensor op | In eval mode, `nn.TransformerEncoder` with a padding mask calls `aten::_nested_tensor_from_mask_left_aligned`, which Apple MPS does not implement; the first validation crashed (the smoke run missed it because none of its windows needed a mask) | `setup()` turns off the MHA fast path on MPS; results are identical, just slower. CUDA is unaffected. |
| `non_blocking=True` copies corrupt data on MPS | Copying a DataLoader batch to MPS with `non_blocking=True` sometimes read the CPU buffer after reuse: ~2.5% of eval label rows arrived as garbage ints (eval crashed in the confusion matrix). Training batches were exposed the same way, so the `local_medium` models saw some corrupted targets. | `to_device` uses non_blocking only on CUDA (where it is safe); eval reads labels from the CPU batch. `local_medium` eval re-run clean; the box (CUDA) run is unaffected. |
| `--set lr=1e-4` arrived as a string | PyYAML (YAML 1.1) reads `1e-4` without a dot as text; AdamW crashed | Overrides that parse to a string are retried as floats (`config._parse_value`). |
| `box.sh sync` deleted files made on the box | The sync is `rsync --delete`, so pseudo-label files, the separation episode list, job scripts and the patched baseline checkpoint created on the box were removed by the next sync. The running SARM2 job had already read its labels, so nothing was lost that couldn't be rebuilt. | Job scripts live in `scripts/`; files made on the box are copied back to the repo; `checkpoints/` and `reports/` are excluded from the sync. |
| Box `torchaudio` doesn't match its `torch` | System torch 2.9.1 with torchaudio 2.11: `transformers` imports torchaudio and crashes with `undefined symbol: torch_library_impl` | The LeRobot venv gets `torchaudio==2.9.1` (cu129) installed into it; system packages untouched. |
| `lerobot` 0.7 splits SARM's dependencies into extras | Plain `pip install lerobot` can't load SARM or datasets; it needs `lerobot[dataset,sarm]` for `datasets` and `transformers` | Baseline runs in its own venv (`~/lr_venv`, see `scripts/baseline_sarm1.py`). |
| GPU box has no `python3-venv` | `python3 -m venv` fails (no ensurepip) on the box's Ubuntu Python 3.12 | `box.sh setup` makes the venv `--without-pip --system-site-packages` and installs with the system pip; no sudo/apt changes. |
| GPU box SSH key has a passphrase | Non-interactive ssh reports `Permission denied (publickey)`, which looks like an unauthorized key | Load it once with `ssh-add --apple-use-keychain ~/.ssh/deplace_gpu`. |

## Corrections to our own earlier assumptions
- The window is **7 frames** for the reward model (episode start + 5 history frames 1 s apart + anchor) and **6** for the stage estimator, not 8 / 7: `n_obs_steps: 6` counts the anchor.
- The `reward` column in opensarm is the training target (progress); `steps_to_go = 1 - progress` is what the reward head regresses.

## Known limitations (not bugs)
- Progress labels are monotonic within each training demo by construction (time through each stage). Backward and stall behaviour come from rewind augmentation and from the model reading appearance; it has to be verified on real failures.
- Frames before the first stage (progress 0) and after the last (progress 1) share the `dummy` class. They are ~1% of frames; watch the end-of-episode predictions.
- The total training loss can be negative: opensarm's MoE auxiliary loss subtracts a gate-entropy bonus. Track `fit_loss`.
- (Superseded: the cause was templated labels, see Data format.) Stage accuracy plateaus at ~75–78% (val) whatever the model: SARM2's 5 s causal stage estimator, regularized/smaller variants, ordered (Viterbi) decoding, combining with predicted progress, and a whole-episode bidirectional labeler all land there. Frame checks show both label timing errors (episode 539: boundaries 5–10 s late) and model errors (episode 763: jumps to the final fold 25 s early). Next: measure label consistency by hand on ~10 episodes, try stronger image features, or relabel with the VLM pipeline.
