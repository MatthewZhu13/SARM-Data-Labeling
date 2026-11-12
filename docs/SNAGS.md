# Snag log: SARM2 (opensarm) on LeRobot v3.0 folding data

Everything that got in the way, or that differs from the paper/repo, and what we did about it.
Code: `xdofai/opensarm` pinned at `9978198` (`third_party/opensarm`).

## Availability and license
| Snag | Detail | Handling |
|---|---|---|
| No SARM2 checkpoints | No releases; authors said in issue #8 (Aug 2026) they will "try our best to open-weight upon paper acceptance" | Train both models from scratch. The frozen SigLIP-2 encoder is the only pretrained part. |
| License | GitHub shows "Other", but `LICENSE` is MIT (© XDOF, Inc.) | **Confirm with Parth** before using it on company data. |
| SARM1 baseline is not a starting point | `lerobot-data-collection/folding_sarm_reward` is SARM1 (CLIP), `single_stage`, top camera only: progress but no stages | Use it only as a progress baseline on the same test episodes. |

## Data format
| Snag | Detail | Handling |
|---|---|---|
| opensarm cannot read v3.0 | It vendors LeRobot v2.1 (`episodes.jsonl`, one parquet/mp4 per episode, renamed columns) | Our own loader reads v3.0 metadata directly (`sarm_pipeline/v3.py`). |
| Stock primitives don't fit folding | Of SARM2's 21 action primitives, folding maps almost entirely to `fold_flatten`, so the stage estimator would learn "folding vs not" | Stage classes replaced with the dataset's 5 folding stages + `dummy`; MoE gates grouped as two-arm fold / one-arm fold / reposition / dummy. |
| Only 451 / 1,200 episodes have stage labels | `dense_subtask_*` columns exist for episodes 537–987 only; `sparse_subtask_*` is a single "Fold t-shirt" stage | Train on the 451. The labels look VLM-generated (boundaries on whole seconds); spot-check needed. |
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
| MPS lacks a nested-tensor op | In eval mode, `nn.TransformerEncoder` with a padding mask calls `aten::_nested_tensor_from_mask_left_aligned`, which Apple MPS does not implement; the first validation crashed (the smoke run missed it because none of its windows needed a mask) | `setup()` turns off the MHA fast path on MPS; results are identical, just slower. CUDA is unaffected. |
| GPU box has no `python3-venv` | `python3 -m venv` fails (no ensurepip) on the box's Ubuntu Python 3.12 | `box.sh setup` makes the venv `--without-pip --system-site-packages` and installs with the system pip; no sudo/apt changes. |
| GPU box SSH key has a passphrase | Non-interactive ssh reports `Permission denied (publickey)`, which looks like an unauthorized key | Load it once with `ssh-add --apple-use-keychain ~/.ssh/deplace_gpu`. |

## Corrections to our own earlier assumptions
- The window is **7 frames** for the reward model (episode start + 5 history frames 1 s apart + anchor) and **6** for the stage estimator, not 8 / 7: `n_obs_steps: 6` counts the anchor.
- The `reward` column in opensarm is the training target (progress); `steps_to_go = 1 - progress` is what the reward head regresses.

## Known limitations (not bugs)
- Progress labels are monotonic within each training demo by construction (time through each stage). Backward and stall behaviour come from rewind augmentation and from the model reading appearance; it has to be verified on real failures.
- Frames before the first stage (progress 0) and after the last (progress 1) share the `dummy` class. They are ~1% of frames; watch the end-of-episode predictions.
- The total training loss can be negative: opensarm's MoE auxiliary loss subtracts a gate-entropy bonus. Track `fit_loss`.
