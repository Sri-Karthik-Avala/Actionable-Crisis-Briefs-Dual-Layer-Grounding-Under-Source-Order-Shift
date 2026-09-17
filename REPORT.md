# Actionable Brief Grounding (Indonesian crisis posts → argument + entity spans) — diagnostic report

**Status:** 3 scored submissions, all between 0.702 and 0.710. Target was 0.734, then 0.74.
Every lever I could think of has been measured; almost all land inside the noise band. This document
contains every number so a reviewer can find what I am missing.

| submission | what changed | holdout estimate | **LB (Script Run Score)** | holdout→LB gap |
|---|---|---|---|---|
| v1 | 6-member BIO+span mix, global threshold | 0.7116 | **0.7103** | −0.0013 |
| v2 | dropped XLM-R-large **and** switched to per-label prior-matched thresholds | ~0.7176 | 0.7022 | **−0.015** |
| v3 | v1 + 2 extra members (pure addition), v1's threshold mechanism | 0.7137 | 0.7048 | −0.009 |

**Key tension:** the holdout tracked the LB almost perfectly for v1, but the harder I tuned against it,
the worse the transfer got. v3 was a pure *addition* to v1 and still lost 0.0055. Leaderboard noise on
832 docs is ≈ ±0.010, so all three scores sit inside one noise band.

---

## 1. Task and metric

One Indonesian crisis post (whitespace tokens) → two **independent, non-overlapping** span layers:

- **arguments** — 14 roles with utility weights: PLACE 2.0, STREET 2.0, TIME 1.6, AFFECTEDOBJECTS 1.6,
  DEATHVICTIM 2.4, WOUNDVICTIM 2.4, OFFICER 1.0, REASON 1.2, INFORMATION 1.0, FIRE-EVENT 1.2,
  FLOOD-EVENT 1.2, EARTHQUAKE-EVENT 1.2, ACCIDENT-EVENT 1.2, FALSE-EVENT 0.8 (sum 20.8)
- **entities** — 5 types: LOC 1.5, PLOC 1.5, EVE 1.0, ORG 1.0, ARG 0.8 (sum 5.8)

Per label, pooled over the whole test set: greedy one-to-one matching by span overlap-F1;
`P` = #pred, `R` = #gold, `E` = #exact matches, `S` = Σ overlap-F1 of matched pairs.
`label_score = 0.7·2E/(P+R) + 0.3·2S/(P+R)`; 1.0 if P=R=0; 0 if exactly one is 0.
`score = 0.75·weighted_mean(14 roles) + 0.25·weighted_mean(5 types)`.

So: exact boundaries dominate (70%), labels are scored separately, and rare high-weight labels
(WOUNDVICTIM / DEATHVICTIM at 2.4) matter disproportionately.

## 2. Hard constraints on any proposed solution (please respect these)

- One end-to-end `solution.py`, **real training inside the script**, single A10G, ~1 h target (1 h 30 m hard).
- HuggingFace pretrained downloads are allowed; no external data; no synthetic training data.
- **No test-set statistics**: no pseudo-labelling, no calibrating/ranking/thresholding against the test
  set's distribution. Anything fitted must be fitted on train (or an internal train holdout) and applied per row.
- No hardcoded dataset findings (thresholds etc. must be derived at runtime).
- The challenge's scoring formula must not appear in the submitted script.
- No TF-IDF / n-gram features anywhere in the script (owner's rule for NLP challenges).
- Rule-only decoders are not allowed; the model must do the learning.

## 3. Data facts (verified)

**Sizes.** train 3318 posts, test 832. Tokens/post: train mean 23.6 (p90 41, max 78); test mean 25.2
(p90 42, **max 126**). Placeholders: only `<PHONE>` (48 train / 13 test) and `<EMPTY>` (3 / 1) occur.
No duplicate texts; no test text appears in train.

**The split is reproducible from row order.** Train rows sit in contiguous crisis-type blocks
(flood 1000 → earthquake 724 → fire 902 → accident 692); test follows the same block order with exactly
the 20% tail of each (250 / 182 / 226 / 174). A keyword change-point DP recovered the boundaries with zero
disagreement. **Local validation = last 20% of each train block (663 docs)**, which mirrors the hidden tail.

**Label counts (train).**

| arg role | spans | docs | mean len | | entity | spans | mean len |
|---|---|---|---|---|---|---|---|
| PLACE | 4481 | 2701 | 1.86 | | ARG | 3935 | 2.49 |
| FALSE-EVENT | 2016 | 1766 | 1.35 | | EVE | 3810 | 1.27 |
| AFFECTEDOBJECTS | 1126 | 911 | 2.05 | | LOC | 3184 | 1.52 |
| OFFICER | 1053 | 784 | 1.63 | | PLOC | 2304 | 1.54 |
| TIME | 1005 | 657 | 1.92 | | ORG | 1063 | 1.57 |
| FIRE-EVENT | 653 | 533 | 1.04 | | | | |
| STREET | 630 | 557 | 3.97 | | | | |
| ACCIDENT-EVENT | 512 | 481 | 1.13 | | | | |
| INFORMATION | 446 | 382 | 3.48 | | | | |
| FLOOD-EVENT | 371 | 333 | 1.15 | | | | |
| EARTHQUAKE-EVENT | 241 | 190 | 1.90 | | | | |
| DEATHVICTIM | 195 | 182 | 2.76 | | | | |
| REASON | 175 | 161 | 3.63 | | | | |
| WOUNDVICTIM | 98 | 92 | 3.31 | | | | |

**The two layers.** 92% of argument spans (11998/13002) coincide exactly with an entity span, and
role→type is near-deterministic (PLACE→LOC/PLOC, OFFICER→ORG, events→EVE, everything else→ARG). But the
entity layer segments more finely (one PLACE-ARG "sungai tukad batu niti desa tulamben kecamatan kubu
karangasem" = four LOC/PLOC entities), so the layers need independent heads.

**Events are a document-level judgment, not a lexicon.** FALSE-EVENT (event word used non-literally)
occurs in every block (flood 777, quake 608, fire 418, accident 213 spans). The same trigger flips:
"banjir" FALSE 671 / FLOOD 315; "kebakaran" FALSE 364 / FIRE 567. Within a post, event roles are
consistent 403/417.

**Source-order drift toward the tail.**
- Posts get longer toward the tail in every block (flood 21.8→26.2 tokens early vs tail chunk;
  fire 26.5→28.1) and test is longer still (flood 24.5, quake 27.0, fire 27.5, accident 21.3).
  Test has 25.4% of posts in the 35–50-token bucket vs 20.8% in train.
- Spans per doc rise with length but spans per token fall (args/doc 2.86 → 5.21, args/token 0.275 → 0.130).
- Label rates per 1000 tokens across chunks 0→4: FLOOD-EVENT 6.51 → 2.85, STREET 5.83 → 8.89,
  TIME 16.7 → 11.3, PLOC 31.6 → 25.1, EARTHQUAKE 4.68 → 3.02.

## 4. Validation reliability

- Tail holdout bootstrap std (200 resamples): **0.0117**.
- The two random halves of the holdout scored 0.698 and 0.660 on the same predictions.
- Holdout ↔ LB: see the status table. v1 transferred within 0.0013; the more-tuned v2/v3 transferred worse.

## 5. Solution architecture (current = v3)

**Members.** Each member = a pretrained encoder, first-subword word pooling, dropout 0.1, and one of two heads:
- **BIO head**: linear → per-token softmax over 1+2·14 argument tags and 1+2·5 entity tags.
- **Span head**: biaffine over (start word, end word) with 150-d GELU projections + start/end linear
  terms + a span-length embedding (max span 20 words); softmax over labels+none per candidate span.

Training: AdamW, encoder LR 2e-5 (large) / 5e-5 (base), head LR 1e-3, weight decay 0.01, linear
warmup 10% then linear decay, batch 16 (length-bucketed), 10 epochs, fp16 + GradScaler, grad clip 1.0,
sum of both layers' CE losses. Trained on 100% of train.

**Decoding — the one real modelling gain of the session ("span-probability mixing").** Every member is
converted to a span-probability tensor `P[start, length, label]`:
- span head: its softmax directly;
- BIO head: `P = p_B(s) · Π p_I(s+1..s+l) · (1 − p_I(s+l+1))` (independent-token marginal of exactly
  that span), none-class = 1 − Σlabels.
Members are averaged; spans are selected greedily by probability above a threshold, non-overlapping.

**Threshold.** Two cheap IndoBERTweet members (BIO + span) train on an internal generic split (last 20%
of each fifth of the file), a grid over thresholds measures plain macro span-F1 on that internal holdout,
and `threshold = best_F1 / 2` (F1-optimal-threshold theory). Selected 0.3544 in every run.

**Versions.**
- v1 (LB 0.7103): XLM-R-large BIO, XLM-R-large span, IndoBERTweet BIO, IndoBERTweet span, mDeBERTa-v3-base BIO, IndoBERT-large-p1 BIO; global threshold.
- v2 (LB 0.7022): IndoBERTweet / mDeBERTa / IndoBERT-large, each BIO+span (no XLM-R-large); per-label prior-matched thresholds.
- v3 (LB 0.7048): all 8 = {XLM-R-large, IndoBERTweet, mDeBERTa-v3-base, IndoBERT-large-p1} × {BIO, span}; global threshold.

Runtime: v3 ≈ 51 min on an uncontended RTX 3080 laptop, estimated 25–30 min on an A10G.

## 6. Single-model bake-off (tail holdout, 10 epochs)

| backbone | BIO last epoch (best) | BIO decoded as span-probs | span head |
|---|---|---|---|
| XLM-R-large | 0.6755 (0.6774) | 0.6754 | 0.6728 |
| XLM-R-large + CRF | 0.6830 Viterbi w/ transitions; 0.6859 w/o | **0.3491** (CRF emissions are not marginals) | — |
| mDeBERTa-v3-base | 0.6746 | **0.6804** | 0.6655 |
| IndoBERT-large-p1 | 0.6692 (0.6722) | **0.6736** | 0.6650 |
| IndoBERTweet-base | 0.6631 (0.6698) | **0.6721** | 0.6692 |
| XLM-R-base | 0.6575 (0.6602) | — | — |
| twitter-XLM-R-base | 0.6501 | — | — |

Indonesian base models are within ~0.01 of XLM-R-large at a third of the cost. InfoXLM-large was fetched
but never measured (GPU memory contention).

## 7. Ensembling and composition (tail holdout)

- BIO-only Viterbi ensembles: 2 models 0.675–0.692, best 3 = 0.6924, best 5 (including a CRF member) = 0.7091.
- **BIO+span mixing**: IndoBERTweet BIO + IndoBERTweet span = 0.7020 at threshold 0.40 (beats every
  3-model BIO ensemble). v1 composition = 0.7116 at 0.3544.
- **Ensemble saturation:** adding a 7th member (mDeBERTa span) to v1's six: 0.7116 → 0.7116.

| composition (threshold 0.3544) | holdout |
|---|---|
| IndoBERTweet BIO+span | 0.6987 |
| mDeBERTa BIO+span | 0.6926 |
| IndoBERT-large BIO+span | 0.6841 |
| IndoBERTweet + mDeBERTa pairs | 0.7135 |
| IndoBERTweet + IndoBERT-large pairs | 0.7113 |
| mDeBERTa + IndoBERT-large pairs | 0.7050 |
| 3 backbones × 2 heads (= v2 members) | 0.7160 |
| … + XLM-R-large BIO | 0.7160 |
| … + XLM-R-large span | 0.7172 |
| all 8 (= v3) | 0.7137 |
| 3 BIO only | 0.6962 |
| 3 span only | 0.7023 |
| IndoBERTweet + mDeBERTa pairs + XLM-R-large span | 0.7195 (max of 13, selection-biased) |

## 8. Every lever tested and its measured effect

| lever | result | verdict |
|---|---|---|
| Span-probability greedy decode instead of Viterbi (per BIO model) | −0.0001 (XLM-R-large) to +0.009 (IndoBERTweet) | **kept** |
| Mixing BIO and span-head members | 0.6924 → 0.7116 | **kept** |
| Global threshold by F1/2 rule vs argmax of grid | 0.7116 vs 0.7082 | **kept** |
| CRF training (single XLM-R-large) | 0.6755 → 0.6830 | noise-level; breaks mixing |
| Per-label bias on BIO/Viterbi, cross-fitted | +0.010 (1-epoch model), +0.006 (XLM-R-large), +0.0056 (3-model) | superseded by mixing |
| Event-consistency relabelling within a post | −0.001 / +0.001 / −0.0013 | rejected |
| Cross-layer fill (add entity for arg-only span) | −0.0003 | rejected |
| Cross-layer probabilistic coupling via P(type\|role) | +0.0002 best (alpha 0.25) | rejected |
| Exact DP (weighted interval scheduling) vs greedy | +0.0001 | rejected |
| Per-label threshold multipliers, cross-fitted | **−0.008** | rejected (overfits) |
| Separate thresholds per layer | best 0.7128 vs 0.7116 | noise |
| **Oracle per-label thresholds (in-sample, cheating)** | **0.7262 (ceiling +0.0146)** | bound |
| Prior-matched per-label thresholds, self-calibrated (per-doc rate) | 0.7169–0.7181 | not shippable as-is |
| … transferred from 2-member calibration ensemble | 0.7132 | +0.0016 |
| … transferred from 4-member calibration ensemble | 0.7122 | no better |
| … shrunk toward global threshold (w = 0 / .25 / .5 / .75 / 1) | 0.7116 / 0.7130 / 0.7120 / 0.7129 / 0.7132 | flat |
| … length-conditional count model (linear in n_tokens) | 0.7151 | worse than per-doc |
| … recency priors (latest chunk only) | 0.7049 | worse |
| Snapshot averaging over last 4 epochs (single model) | +0.0009 | negligible |
| 16 epochs vs 10 (span head) | last epoch 0.6607 vs 0.6608 | no gain |
| Utility-weighted CE (span members) | per member +0.0124 / +0.0058 / +0.0032 (3/3 positive) | — |
| … same, at ensemble level, fixed threshold | 0.7099 vs 0.7160 | worse |
| … same, at ensemble level, prior-matched (calibration absorbed) | 0.7157 vs 0.7208 | **rejected** — it was a calibration shift |
| Span reranker (HistGBM on candidate features, cross-fitted) | 0.7017 (bugged threshold) / **0.7032** vs 0.7137 | rejected |
| Placeholder mapping (`<PHONE>` → word) | not isolated (48 train tokens) | irrelevant |
| Inverse-frequency class-balanced CE | implemented, **not run** | open |
| 80% → 100% training-data learning curve | queued, **not run** | open |
| InfoXLM-large as an extra backbone | fetched, **not run** (VRAM) | open |

## 9. Error analysis (v1 composition on the tail holdout, threshold 0.3544 → 0.7116)

**Per-label** (score = the metric's label_score; weighted loss = weight·(1−score) scaled into the total):

| label | P | R | E | score | share of total score lost |
|---|---|---|---|---|---|
| WOUNDVICTIM | 19 | 24 | 8 | 0.400 | **0.052** |
| STREET | 99 | 150 | 67 | 0.573 | **0.031** |
| DEATHVICTIM | 38 | 35 | 26 | 0.749 | 0.022 |
| INFORMATION | 63 | 91 | 34 | 0.469 | 0.019 |
| PLOC | 468 | 424 | 312 | 0.702 | 0.019 |
| AFFECTEDOBJECTS | 221 | 229 | 149 | 0.695 | 0.018 |
| LOC | 712 | 655 | 514 | 0.757 | 0.016 |
| FLOOD-EVENT | 74 | 48 | 42 | 0.695 | 0.013 |
| REASON | 25 | 35 | 21 | 0.708 | 0.013 |
| ORG | 229 | 232 | 163 | 0.719 | 0.012 |
| EARTHQUAKE-EVENT | 63 | 51 | 41 | 0.727 | 0.012 |
| OFFICER | 226 | 221 | 155 | 0.706 | 0.011 |
| ARG | 765 | 797 | 528 | 0.704 | 0.010 |
| PLACE | 990 | 872 | 794 | 0.862 | 0.010 |
| TIME | 218 | 191 | 168 | 0.839 | 0.009 |
| ACCIDENT-EVENT | 108 | 99 | 80 | 0.801 | 0.009 |
| FIRE-EVENT | 153 | 129 | 117 | 0.836 | 0.007 |
| FALSE-EVENT | 385 | 421 | 346 | 0.860 | 0.004 |
| EVE | 797 | 754 | 721 | 0.936 | 0.003 |

**Error budget — label confusion dominates, not boundaries or detection.**

| | arguments | entities |
|---|---|---|
| predictions matching a gold span of the same label | 2240 | 2415 |
| predictions overlapping only a gold span of a **different** label | 218 | 302 |
| predictions with no gold span there at all | 224 | 254 |
| gold spans found | 2228 | 2415 |
| gold spans covered only by a wrong-label prediction | 186 | 257 |
| gold spans with nothing predicted | 182 | 190 |

Top confusions: **PLACE→STREET 86**, FLOOD-EVENT→FALSE 26, FIRE-EVENT→FALSE 24, PLACE→INFORMATION 13;
entities **PLOC→LOC 91, LOC→PLOC 77, LOC→ARG 76**, PLOC→ARG 34.

**Boundaries are minor.** Of argument predictions: 2048 exact, 192 overlapping-but-inexact (115 within
one token at both ends). For inexact same-label matches pooled over both layers, start delta (pred−gold)
{−2:14, −1:79, 0:185, +1:40, +2:22} and end delta {−2:29, −1:41, 0:164, +1:77, +2:27} — a mild
over-extension, no systematic off-by-one. Gold
conventions are near-exception-free ("di" starts a gold span 2/3366 times, "dan" 0/917, punctuation 0),
and the model already respects them; remaining off-by-one cases are content words (sto, kota, provinsi,
beruntun, tunggal).

**Candidate-pool recall — the headroom that no selector has captured.** Fraction of gold spans whose
exact (span, label) has ensemble probability above τ: arguments 0.935 (τ=0.02), 0.919 (0.05), 0.893
(0.10), 0.858 (0.20); entities 0.944 / 0.921 / 0.899 / 0.851. Actual exact selection ≈ 69%. Rank of the
gold span among all of a post's candidates: WOUNDVICTIM median 6 (never rank 0), STREET median 6,
INFORMATION median 9, PLACE median 1 (rank-0 38%, top-5 85%).

## 10. Why I believe the ceiling is near 0.71–0.72

**LOC vs PLOC is close to a coin flip in the gold labels** (13% of the total metric weight):
- "jakarta" LOC 62 / PLOC 64; "indonesia" LOC 61 / PLOC 94; "dki" 6/11; "malang" 12/5.
- 122 of 181 place strings seen ≥5 times are labelled both ways (minority share > 25%).
- Mean minority share for strings seen ≥4 times: **0.445** (0.5 = pure coin flip).
- Administrative head words don't disambiguate: P(PLOC | kecamatan/kabupaten/kota/desa/…) = 0.51;
  physical head words P(PLOC) = 0.21; bare names 0.40.
- Chain position is only a soft tendency: P(LOC) by position in a place chain 0.66 / 0.41 / 0.34 / 0.47;
  only 464 of 1092 multi-span chains follow "LOC then PLOC…".
- String-identity majority oracle on the holdout: 0.652; **our ensemble: 0.797** (already beyond it).
  In-sample Bayes bound from string identity alone: 0.838.
- Same doc can contain "kecamatan mojo" as LOC in one place and PLOC in another.

**WOUNDVICTIM (weight 2.4) is annotated inconsistently for near-identical posts.** Examples from the holdout:
- gold = one span "107 orang luka berat 808 orang luka ringan 94 orang hilang 16783 jiwa/3876 kakak mengungsi";
  a near-identical post has gold "9 orang hilang 2 orang luka berat" + "2 orang luka ringan" as two spans.
- evacuee counts: "9000 jiwa" (mengungsi) is gold, "4728 orang" / "1000 kakak" / "1317 jiwa" in the same
  context are not.

**STREET vs PLACE is learnable but still confused.** A street cue token (jl, jalan, km, gg, gang, rw, rt,
tol, raya, jln) gives P(STREET) = 0.87; without one P(STREET) = 0.02 (STREET: 525 with cue / 105 without;
PLACE: 77 / 4404). PLACE is 7× more frequent than STREET, so class imbalance is the likely cause of the 86
confusions — class-balanced CE is implemented but untested.

## 11. Process mistakes (so the reviewer can discount my numbers appropriately)

1. v2 shipped a composition change whose holdout gain (+0.0044, the max over 13 combinations) was inside
   the ±0.012 noise band, **bundled** with a threshold change — the −0.008 regression is unattributable.
2. v3 was chosen as the "safe" pure addition; it still lost 0.0055. Conclusion: at this sample size the
   holdout cannot rank configurations that differ by < ~0.01.
3. Many experiments ran under heavy GPU contention from other jobs on the same machine; some runs were
   killed or duplicated and re-run. All numbers above come from completed runs.

## 12. Questions for the reviewer

1. **Label-noise-optimal prediction.** LOC/PLOC are scored as separate labels with pooled F1 and gold is
   ≈ coin-flip per string. Is there a prediction strategy that raises expected pooled F1 under this kind of
   symmetric label noise (e.g., always predicting the globally more frequent type for ambiguous strings,
   or deliberately biasing toward one type), given spans in a layer cannot overlap so hedging both labels
   on one mention is impossible?
2. **Exploiting the source-order tail compliantly.** Posts get longer and label rates drift (FLOOD-EVENT
   down, STREET up) toward the tail, and test is the next tail. What train-only, per-row-applied techniques
   exploit that trend without calibrating on the test set?
3. **Selection headroom.** 93.5% of gold argument spans are candidates with p > 0.02 but ~69% are selected,
   and a GBM reranker on candidate features (probabilities, per-member probabilities, margins, overlap
   statistics, length, position, rank) did worse than raw probability. What reranking formulation would work
   at ~3300 training posts (listwise per post? pairwise within overlapping candidates? a trained second-stage
   transformer that sees the text plus the candidate span?)
4. **Stronger backbones within 1 h on an A10G** for noisy lowercase Indonesian/Malay social text: anything
   likely to beat XLM-R-large + IndoBERT-large + IndoBERTweet here (NusaBERT? a fine-tuned small decoder LLM
   for span extraction? RemBERT?) — and would it be diverse enough to escape the ensemble saturation?
5. **Validation under ±0.01 noise.** What CV design lets me rank configurations that differ by ~0.005 when
   the tail holdout has 663 posts and the LB has 832? (Repeated chunk-wise CV across all five chunks?
   multiple seeds per configuration? paired bootstrap?)
6. **Why transfer degrades with tuning.** v1 transferred within 0.0013; v2/v3 lost 0.009–0.015 against their
   holdout estimates. Is this ordinary holdout overfitting from ~40 configuration comparisons on one 663-doc
   split, or does it suggest the test tail differs from my holdout in a way I haven't measured?
7. **Joint modelling of the two layers.** 92% of argument spans equal an entity span and role→type is almost
   deterministic, yet probabilistic coupling at decode time added nothing. Would a joint output space
   (combined role×type labels) or a shared span head trained on both layers do better than two heads on a
   shared encoder?
8. **The 30% soft-overlap component.** For labels where exact match is hopeless (WOUNDVICTIM, INFORMATION),
   is there a principled way to trade exact-match for overlap credit?
9. **Anything structural I'm missing** in the metric or data that a different framing (sequence-to-sequence
   generation, question-answering per role, token-level multi-label) would exploit.

## 13. Files

- `solution.py` — v3 (current forward version).
- `working/submission.csv` — v3 predictions (validated); `working/submission_v1.csv` — v1 predictions.
- `dev/tagger.py` (BIO/CRF harness), `dev/spanner.py` (biaffine span harness, weighted / class-balanced loss
  options), `dev/mix.py` (span-probability conversion and mixed decode), `dev/abgs_metric.py` (offline metric).
- Analyses: `dev/compsearch.py`, `dev/perlabel.py`, `dev/oracle_thr.py`, `dev/prior_*.py`, `dev/joint.py`,
  `dev/dpdec.py`, `dev/rerank.py`, `dev/errbudget.py`, `dev/confus.py`, `dev/ceiling_locploc.py`,
  `dev/drift.py`, `dev/drift2.py`, `dev/bound.py`, `dev/bound_lex.py`, `dev/pool.py`, `dev/chain.py`.
- Holdout prediction dumps: `colab_out/out/*_last.npy`, `dev/out/*_last.npy` (per-post log-probs on the
  663-post tail holdout, from models trained on the other 80%).
