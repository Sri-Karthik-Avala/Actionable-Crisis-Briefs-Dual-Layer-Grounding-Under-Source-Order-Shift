# Actionable Crisis Briefs: Dual-Layer Grounding under Source-Order Shift

| | |
| --- | --- |
| Final rank | not ranked |
| Domain | Sequence To Sequence |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Creator | yenwee0804 |
| Solutions submitted | 4 |
| Last submission | 2026-09-17 |

## Problem statement

### Overview

In a fast-moving crisis stream, a useful brief must preserve the facts that determine what responders can do next: where the incident is, which street or time is mentioned, what was affected, and whether victims or officials are involved. This challenge asks you to turn one Indonesian crisis post into a compact structured brief whose spans remain grounded in the post.

This is not ordinary token classification, standalone named-entity recognition, or free-form summarization. A conventional NER system labels mention identity and a conventional semantic-role system labels argument function; this task requires both views as a jointly useful evidence packet, even when the two layers cover the same tokens. The metric reflects an operator's asymmetric cost of missing a location, street, time, or victim rather than treating every label as equally useful.

The input is intentionally short and noisy, while the hidden evaluation tail follows the source order of each crisis type. This tests transfer to a held-out tail rather than random memorization of near-duplicate wording. The task is a sequence-to-sequence structured extraction problem: predict both actionable argument spans and named-entity anchors with token boundaries, then keep each predicted fact tied to an exact piece of visible evidence.

### Dataset

### File descriptions

- `train.csv`: public training posts with `id`, sanitized whitespace-tokenized `text`, and `n_tokens`.
- `train_targets.jsonl`: public structured training targets keyed by `id`; each line contains an `id` and a JSON `target` object used for learning. JSONL keeps the nested span target unambiguous.
- `test.csv`: hidden-holdout posts with the same feature columns as `train.csv`; labels are not included.
- `sample_submission.csv`: valid-format example predictions for every test ID.

### Column descriptions

- `id`: opaque challenge identifier for one post. Treat it as an identifier, not a feature.
- `text`: the post represented as whitespace-separated tokens. Token positions are the positions in this exact string after splitting on whitespace. Handles, URLs, email-like tokens, phone-like tokens, and blank source tokens may appear as `<USER>`, `<URL>`, `<EMAIL>`, `<PHONE>`, and `<EMPTY>` placeholders; positions are preserved.
- `n_tokens`: number of whitespace-separated tokens in `text`.
- `target`: a JSON object used only in `train_targets.jsonl`. It follows the submission format below and contains the annotated training brief.

Every row in `train.csv` has one matching line in `train_targets.jsonl`, and no test ID appears in that sidecar. The columns in `train.csv` and `test.csv` are identical and in the same order. The source document IDs, source event prefixes, collection timestamps, and raw token labels are not public features.

The split is a deterministic source-order holdout. During preparation, documents are grouped by crisis type and sorted by their source ordinal within each group; the first `floor(0.80 * N)` documents in each group form training data and the remaining documents form the hidden evaluation tail. This produces 3,318 training posts and 832 hidden posts, approximately 20% held out overall. Each complete post is assigned to exactly one side, so no token rows, annotations, or opaque IDs from a post cross the boundary. The tail design tests transfer to later posts within every crisis type instead of allowing random splitting to place near-neighbor source-order examples on both sides.

The entity list is a second, independent annotation layer over the same token positions. Use these entity types as follows:

- `ARG`: a token span that carries an event argument but is not typed as a place, event trigger, or organization.
- `EVE`: the word or phrase that names or triggers the crisis event.
- `LOC`: a physical or local place mention that is not represented as a political or administrative place.
- `PLOC`: a political, administrative, or geopolitical place mention such as a country, city, district, or province.
- `ORG`: an organization, public authority, or official institution mention.

An entity span and an argument span may cover the same tokens because the two lists answer different questions: the entity layer describes what kind of mention it is, while the argument layer describes the operational role it plays in the brief. Predict both layers independently.

### Evaluation

The submission is parsed as JSON. Each argument or entity span is represented by a label and a zero-based half-open interval `[start, end)`. For a predicted span and reference span with the same label, token-overlap F1 is `2 * overlap / (predicted_length + reference_length)`, where `overlap` is the number of shared token positions. Matching is greedy and one-to-one: all positive-overlap pairs are sorted by decreasing overlap F1, then predicted index, then reference index, and a pair is accepted only if neither span was matched before.

For each argument role and entity type, the grader aggregates matched spans over the whole hidden set. Let `P` be the number of predicted spans, `R` the number of reference spans, `E` the number of matched spans with exactly equal boundaries, and `S` the sum of overlap F1 for matched pairs. If both `P` and `R` are zero, that label scores 1. If exactly one is zero, it scores 0. Otherwise `exact_f1 = 2 * E / (P + R)`, `soft_f1 = 2 * S / (P + R)`, and `label_score = 0.7 * exact_f1 + 0.3 * soft_f1`.

The argument roles and utility weights are:

- `PLACE-ARG`: 2.0
- `STREET-ARG`: 2.0
- `TIME-ARG`: 1.6
- `AFFECTEDOBJECTS-ARG`: 1.6
- `DEATHVICTIM-ARG`: 2.4
- `WOUNDVICTIM-ARG`: 2.4
- `OFFICER-ARG`: 1.0
- `REASON-ARG`: 1.2
- `INFORMATION-ARG`: 1.0
- `FIRE-EVENT`: 1.2
- `FLOOD-EVENT`: 1.2
- `EARTHQUAKE-EVENT`: 1.2
- `ACCIDENT-EVENT`: 1.2
- `FALSE-EVENT`: 0.8

The entity types and utility weights are:

- `LOC`: 1.5
- `PLOC`: 1.5
- `EVE`: 1.0
- `ORG`: 1.0
- `ARG`: 0.8

`argument_score` is the weighted mean of the 14 argument-role scores. `entity_score` is the weighted mean of the five entity-type scores. The final `Actionable Brief Grounding Score` is `0.75 * argument_score + 0.25 * entity_score`. It is a maximize metric in the range `[0, 1]`. The exact-boundary component discourages bloated spans, while the overlap component gives useful partial credit when a boundary is close.

The following is the complete scoring implementation used by the challenge:

```
ARGUMENT_ROLES = ["PLACE-ARG", "STREET-ARG", "TIME-ARG", "AFFECTEDOBJECTS-ARG", "DEATHVICTIM-ARG", "WOUNDVICTIM-ARG", "OFFICER-ARG", "REASON-ARG", "INFORMATION-ARG", "FIRE-EVENT", "FLOOD-EVENT", "EARTHQUAKE-EVENT", "ACCIDENT-EVENT", "FALSE-EVENT"]
ENTITY_TYPES = ["ARG", "EVE", "LOC", "PLOC", "ORG"]
ARGUMENT_WEIGHTS = {"PLACE-ARG": 2.0, "STREET-ARG": 2.0, "TIME-ARG": 1.6, "AFFECTEDOBJECTS-ARG": 1.6, "DEATHVICTIM-ARG": 2.4, "WOUNDVICTIM-ARG": 2.4, "OFFICER-ARG": 1.0, "REASON-ARG": 1.2, "INFORMATION-ARG": 1.0, "FIRE-EVENT": 1.2, "FLOOD-EVENT": 1.2, "EARTHQUAKE-EVENT": 1.2, "ACCIDENT-EVENT": 1.2, "FALSE-EVENT": 0.8}
ENTITY_WEIGHTS = {"LOC": 1.5, "PLOC": 1.5, "EVE": 1.0, "ORG": 1.0, "ARG": 0.8}

def overlap_f1(left, right):
    overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    return 0.0 if overlap == 0 else 2.0 * overlap / (left[1] - left[0] + right[1] - right[0])

def match_spans(predicted, reference):
    candidates = []
    for p, pred in enumerate(predicted):
        for r, ref in enumerate(reference):
            score = overlap_f1(pred, ref)
            if score > 0:
                candidates.append((score, p, r))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_p, used_r, soft, exact = set(), set(), 0.0, 0
    for score, p, r in candidates:
        if p in used_p or r in used_r:
            continue
        used_p.add(p)
        used_r.add(r)
        soft += score
        exact += int(predicted[p] == reference[r])
    return soft, exact

def label_score(rows, label):
    predicted_count = reference_count = exact = 0
    soft = 0.0
    for predicted_row, reference_row in rows:
        predicted = [(start, end) for item, start, end in predicted_row if item == label]
        reference = [(start, end) for item, start, end in reference_row if item == label]
        predicted_count += len(predicted)
        reference_count += len(reference)
        row_soft, row_exact = match_spans(predicted, reference)
        soft += row_soft
        exact += row_exact
    if predicted_count == 0 and reference_count == 0:
        return 1.0
    if predicted_count == 0 or reference_count == 0:
        return 0.0
    denominator = predicted_count + reference_count
    return 0.7 * (2.0 * exact / denominator) + 0.3 * (2.0 * soft / denominator)

def weighted_score(rows, labels, weights):
    total = sum(weights[label] for label in labels)
    return sum(weights[label] * label_score(rows, label) for label in labels) / total

def actionable_brief_grounding_score(rows):
    argument_rows = [(row["arguments"], row["reference_arguments"]) for row in rows]
    entity_rows = [(row["entities"], row["reference_entities"]) for row in rows]
    return 0.75 * weighted_score(argument_rows, ARGUMENT_ROLES, ARGUMENT_WEIGHTS) + 0.25 * weighted_score(entity_rows, ENTITY_TYPES, ENTITY_WEIGHTS)
```

The code block shows the metric primitives; the packaged `challenge/grade.py` additionally validates IDs, JSON schema, token counts, allowed labels, boundaries, and non-overlap before calling the score. Scoring is performed per document and per label before aggregation, so spans from different posts cannot match each other.

### Submission

Submit `working/submission.csv` with one row for every ID in `test.csv` and exactly these columns:

- `id`: the corresponding opaque test ID.
- `prediction`: a JSON string containing the structured brief.

Example for a 29-token post:

```
id,prediction
c_88938ed5f6b6,"{""n_tokens"":29,""arguments"": [{""role"": ""PLACE-ARG"", ""start"": 1, ""end"": 3}], ""entities"": [{""type"": ""LOC"", ""start"": 1, ""end"": 3}]}"
```

### Requirements

- `prediction` must be valid JSON with exactly `n_tokens`, `arguments`, and `entities` as top-level keys.
- `n_tokens` must be a positive integer and should normally copy the row's `n_tokens` value; it guards the submitted span boundaries but is not itself scored.
- Each argument object must have exactly `role`, `start`, and `end`; each role must be one of the 14 listed argument roles.
- Each entity object must have exactly `type`, `start`, and `end`; each type must be one of `ARG`, `EVE`, `LOC`, `PLOC`, or `ORG`.
- Boundaries are zero-based, half-open, and must satisfy `0 <= start < end <= n_tokens`.
- Spans within the argument list and within the entity list must not overlap.
- IDs must be unique, and every test ID must appear exactly once.
- Generate the file at `./working/submission.csv` using only solver-visible files under `./dataset/public/`.

### What Not To Use

- Do not use source document IDs, token IDs, event prefixes, collection timestamps, raw annotation labels, or the hidden answer file.
- Do not reconstruct the source archive or query an external service at solution time. The public post is the complete intended evidence for each prediction.
- Do not use a fixed source-ID lookup, memorized answer list, or a rule-only decoder that ignores the public training targets.
- Do not add facts that are not grounded in the visible post. This is span grounding, not free-form crisis summarization.
