import json

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


def label_stats(rows, label):
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
    return predicted_count, reference_count, exact, soft


def label_score(rows, label):
    P, R, E, S = label_stats(rows, label)
    if P == 0 and R == 0:
        return 1.0
    if P == 0 or R == 0:
        return 0.0
    d = P + R
    return 0.7 * (2.0 * E / d) + 0.3 * (2.0 * S / d)


def weighted_score(rows, labels, weights):
    total = sum(weights[label] for label in labels)
    return sum(weights[label] * label_score(rows, label) for label in labels) / total


def to_triples(obj, key, lab):
    return [(x[lab], int(x["start"]), int(x["end"])) for x in obj.get(key, [])]


def score_docs(preds, refs, verbose=False):
    arg_rows = [(to_triples(p, "arguments", "role"), to_triples(r, "arguments", "role")) for p, r in zip(preds, refs)]
    ent_rows = [(to_triples(p, "entities", "type"), to_triples(r, "entities", "type")) for p, r in zip(preds, refs)]
    a = weighted_score(arg_rows, ARGUMENT_ROLES, ARGUMENT_WEIGHTS)
    e = weighted_score(ent_rows, ENTITY_TYPES, ENTITY_WEIGHTS)
    total = 0.75 * a + 0.25 * e
    if verbose:
        print(f"total {total:.4f}  arg {a:.4f}  ent {e:.4f}")
        for lab in ARGUMENT_ROLES:
            P, R, E, S = label_stats(arg_rows, lab)
            print(f"  {lab:22s} w={ARGUMENT_WEIGHTS[lab]:.1f} P={P:5d} R={R:5d} E={E:5d} S={S:8.1f} score={label_score(arg_rows, lab):.4f}")
        for lab in ENTITY_TYPES:
            P, R, E, S = label_stats(ent_rows, lab)
            print(f"  {lab:22s} w={ENTITY_WEIGHTS[lab]:.1f} P={P:5d} R={R:5d} E={E:5d} S={S:8.1f} score={label_score(ent_rows, lab):.4f}")
    return total, a, e


def load_targets(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = d["target"]
    return out
