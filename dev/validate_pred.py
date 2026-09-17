import json, sys
import pandas as pd

D = "C:/Users/srika/Downloads/eris_actionable/"
ROLES = {"PLACE-ARG", "STREET-ARG", "TIME-ARG", "AFFECTEDOBJECTS-ARG", "DEATHVICTIM-ARG", "WOUNDVICTIM-ARG", "OFFICER-ARG", "REASON-ARG", "INFORMATION-ARG", "FIRE-EVENT", "FLOOD-EVENT", "EARTHQUAKE-EVENT", "ACCIDENT-EVENT", "FALSE-EVENT"}
TYPES = {"ARG", "EVE", "LOC", "PLOC", "ORG"}

sub = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else D + "working/submission.csv")
test = pd.read_csv(D + "test.csv")
errors = []
if list(sub.columns) != ["id", "prediction"]:
    errors.append(f"columns {list(sub.columns)}")
if len(sub) != len(test):
    errors.append(f"row count {len(sub)} vs {len(test)}")
if sub["id"].duplicated().any():
    errors.append("duplicate ids")
if set(sub["id"]) != set(test["id"]):
    errors.append("id set mismatch")
n_tokens = dict(zip(test["id"], test["n_tokens"]))
counts = {"arguments": 0, "entities": 0}
role_counts = {}
for i, p in zip(sub["id"], sub["prediction"]):
    try:
        d = json.loads(p)
    except Exception as e:
        errors.append(f"{i} bad json {e}")
        continue
    if set(d.keys()) != {"n_tokens", "arguments", "entities"}:
        errors.append(f"{i} keys {sorted(d.keys())}")
        continue
    n = d["n_tokens"]
    if not isinstance(n, int) or n <= 0:
        errors.append(f"{i} n_tokens {n}")
    if n != n_tokens[i]:
        errors.append(f"{i} n_tokens {n} != {n_tokens[i]}")
    for key, field, allowed in (("arguments", "role", ROLES), ("entities", "type", TYPES)):
        seen = set()
        for s in d[key]:
            counts[key] += 1
            if set(s.keys()) != {field, "start", "end"}:
                errors.append(f"{i} {key} keys {sorted(s.keys())}")
                continue
            if s[field] not in allowed:
                errors.append(f"{i} bad label {s[field]}")
            role_counts[s[field]] = role_counts.get(s[field], 0) + 1
            if not (isinstance(s["start"], int) and isinstance(s["end"], int)):
                errors.append(f"{i} non-int span")
                continue
            if not (0 <= s["start"] < s["end"] <= n):
                errors.append(f"{i} bounds {s['start']}-{s['end']} n={n}")
                continue
            span = set(range(s["start"], s["end"]))
            if span & seen:
                errors.append(f"{i} overlap in {key} at {s['start']}-{s['end']}")
            seen |= span
print("rows", len(sub), "arguments", counts["arguments"], "entities", counts["entities"])
print("per label:", dict(sorted(role_counts.items(), key=lambda x: -x[1])))
empty = sum(1 for p in sub["prediction"] if not json.loads(p)["arguments"] and not json.loads(p)["entities"])
print("docs with no spans at all:", empty)
if errors:
    print("ERRORS", len(errors))
    for e in errors[:20]:
        print("  ", e)
    sys.exit(1)
print("OK - submission satisfies the stated format rules")
