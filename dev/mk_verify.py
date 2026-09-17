s=open("solution.py",encoding="utf-8").read()
old_members = s[s.index("MEMBERS = ["):s.index("]\nSNAPSHOTS")+1]
new_members = '''MEMBERS = [
    {"backbone": "indolem/indobertweet-base-uncased", "head": "bio", "lr": 5e-5, "epochs": 1, "seed": 11},
    {"backbone": "microsoft/mdeberta-v3-base", "head": "span", "lr": 5e-5, "epochs": 1, "seed": 14},
]'''
s = s.replace(old_members, new_members, 1)
s = s.replace('"epochs": 10, "seed": 21', '"epochs": 1, "seed": 21').replace('"epochs": 10, "seed": 22', '"epochs": 1, "seed": 22')
s = s.replace('train = pd.read_csv(public_dir / "train.csv")','train = pd.read_csv(public_dir / "train.csv").iloc[::8].reset_index(drop=True)',1)
s = s.replace('test = pd.read_csv(public_dir / "test.csv")','test = pd.read_csv(public_dir / "test.csv").iloc[:60].reset_index(drop=True)',1)
open("dev/verify_sol.py","w",encoding="utf-8",newline="\n").write(s)
print("ok; MEMBERS in copy:", "MEMBERS = [" in s, "| calib epochs 1:", '"epochs": 1, "seed": 21' in s)
