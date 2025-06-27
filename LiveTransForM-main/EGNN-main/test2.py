import pandas as pd

df = pd.read_csv("data/one_organic_ip_ea_rp_test.csv")
df = df.drop(columns=["Name", "InChI", "CAS Link", "IE / eV","EA / eV", "pubchem_smiles", "Reaction", "Solvent"])
df.to_csv("data/one_organic_ip_ea_rp_test.csv", index=False)