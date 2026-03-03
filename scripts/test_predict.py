from src.inference import RGCNInference

model = RGCNInference()

smiles_1 = list(model.drug_to_id.keys())[0]
smiles_2 = list(model.drug_to_id.keys())[1]
side = list(model.side_to_id.keys())[0]

prob = model.predict(smiles_1, smiles_2, side)

print(f"Probability: {prob:.4f}")