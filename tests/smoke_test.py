import os
import sys
import random
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from web_app.services.mapping_service import MappingService
    from web_app.services.model_loader import ModelLoader
    from web_app.services.predictor_service import PredictorService
    import torch

    print('Initializing services...')
    ms = MappingService()
    drug_to_id, side_to_id = ms.get_maps()
    drug_list = ms.get_drug_list()
    print('Drugs available:', len(drug_list))
    print('Sides available:', len(side_to_id))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ml = ModelLoader(
        model_path=os.path.join(ROOT, 'models', 'r_gcn_full_model.pth'),
        num_nodes=ms.num_nodes,
        num_relations=ms.num_relations,
        hidden_channels=64,
        embedding_dim=16,
        device=device
    )
    model = ml.get_model()
    pred = PredictorService(model=model, drug_map=drug_to_id, side_map=side_to_id, device=device, side_to_vn=ms.side_to_vn)

    # 1) 10 random pair predictions
    print('\n=== Running 10 random pair predictions ===')
    if len(drug_list) < 2:
        raise RuntimeError('Not enough drugs in mapping to run pair tests')

    for i in range(10):
        a, b = random.sample(drug_list, 2)
        res = pred.predict_pair(a, b)
        top = res.get('top_10_risks', [])[:3]
        print(f'{i+1}. Pair: {a} | {b} -> top risks:')
        for t in top:
            print('   -', t.get('side_effect'), f"(p={t.get('probability'):.3f})")

    # 2) Single-drug embedding check
    print('\n=== Single-drug embedding check ===')
    sample_drug = drug_list[0]
    node_id = drug_to_id[sample_drug]
    # verify id in range, no need to call model directly
    if node_id < 0 or node_id >= ms.num_nodes:
        raise ValueError('node_id out of range')
    print('Sample drug:', sample_drug, 'node_id:', node_id)

    # 3) Side effect catalog sample
    print('\n=== Side effect catalog sample ===')
    side_names = list(side_to_id.keys())
    print('Total side effects:', len(side_names))
    print('Sample 5 sides:', side_names[:5])

    print('\nSMOKE TEST PASSED')

except Exception:
    print('SMOKE TEST FAILED')
    traceback.print_exc()
    sys.exit(2)

sys.exit(0)
