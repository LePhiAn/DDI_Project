import sys
import os
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from web_app.services.mapping_service import MappingService
    from web_app.services.model_loader import ModelLoader
    from web_app.services.predictor_service import PredictorService
    import torch
    
    print('MappingService instantiation...')
    ms = MappingService()
    drug_to_id, side_to_id = ms.get_maps()
    print('num_nodes, num_relations:', ms.num_nodes, ms.num_relations)
    drug_list = ms.get_drug_list()
    sample = drug_list[:2]
    print('Sample drugs:', sample)

    print('Loading model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ml = ModelLoader(model_path=os.path.join(ROOT, 'models', 'r_gcn_full_model.pth'),
                     num_nodes=ms.num_nodes,
                     num_relations=ms.num_relations,
                     hidden_channels=64,
                     embedding_dim=16,
                     device=device)
    model = ml.get_model()
    print('Model loaded')

    print('Creating predictor...')
    pred = PredictorService(model=model, drug_map=drug_to_id, side_map=side_to_id, device=device, side_to_vn=ms.side_to_vn)
    print('Predicting pair...')
    res = pred.predict_pair(sample[0], sample[1])
    print('Result:')
    print(res)

except Exception:
    print('EXCEPTION:')
    traceback.print_exc()
    sys.exit(2)

print('DIAGNOSTIC DONE')
