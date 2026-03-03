import importlib, traceback
print('Import test:')
for mod in ('numpy','torch','torch_geometric','torch_scatter','torch_sparse'):
    try:
        importlib.import_module(mod)
        print(mod, 'OK')
    except Exception as e:
        print(mod, 'ERROR')
        traceback.print_exc()
