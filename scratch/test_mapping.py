import sys
import os
sys.path.insert(0, os.getcwd())
from web_app.services.mapping_service import MappingService


try:
    mapping = MappingService()
    print("✅ MappingService initialized successfully!")
    print(f"Drugs: {len(mapping.get_drug_list())}")
    print(f"Sides: {len(mapping.get_side_options())}")
except Exception as e:
    print(f"❌ Error: {e}")
