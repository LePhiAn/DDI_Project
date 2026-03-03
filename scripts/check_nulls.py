import os
import sys

found = False
for root, dirs, files in os.walk('web_app'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            with open(path, 'rb') as fin:
                data = fin.read()
            if b'\x00' in data:
                print('NULL BYTE in', path)
                found = True
if not found:
    print('no null bytes detected')
