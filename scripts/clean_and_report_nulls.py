import os

ROOT='web_app'
modified=[]
for dirpath,dirnames,filenames in os.walk(ROOT):
    for fn in filenames:
        if fn.endswith('.py'):
            path=os.path.join(dirpath,fn)
            with open(path,'rb') as f:
                data=f.read()
            if b'\x00' in data:
                bak=path+'.bak'
                print('Found NULL bytes in',path,'-> backup',bak)
                with open(bak,'wb') as b:
                    b.write(data)
                cleaned=bytes([bb for bb in data if bb!=0])
                with open(path,'wb') as f:
                    f.write(cleaned)
                modified.append(path)
if not modified:
    print('No files modified; no null bytes detected')
else:
    print('Cleaned files:')
    for m in modified:
        print('-',m)
