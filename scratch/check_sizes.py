import os

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                size = os.path.getsize(fp)
                if size > 1 * 1024 * 1024: # Only show files > 1MB
                    print(f"{fp}: {size / (1024*1024):.2f} MB")

print("--- File sizes > 1MB ---")
get_size('data')
get_size('models')
