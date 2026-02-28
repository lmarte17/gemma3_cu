from datasets import load_dataset

try:
    ds = load_dataset('GUI-Libra/GUI-Libra-81K-RL', split='train', streaming=True)
    row = next(iter(ds))
    print('KEYS:', row.keys())
except Exception as e:
    print('ERROR:', e)
