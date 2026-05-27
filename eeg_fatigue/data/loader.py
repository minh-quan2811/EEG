import os
from ..config import SUBJECTS

def get_valid_subjects(after_root, before_root):
    after_subs = set(os.listdir(after_root)) if os.path.isdir(after_root) else set()
    before_subs = set(os.listdir(before_root)) if os.path.isdir(before_root) else set()
    valid = sorted(after_subs & before_subs)
    if SUBJECTS:
        requested = set(SUBJECTS)
        missing = requested - set(valid)
        if missing:
            print(f"  [WARN] Subjects in config not found in data: {sorted(missing)}")
        valid = sorted(requested & set(valid))
        print(f"  Using subject filter from config: {valid}")
    else:
        print(f"  No subject filter — using all {len(valid)} valid subjects.")
    print(f"  Found {len(after_subs)} in 'after', "
          f"{len(before_subs)} in 'before', "
          f"{len(valid)} will be processed: {valid}")
    return valid