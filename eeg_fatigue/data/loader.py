import os

def get_valid_subjects(after_root, before_root):
    """Find subjects with matching before & after folders"""
    after_subs = set(os.listdir(after_root)) if os.path.isdir(after_root) else set()
    before_subs = set(os.listdir(before_root)) if os.path.isdir(before_root) else set()
    
    valid = sorted(after_subs & before_subs)

    print(f"Found {len(after_subs)} subjects in 'after', {len(before_subs)} in 'before'.")
    print(f"Subjects with BOTH conditions ({len(valid)}): {valid}")

    return valid
