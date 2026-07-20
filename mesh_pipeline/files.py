"""Filesystem and byte-stream helpers shared by MMB workflows."""

import io
import os
import re

def _mod_file_output(src_path: str, overwrite: bool = False) -> str:
    """
    Determine the output file path with overwrite protection.

    - If overwrite is True, return 'src_path' directly (overwrite the loaded file).
    - If src_path already contains '_MOD' in its stem, return it directly. (overwrite it)
    - If '<stem>_MOD.mmb' doesn't exist, return it. If it does, increment: '<stem>_MOD1.mmb', '<stem>_MOD2.mmb', etc.
    """
    if overwrite:
        return src_path
    stem, _ = os.path.splitext(src_path)
    # If already a _MOD file, overwrite it
    if '_MOD' in os.path.basename(stem):
        return src_path
    base = stem + "_MOD.mmb"
    if not os.path.isfile(base):
        return base
    i = 1
    while True:
        candidate = f"{stem}_MOD{i}.mmb"
        if not os.path.isfile(candidate):
            return candidate
        i += 1

def CopyFile(read,write,offset,size,buffer_size=500000):
    read.seek(offset)
    chunks = size // buffer_size
    for o in range(chunks):
        write.write(read.read(buffer_size))
    write.write(read.read(size%buffer_size))
def get_merged_mmb(mmb):
    files = []
    if str(mmb).endswith("mmb"):
        files.append(mmb)
    else:
        i = 0
        while True:
            current_file = f"{str(mmb)[:-1]}{i}"
            if os.path.isfile(current_file):
                files.append(current_file)
                i += 1
            else:
                break

    f = io.BytesIO()
    for file_dir in files:
        with open(file_dir, 'rb') as file:
            f.write(file.read())
    return f

_MOD_SUFFIX_RE = re.compile(r'^(.*?)(_MOD\d*)$')

def _strip_mod_suffix(stem):
    """'head_MOD' -> 'head', 'head_MOD1' -> 'head', 'head' -> 'head' (no change)."""
    m = _MOD_SUFFIX_RE.match(stem)
    return m.group(1) if m else stem
