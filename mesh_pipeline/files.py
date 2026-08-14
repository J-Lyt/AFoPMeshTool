"""Filesystem and byte-stream helpers shared by MMB workflows."""

import io
import os
import re

def _mod_file_output(src_path: str, overwrite: bool = False) -> str:
    """
    Determine the output file path with overwrite protection.

    ``src_path`` supplies the output filename family; an existing ``_MOD``
    suffix is normalized first.  When overwrite is enabled the canonical
    ``_MOD`` file is replaced; otherwise an existing output is protected by
    incrementing the suffix (``_MOD1``, ``_MOD2``, ...).  Export callers guard
    a separately retained source path from ever being selected as the output.
    """
    stem, extension = os.path.splitext(src_path)
    directory, filename_stem = os.path.split(stem)
    base_stem = _strip_mod_suffix(filename_stem)
    base = os.path.join(directory, base_stem + "_MOD" + extension)
    if overwrite or not os.path.isfile(base):
        return base
    i = 1
    while True:
        candidate = os.path.join(
            directory, f"{base_stem}_MOD{i}{extension}")
        if not os.path.isfile(candidate):
            return candidate
        i += 1


def source_setting_path(settings, current_name, source_name):
    """Return an optional retained source path, falling back to the UI path."""
    source = getattr(settings, source_name, "")
    return source.strip() if source and source.strip() else getattr(
        settings, current_name, "")

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
