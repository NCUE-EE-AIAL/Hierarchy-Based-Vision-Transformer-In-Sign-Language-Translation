from glob import glob
import os

def find_files(directory, pattern='**/*.*', interval=1):
    """Recursively finds all files matching the pattern and returns every other file."""
    files = glob(os.path.join(directory, pattern), recursive=True)
    return files[::interval]