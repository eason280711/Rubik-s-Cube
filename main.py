import config
import os

file = config.file
path_to_file = os.path.join(os.path.dirname(__file__), file)
os.system(f"python {path_to_file}")
