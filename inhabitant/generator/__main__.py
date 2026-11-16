"""Start the generator service with Kimodo's environment:  python -m inhabitant.generator --rpc <dir> [--shared] [--device cuda]"""
import os, sys

from . import kimodo_env

os.execvpe(sys.executable, [sys.executable, "-m", "inhabitant.generator.gen_server", *sys.argv[1:]], kimodo_env())
