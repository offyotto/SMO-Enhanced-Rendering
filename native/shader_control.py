"""Enable or disable the installed shader preset without a game restart."""
from pathlib import Path
import json
import os
import sys
import tempfile

root = Path.home() / 'Library/Containers/V380-Ori.Astris/Data/Documents/SMOShaders'
path = root / 'preset.json'
settings = json.loads(path.read_text())
settings['enabled'] = len(sys.argv) < 2 or sys.argv[1] != 'off'
with tempfile.NamedTemporaryFile('w', dir=root, delete=False) as stream:
    json.dump(settings, stream, indent=2)
    stream.write('\n')
os.replace(stream.name, path)
print('Shaders enabled.' if settings['enabled'] else 'Shaders disabled.')
