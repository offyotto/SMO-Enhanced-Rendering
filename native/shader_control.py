"""Control the installed shader preset without restarting the game.

Usage:
  python3 shader_control.py on
  python3 shader_control.py off
  python3 shader_control.py debug <0-9> [mip 0-6]
  python3 shader_control.py normal
  python3 shader_control.py status
"""
from pathlib import Path
import json
import os
import sys
import tempfile

root = Path.home() / 'Library/Containers/V380-Ori.Astris/Data/Documents/SMOShaders'
path = root / 'preset.json'
settings = json.loads(path.read_text())

args = sys.argv[1:]
command = args[0].lower() if args else 'on'
changed = False

if command in {'on', 'enable'}:
    settings['enabled'] = True
    changed = True
elif command in {'off', 'disable'}:
    settings['enabled'] = False
    changed = True
elif command in {'normal', 'reset'}:
    settings['enabled'] = True
    settings['debugView'] = 0
    settings['debugMip'] = 0
    changed = True
elif command == 'debug':
    if len(args) < 2:
        raise SystemExit('Usage: shader_control.py debug <0-9> [mip 0-6]')
    try:
        view = int(args[1])
        mip = int(args[2]) if len(args) > 2 else int(settings.get('debugMip', 0))
    except ValueError as exc:
        raise SystemExit('debug view and mip must be integers') from exc
    if not 0 <= view <= 9:
        raise SystemExit('debug view must be between 0 and 9')
    if not 0 <= mip <= 6:
        raise SystemExit('debug mip must be between 0 and 6')
    settings['enabled'] = True
    settings['debugView'] = view
    settings['debugMip'] = mip
    changed = True
elif command == 'status':
    print(json.dumps(settings, indent=2))
else:
    raise SystemExit('Usage: shader_control.py {on|off|normal|status|debug <0-9> [mip 0-6]}')

if changed:
    with tempfile.NamedTemporaryFile('w', dir=root, delete=False) as stream:
        json.dump(settings, stream, indent=2)
        stream.write('\n')
    os.replace(stream.name, path)

    if command == 'debug':
        print(f"Shaders enabled. debugView={settings['debugView']} debugMip={settings['debugMip']}")
    elif command in {'normal', 'reset'}:
        print('Shaders enabled. Normal composite restored.')
    else:
        print('Shaders enabled.' if settings['enabled'] else 'Shaders disabled.')
