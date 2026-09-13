"""Install the Metal shaders for Astris 1.0.18, build 3814."""

from pathlib import Path
import hashlib
import json
import os
import plistlib
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
APP = Path('/Applications/Astris.app')
DATA = Path.home() / 'Library/Containers/V380-Ori.Astris/Data/Documents/SMOShaders'
MOD = Path.home() / 'Library/Application Support/mods/contents/0100000000010000/SMO Gameplay Tweaks/native-shaders'
LOAD_PATH = '@executable_path/../Frameworks/libSMOShaderLoader.dylib'
MANIFEST = DATA / 'installation.json'
CHANGED = [
    'Contents/Frameworks/libSMOShaderLoader.dylib',
    'Contents/_CodeSignature/CodeResources',
    'Contents/MacOS/Astris',
]


def run(*args):
    return subprocess.run(args, check=True, capture_output=True).stdout


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
        temporary = Path(f.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def add_library(binary):
    data = bytearray(binary.read_bytes())
    magic, cpu, subtype, kind, count, length, flags, reserved = struct.unpack_from('<8I', data)
    if magic != 0xFEEDFACF or cpu != 0x0100000C:
        raise RuntimeError('Expected a native ARM64 executable.')
    end = 32 + length
    first_section = len(data)
    position = 32
    for _ in range(count):
        command, size = struct.unpack_from('<II', data, position)
        if command == 0xC:
            name_offset = struct.unpack_from('<I', data, position + 8)[0]
            name = data[position + name_offset:position + size].split(b'\0', 1)[0]
            if name == LOAD_PATH.encode():
                raise RuntimeError('This executable already has the shader loader.')
        if command == 0x19:
            sections = struct.unpack_from('<I', data, position + 64)[0]
            for index in range(sections):
                offset = struct.unpack_from('<I', data, position + 72 + index * 80 + 48)[0]
                if offset:
                    first_section = min(first_section, offset)
        position += size
    name = LOAD_PATH.encode() + b'\0'
    size = (24 + len(name) + 7) & ~7
    if end != position or end + size > first_section or any(data[end:end + size]):
        raise RuntimeError('The executable has no suitable load command space.')
    command = struct.pack('<6I', 0xC, size, 24, 0, 0x10000, 0x10000) + name
    data[end:end + size] = command.ljust(size, b'\0')
    struct.pack_into('<II', data, 16, count + 1, length + size)
    binary.write_bytes(data)


def restore():
    manifest = json.loads(MANIFEST.read_text())
    for relative in CHANGED:
        target = APP / relative
        if not target.exists() or sha(target) != manifest['installed'][relative]:
            raise RuntimeError('Astris changed after installation. The original backup remains available.')
    backup = Path(manifest['backup'])
    for relative in reversed(CHANGED):
        original = backup / relative
        if original.exists():
            atomic_copy(original, APP / relative)
    loader = APP / CHANGED[0]
    if not (backup / CHANGED[0]).exists():
        atomic_copy(loader, DATA / 'removed-libSMOShaderLoader.dylib')
        loader.unlink()
    settings = json.loads((DATA / 'preset.json').read_text())
    settings['enabled'] = False
    (DATA / 'preset.json').write_text(json.dumps(settings, indent=2) + '\n')
    run('codesign', '--verify', '--deep', '--strict', str(APP))
    os.replace(MANIFEST, DATA / 'restored-installation.json')
    print('Original Astris restored. The current shader effects are disabled.')


def install():
    info = plistlib.loads((APP / 'Contents/Info.plist').read_bytes())
    if info['CFBundleIdentifier'] != 'V380-Ori.Astris' or str(info['CFBundleVersion']) != '3814':
        raise RuntimeError('This loader requires Astris 1.0.18, build 3814.')
    if MANIFEST.exists():
        print('The shader loader is already installed.')
        return
    run('codesign', '--verify', '--deep', '--strict', str(APP))
    entitlements = run('codesign', '-d', '--entitlements', ':-', str(APP))
    original_entitlements = plistlib.loads(entitlements)
    DATA.mkdir(parents=True, exist_ok=True)
    backup_root = Path(tempfile.mkdtemp(prefix='app-backup-', dir=DATA))
    backup = backup_root / 'Astris.app'
    run('ditto', str(APP), str(backup))
    with tempfile.TemporaryDirectory(prefix='shader-stage-', dir=DATA) as temp:
        stage = Path(temp) / 'Astris.app'
        run('ditto', str(APP), str(stage))
        entitlement_file = Path(temp) / 'entitlements.plist'
        entitlement_file.write_bytes(entitlements)
        atomic_copy(ROOT / 'libSMOShaderLoader.dylib', stage / CHANGED[0])
        add_library(stage / 'Contents/MacOS/Astris')
        run('codesign', '--force', '--sign', '-', '--options', 'runtime', '--entitlements', str(entitlement_file), str(stage))
        run('codesign', '--verify', '--deep', '--strict', str(stage))
        stage_entitlements = plistlib.loads(run('codesign', '-d', '--entitlements', ':-', str(stage)))
        if stage_entitlements != original_entitlements:
            raise RuntimeError('The staged entitlements differ from the original entitlements.')
        for filename in ['libSMOMetalBridge.dylib', 'libSMOCinematic.dylib', 'Cinematic.metal', 'preset.json']:
            atomic_copy(ROOT / filename, DATA / filename)
        installed = {relative: sha(stage / relative) for relative in CHANGED}
        manifest = {'appVersion': info['CFBundleShortVersionString'], 'appBuild': str(info['CFBundleVersion']),
                    'backup': str(backup), 'installed': installed, 'loadPath': LOAD_PATH,
                    'originalExecutableSHA256': sha(backup / 'Contents/MacOS/Astris')}
        try:
            for relative in CHANGED:
                atomic_copy(stage / relative, APP / relative)
            run('codesign', '--verify', '--deep', '--strict', str(APP))
        except Exception:
            for relative in reversed(CHANGED):
                if (backup / relative).exists():
                    atomic_copy(backup / relative, APP / relative)
            if not (backup / CHANGED[0]).exists():
                (APP / CHANGED[0]).unlink(missing_ok=True)
            raise
        MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    MOD.mkdir(parents=True, exist_ok=True)
    files = ['MetalBridge.m', 'Cinematic.m', 'Cinematic.metal', 'ShaderLoader.m', 'preset.json',
             'libSMOMetalBridge.dylib', 'libSMOCinematic.dylib', 'libSMOShaderLoader.dylib',
             'install_shaders.py', 'shader_control.py', 'README.md',
             'Enable Shaders.command', 'Disable Shaders.command']
    for filename in files:
        atomic_copy(ROOT / filename, MOD / filename)
    print('Shader loader installed. Astris signature and entitlements passed.')
    print('Original application backup:', backup)
    print('Shader files:', MOD)


if __name__ == '__main__':
    if '--restore' in sys.argv:
        restore()
    else:
        install()
