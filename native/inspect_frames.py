"""Create a small diagnostic image from the captured Metal buffers."""
from pathlib import Path
import json
import struct
import sys
import zlib

capture=Path(sys.argv[1])
output=Path(sys.argv[2])
files=sys.argv[3:]
info=json.loads((capture/'capture.json').read_text())
w,h=768,432
rows=[bytearray([0]) for _ in range(h)]
for arg in files:
    linear=arg.startswith('linear:')
    data=Path(arg.removeprefix('linear:')).read_bytes()
    for y in range(h):
        offset=int(y*info['colorHeight']/h)*info['colorStride']
        for x in range(w):
            color=struct.unpack_from('<eeee',data,offset+int(x*info['colorWidth']/w)*8)[:3]
            if linear:
                rows[y].extend(round(max(0,min(1,v))*255) for v in color)
            else:
                rows[y].extend(round(max(0,min(1,v*1.5))**(1/2.2)*255) for v in color)
def chunk(kind,data):
    return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w*len(files),h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(rows)))+chunk(b'IEND',b'')
output.write_bytes(png)
