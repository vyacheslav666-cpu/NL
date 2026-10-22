import os, re, sys
DST = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'index.html')
html = open(DST).read()
b64 = open('nets_b64.txt').read().strip()
stats = open('stats.json').read().strip()

html, n1 = re.subn(r'(const NETS_B64 = ")[^"]+(";)', lambda m: m.group(1) + b64 + m.group(2), html, count=1)
html, n2 = re.subn(r'(const STATS = )\{.*?\};\n', lambda m: m.group(1) + stats + ';\n', html, count=1, flags=re.S)
assert n1 == 1 and n2 == 1, (n1, n2)
open(DST, 'w').write(html)
print(f'записан {DST}: {len(html)} байт, base64 {len(b64)}')
