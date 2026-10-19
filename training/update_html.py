import json, re, sys
SRC = '/root/.claude/projects/-home-user-NL/f7f0521c-69c1-529c-b447-9585f190f03c/tool-results/artifact-726d7d2a-1789580786-a76b.html'
DST = sys.argv[1] if len(sys.argv) > 1 else '/home/user/NL/index.html'
html = open(SRC).read()
b64 = open('nets_b64.txt').read().strip()
stats = open('stats.json').read().strip()

html, n1 = re.subn(r'(const NETS_B64 = ")[^"]+(";)', lambda m: m.group(1) + b64 + m.group(2), html, count=1)
html, n2 = re.subn(r'(const STATS = )\{.*?\};\n', lambda m: m.group(1) + stats + ';\n', html, count=1, flags=re.S)
assert n1 == 1 and n2 == 1, (n1, n2)
open(DST, 'w').write(html)
print(f'записан {DST}: {len(html)} байт, base64 {len(b64)}')
