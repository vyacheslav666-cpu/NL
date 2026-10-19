"""nets_new.npz -> base64(gzip(float32)) в порядке, который ждёт загрузчик артефакта, + STATS."""
import base64, gzip, io, json, sys, numpy as np, data

RES, HID = (8, 10, 16), 64
w = np.load(sys.argv[1] if len(sys.argv) > 1 else 'nets_new.npz')

buf = bytearray()
for n in RES:
    NIN = n*n
    assert w[f'{n}_W1'].shape == (HID, NIN) and w[f'{n}_W2'].shape == (10, HID)
    for k in ('W1', 'b1', 'W2', 'b2'):
        buf += np.ascontiguousarray(w[f'{n}_{k}'], '<f4').tobytes()
raw = bytes(buf)
gz = io.BytesIO()
with gzip.GzipFile(fileobj=gz, mode='wb', compresslevel=9, mtime=0) as f:
    f.write(raw)
b64 = base64.b64encode(gz.getvalue()).decode()
open('nets_b64.txt', 'w').write(b64)
print(f'float32: {len(raw)//4}, gzip: {len(gz.getvalue())} B, base64: {len(b64)} символов')

xt, yt = data.load('test')
stats = {}
for n in RES:
    X = np.concatenate([data.to_grid(xt[i:i+2500], n) for i in range(0, len(xt), 2500)])
    z = np.maximum(X @ w[f'{n}_W1'].T + w[f'{n}_b1'], 0) @ w[f'{n}_W2'].T + w[f'{n}_b2']
    pred = z.argmax(1)
    per = [round(float(100*(pred[yt == c] == c).mean()), 1) for c in range(10)]
    cm = np.zeros((10, 10), int)
    for t, p in zip(yt, pred): cm[t, p] += 1
    np.fill_diagonal(cm, 0)
    flat = sorted(((int(cm[t, p]), t, p) for t in range(10) for p in range(10)), reverse=True)[:3]
    stats[str(n)] = {'acc': round(float(100*(pred == yt).mean()), 2), 'per': per,
                     'pairs': [[t, p, c] for c, t, p in flat]}
    print(f'{n}x{n}: acc {stats[str(n)]["acc"]}  9:{per[9]}  пары {stats[str(n)]["pairs"]}')
open('stats.json', 'w').write(json.dumps(stats, separators=(', ', ': ')))
