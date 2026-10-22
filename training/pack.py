"""nets36.npz -> base64(gzip(float16)) в порядке, который ждёт загрузчик страницы, + STATS."""
import base64, gzip, io, json, sys, numpy as np, data

RES, NC = (8, 10, 16), len(data.CLASSES)
w = np.load(sys.argv[1] if len(sys.argv) > 1 else 'nets36.npz')
HID = w['16_b1'].shape[0]

buf = bytearray()
for n in RES:
    assert w[f'{n}_W1'].shape == (HID, n*n) and w[f'{n}_W2'].shape == (NC, HID)
    for k in ('W1', 'b1', 'W2', 'b2'):
        buf += np.ascontiguousarray(w[f'{n}_{k}'], '<f2').tobytes()
raw = bytes(buf)
gz = io.BytesIO()
with gzip.GzipFile(fileobj=gz, mode='wb', compresslevel=9, mtime=0) as f:
    f.write(raw)
b64 = base64.b64encode(gz.getvalue()).decode()
open('nets_b64.txt', 'w').write(b64)
print(f'float16: {len(raw)//2}, gzip: {len(gz.getvalue())} B, base64: {len(b64)} символов')

# считаем по тем же float16-весам, что поедут в страницу
f16 = {k: w[k].astype(np.float16).astype(np.float32) for k in w.files}
xe, ye = data.load_emnist('test')
xm, ym = data.load('test')
xt = np.concatenate([xe.astype(np.float32), xm]); yt = np.concatenate([ye, ym])
mnist = np.arange(len(yt)) >= len(ye)
stats = {}
for n in RES:
    X = np.concatenate([data.to_grid(xt[i:i+5000], n) for i in range(0, len(xt), 5000)])
    z = np.maximum(X @ f16[f'{n}_W1'].T + f16[f'{n}_b1'], 0) @ f16[f'{n}_W2'].T + f16[f'{n}_b2']
    pred = z.argmax(1)
    per = [round(float(100*(pred[yt == c] == c).mean()), 1) for c in range(NC)]
    cm = np.zeros((NC, NC), int)
    np.add.at(cm, (yt, pred), 1)
    rate = cm / cm.sum(1, keepdims=True)            # доля класса, ушедшая в чужой: классы разного размера
    np.fill_diagonal(rate, 0)
    flat = sorted(((rate[t, p], t, p) for t in range(NC) for p in range(NC)), reverse=True)[:3]
    stats[str(n)] = {'acc': round(float(np.mean(per)), 2), 'per': per,
                     'pairs': [[int(t), int(p), round(float(100*r))] for r, t, p in flat],
                     'mnist': round(float(100*(pred[mnist] == yt[mnist]).mean()), 2)}
    print(f'{n}x{n}: по классам в среднем {stats[str(n)]["acc"]}  MNIST {stats[str(n)]["mnist"]}  '
          f'пары {[(data.CLASSES[t], data.CLASSES[p], r) for t, p, r in stats[str(n)]["pairs"]]}')
print('тестовых примеров:', len(yt))
open('stats.json', 'w').write(json.dumps(stats, separators=(', ', ': ')))
