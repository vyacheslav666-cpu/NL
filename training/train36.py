"""36 классов: цифры 0-9 (MNIST + EMNIST) и буквы A-Z (EMNIST, строчные слиты с заглавными).

python train36.py ЭПОХ СКРЫТЫХ РАЗРЕШЕНИЯ [ВЫХОД]
python train36.py 40 128 10,16,20 nets36.npz
"""
import numpy as np, time, sys, data

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
HID    = int(sys.argv[2]) if len(sys.argv) > 2 else 128
RES    = tuple(int(v) for v in sys.argv[3].split(',')) if len(sys.argv) > 3 else (10, 16, 20)
OUT    = sys.argv[4] if len(sys.argv) > 4 else 'nets36.npz'
NCLS, BS, CLEAN, PER, CHUNK = len(data.CLASSES), 256, 0.25, 5000, 6000
rng = np.random.default_rng(1)

xm, ym = data.load('train')
xe, ye = data.load_emnist('train')
xs = np.concatenate([xm.astype(np.uint8), xe]); ys = np.concatenate([ym, ye])
hold = rng.random(len(xs)) < 0.05                       # валидация: 5% обучающих
xtr, ytr, xva, yva = xs[~hold], ys[~hold], xs[hold], ys[hold]
del xs, ys, xm, xe
byc = [np.flatnonzero(ytr == c) for c in range(NCLS)]
VPC = np.bincount(yva).min()                            # поровну на класс, по редкой букве
va = np.concatenate([rng.choice(np.flatnonzero(yva == c), VPC, replace=False) for c in range(NCLS)])
xva, yva = xva[va].astype(np.float32), yva[va]
GVA = {n: data.to_grid(xva, n) for n in RES}
print(f'обучение {len(ytr)}, валидация {len(yva)} (по {VPC} на класс), за эпоху {PER*NCLS}', flush=True)

class MLP:
    def __init__(s, nin):
        s.W1 = (rng.standard_normal((HID, nin)) * np.sqrt(2/nin)).astype(np.float32)
        s.b1 = np.zeros(HID, np.float32)
        s.W2 = (rng.standard_normal((NCLS, HID)) * np.sqrt(2/HID)).astype(np.float32)
        s.b2 = np.zeros(NCLS, np.float32)
        s.m = {k: np.zeros_like(getattr(s, k)) for k in ('W1','b1','W2','b2')}
        s.v = {k: np.zeros_like(getattr(s, k)) for k in ('W1','b1','W2','b2')}
        s.t = 0
    def logits(s, X):
        return np.maximum(X @ s.W1.T + s.b1, 0) @ s.W2.T + s.b2
    def step(s, X, y, lr, wd=5e-5):
        B = len(X)
        h = np.maximum(X @ s.W1.T + s.b1, 0)
        z = h @ s.W2.T + s.b2
        z -= z.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
        d = p.copy(); d[np.arange(B), y] -= 1; d /= B
        g = {'W2': d.T @ h + wd*s.W2, 'b2': d.sum(0)}
        dh = (d @ s.W2) * (h > 0)
        g['W1'] = dh.T @ X + wd*s.W1; g['b1'] = dh.sum(0)
        s.t += 1
        for k, gk in g.items():
            s.m[k] = .9*s.m[k] + .1*gk
            s.v[k] = .999*s.v[k] + .001*gk*gk
            mh = s.m[k]/(1-.9**s.t); vh = s.v[k]/(1-.999**s.t)
            setattr(s, k, getattr(s, k) - lr*mh/(np.sqrt(vh)+1e-8))

nets = {n: MLP(n*n) for n in RES}
best = {n: (0, None) for n in RES}
t0 = time.time()
for ep in range(EPOCHS):
    lr = 1e-4 + .5*(3e-3 - 1e-4)*(1 + np.cos(np.pi*ep/EPOCHS))
    # поровну каждого класса: цифр в данных в 30-70 раз больше, чем редких букв
    idx = np.concatenate([rng.choice(b, PER, replace=len(b) < PER) for b in byc])
    rng.shuffle(idx)
    for c in range(0, len(idx), CHUNK):
        part = idx[c:c+CHUNK]
        src = xtr[part].astype(np.float32)
        a = data.augment(src, rng, thick=.45)
        keep = rng.random(len(part)) < CLEAN
        a[keep] = src[keep]
        yb = ytr[part]
        for n in RES:
            X = data.to_grid(a, n, k=2)
            for b in range(0, len(X), BS):
                nets[n].step(X[b:b+BS], yb[b:b+BS], lr)
    line = []
    for n in RES:
        acc = 100*(nets[n].logits(GVA[n]).argmax(1) == yva).mean()
        if acc > best[n][0]:
            best[n] = (acc, {k: getattr(nets[n], k).copy() for k in ('W1','b1','W2','b2')})
        line.append(f'{n}:{acc:.2f}')
    print(f'ep {ep+1:2d}/{EPOCHS} lr {lr:.4f} val ' + ' '.join(line) + f'  [{time.time()-t0:.0f}s]', flush=True)

out = {f'{n}_{k}': v for n in RES for k, v in best[n][1].items()}
np.savez(OUT, **out)
print('saved', OUT)
