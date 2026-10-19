import numpy as np, time, sys, data

CLEAN = 0.25
RES, HID, EPOCHS, BS = (8, 10, 16), 64, int(sys.argv[1]) if len(sys.argv) > 1 else 30, 256
rng = np.random.default_rng(1)

xs, ys = data.load('train')
xt, yt = data.load('test')
xtr, ytr, xva, yva = xs[:55000], ys[:55000], xs[55000:], ys[55000:]
GVA = {n: data.to_grid(xva, n) for n in RES}
GTE = {n: np.concatenate([data.to_grid(xt[i:i+2500], n) for i in range(0, len(xt), 2500)]) for n in RES}

class MLP:
    def __init__(s, nin):
        s.W1 = (rng.standard_normal((HID, nin)) * np.sqrt(2/nin)).astype(np.float32)
        s.b1 = np.zeros(HID, np.float32)
        s.W2 = (rng.standard_normal((10, HID)) * np.sqrt(2/HID)).astype(np.float32)
        s.b2 = np.zeros(10, np.float32)
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
    perm = rng.permutation(len(xtr))
    for c in range(0, len(perm), 5500):
        idx = perm[c:c+5500]
        a = data.augment(xtr[idx], rng, thick=.45)
        keep = rng.random(len(idx)) < CLEAN          # часть примеров идёт без искажений
        a[keep] = xtr[idx][keep]
        yb = ytr[idx]
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

out = {}
print('\n=== тест (10 000, чистый MNIST) ===')
for n in RES:
    w = best[n][1]
    for k, v in w.items(): out[f'{n}_{k}'] = v
    z = np.maximum(GTE[n] @ w['W1'].T + w['b1'], 0) @ w['W2'].T + w['b2']
    pred = z.argmax(1)
    per = [100*(pred[yt==c]==c).mean() for c in range(10)]
    print(f'{n}x{n}: {100*(pred==yt).mean():.2f}   по цифрам ' + ' '.join(f'{c}:{v:.1f}' for c,v in enumerate(per)))
np.savez('nets_v2.npz', **out)
print('saved nets_v2.npz')
