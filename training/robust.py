"""Точность на MNIST, искажённом фиксированным набором деформаций (другой seed, чем при обучении)."""
import numpy as np, sys, data
w = np.load(sys.argv[1])
xt, yt = data.load('test')
rng = np.random.default_rng(12345)
accs = {}
for n in (8, 10, 16):
    r = np.random.default_rng(999)        # одни и те же искажения для всех моделей
    P, Y = [], []
    for i in range(0, len(xt), 2500):
        a = data.augment(xt[i:i+2500], r, thick=.45)
        X = data.to_grid(a, n)
        z = np.maximum(X @ w[f'{n}_W1'].T + w[f'{n}_b1'], 0) @ w[f'{n}_W2'].T + w[f'{n}_b2']
        P.append(z.argmax(1)); Y.append(yt[i:i+2500])
    P, Y = np.concatenate(P), np.concatenate(Y)
    accs[n] = (100*(P == Y).mean(), 100*(P[Y == 9] == 9).mean())
print(sys.argv[1], ' '.join(f'{n}: все {a:.2f} / девятки {b:.1f}' for n, (a, b) in accs.items()))
