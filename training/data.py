"""Аугментация MNIST + препроцессинг, повторяющий extract() из приложения."""
import gzip, numpy as np

FRAC = 0.82          # доля кадра, которую занимает нарисованное (см. side в extract)

def load(split):
    p = 'train' if split == 'train' else 't10k'
    with gzip.open(f'{p}-images-idx3-ubyte.gz') as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)
    with gzip.open(f'{p}-labels-idx1-ubyte.gz') as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return x, y

def _sample(img, sx, sy):
    """Билинейная выборка. img (B,H,W); sx, sy (B,...) в пикселях. Вне кадра — 0."""
    B, H, W = img.shape
    x0 = np.floor(sx).astype(np.int32); y0 = np.floor(sy).astype(np.int32)
    fx = (sx - x0).astype(np.float32); fy = (sy - y0).astype(np.float32)
    out = np.zeros(sx.shape, np.float32)
    bi = np.arange(B).reshape((B,) + (1,) * (sx.ndim - 1))
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
            out += np.where(ok, img[bi, np.clip(yi, 0, H-1), np.clip(xi, 0, W-1)] * w, 0)
    return out

def augment(img, rng, rot=12., shear=.25, scale=.15, bend=.32, vbend=.14, thick=.35):
    """Аффинные искажения + изгиб (разные школы письма) + толщина штриха."""
    B, H, W = img.shape
    cy, cx = (H - 1) / 2, (W - 1) / 2
    th  = np.radians(rng.uniform(-rot, rot, B)).astype(np.float32)
    sh  = rng.uniform(-shear, shear, B).astype(np.float32)
    kx  = np.exp(rng.uniform(-scale, scale, B)).astype(np.float32)
    ky  = np.exp(rng.uniform(-scale, scale, B)).astype(np.float32)
    bx  = rng.uniform(-bend, bend, B).astype(np.float32)      # низ уводит вбок
    by  = rng.uniform(-vbend, vbend, B).astype(np.float32)    # бока уводит вниз/вверх

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
    X = (xx - cx)[None]; Y = (yy - cy)[None]
    # обратное преобразование: сначала снимаем изгиб, затем аффинную часть
    u = X - bx[:, None, None] * (Y / cy) ** 2 * cx
    v = Y - by[:, None, None] * (X / cx) ** 2 * cy
    c, s = np.cos(-th)[:, None, None], np.sin(-th)[:, None, None]
    u, v = u * c - v * s, u * s + v * c
    u = u - sh[:, None, None] * v
    u = u / kx[:, None, None]; v = v / ky[:, None, None]
    out = _sample(img, u + cx, v + cy)

    # толще / тоньше штрих — мягкое смешивание с max- и min-фильтром 3x3
    p = np.pad(out, ((0,0),(1,1),(1,1)))
    st = np.stack([p[:, dy:dy+H, dx:dx+W] for dy in (0,1,2) for dx in (0,1,2)])
    fat, thin = st.max(0), st.min(0)
    a = rng.uniform(-thick, thick, B).astype(np.float32)[:, None, None]
    out = np.where(a > 0, out + a * (fat - out), out + (-a) * (thin - out))
    return out

def to_grid(img, N, k=3, frac=FRAC):
    """28x28 -> N*N по правилам extract(): bbox, вписывание, area-даунсемпл, max-норма."""
    B, H, W = img.shape
    m = img > 0
    rows, cols = m.any(2), m.any(1)
    empty = ~m.any((1, 2))
    y0 = rows.argmax(1); y1 = H - rows[:, ::-1].argmax(1)
    x0 = cols.argmax(1); x1 = W - cols[:, ::-1].argmax(1)
    ext = np.maximum(np.maximum(y1 - y0, x1 - x0), 1).astype(np.float32)
    frame = ext / frac
    cxb = (x0 + x1 - 1) / 2; cyb = (y0 + y1 - 1) / 2

    g = (np.arange(N, dtype=np.float32)[:, None] + (np.arange(k, dtype=np.float32) + .5) / k).ravel() / N
    gy, gx = np.meshgrid(g, g, indexing='ij')
    sx = cxb[:, None, None] + (gx[None] - .5) * frame[:, None, None]
    sy = cyb[:, None, None] + (gy[None] - .5) * frame[:, None, None]
    s = _sample(img, sx, sy).reshape(B, N, k, N, k).mean((2, 4)) / 255
    mx = s.max((1, 2), keepdims=True)
    s = np.where(mx > 0.02, np.minimum(1, s / np.maximum(mx, 1e-9)), s)
    s[empty] = 0
    return s.reshape(B, N * N).astype(np.float32)
