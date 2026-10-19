"""Точная реплика extract() из артефакта: векторные штрихи -> N*N."""
import numpy as np
from PIL import Image, ImageDraw
SCR = 240
def extract(strokes, N):
    pts = np.array([p for st in strokes for p in st], float)
    minX, minY = pts.min(0); maxX, maxY = pts.max(0)
    thick = 0.1*N; cell = SCR/N
    ext = max(maxX-minX, maxY-minY, 1)
    side = 1.22*ext/(1 - 1.22*thick/N)
    k = SCR/side; cx, cy = (minX+maxX)/2, (minY+maxY)/2
    T = lambda p: ((p[0]-cx)*k + SCR/2, (p[1]-cy)*k + SCR/2)
    im = Image.new('L', (SCR, SCR), 0)
    dr = ImageDraw.Draw(im)
    w = max(1, int(round(thick*cell)))
    for st in strokes:
        xy = [T(p) for p in st]
        if len(xy) == 1:
            x, y = xy[0]; r = w/2
            dr.ellipse([x-r, y-r, x+r, y+r], fill=255)
        else:
            dr.line(xy, fill=255, width=w, joint='curve')
            r = w/2
            for x, y in (xy[0], xy[-1]): dr.ellipse([x-r, y-r, x+r, y+r], fill=255)
    a = np.asarray(im, np.float32)
    B = SCR//N
    g = a[:N*B, :N*B].reshape(N, B, N, B).mean((1, 3))/255
    if g.max() > 0.02: g = np.minimum(1, g/g.max())
    return g

d = np.load('nets.npz')
def predict(n, g):
    x = g.ravel()
    a1 = np.maximum(x @ d[f'{n}_W1'].T + d[f'{n}_b1'], 0)
    z = a1 @ d[f'{n}_W2'].T + d[f'{n}_b2']
    e = np.exp(z - z.max()); return e/e.sum()

def show(g):
    ch = ' .:-=+*#%@'
    return '\n'.join(''.join(ch[min(9, int(v*9.99))] for v in row) for row in g)
