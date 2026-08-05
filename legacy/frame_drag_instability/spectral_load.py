import numpy as np

def load(fn):
    d = {}
    with open(fn) as f:
        first = f.readline().split()
        meta = dict(zip(
            ["L", "N_r", "r_max", "n_he", "n_ve", "n_tri", "n_rect",
             "e_split", "b_split"],
            [float(x) if "." in x else int(x) for x in first[1:]]))
        while True:
            hdr = f.readline().split()
            if not hdr:
                break
            name, n = hdr[0], int(hdr[1])
            a = np.fromfile(f, count=n, sep="\n")
            d[name] = a
    return meta, d
