import numpy as np

class AiMC():
    def __init__(self, n_rows, n_cols, w_values, cols_per_block, rows_per_block):
        if (n_rows%rows_per_block):
            print("Wrong AiMC initialization. ROWS: {} ROW_BLOCK:{}".format(n_rows,rows_per_block))
            raise ValueError
        elif (n_cols%cols_per_block):
            print("Wrong AiMC initialization. ROWS: {} ROW_BLOCK:{}".format(n_rows,rows_per_block))
            raise ValueError
        else:
            self.row_blocks = n_rows//cols_per_block
            self.cols_block = n_cols//cols_per_block
        self.n_rows     = n_rows
        self.n_cols     = n_cols
        self.w_values   = w_values
        self.cols_per_block = cols_per_block
        self.rows_per_block = rows_per_block

DIANA_AiMC = AiMC(  n_rows=1152,
                    n_cols=512,
                    w_values=list(range(-1,2)),
                    cols_per_block=32,
                    rows_per_block=64)

def tern_to_bin(w):
    if w==1:
        return 2
    elif w==-1:
        return 1
    else:
        return 0

def gen_weights(n_rows, n_cols, tiles=1):
    # import torch here locally to not make it a dependency to this library
    import torch
    return torch.randint(low=-1, high=2, size=(1, n_rows, n_cols)).tolist()

def mirror_rows(w):
    wg = []
    for t in range(len(w)):
        w_t = []
        for r in range( len(w[0])-1, -1, -1):
            w_t.append((w[t][r]))
        wg.append(w_t)
    return wg

def flip_weights(w,toBin=False):
    for t in range(len(w)):
        for r in range(len(w[0])):
            for c in range(len(w[0][0])):
                b = r//64
                if ((c%2==0) and not (b in [6, 7, 8, 9, 10, 11])): #if odd column
                    w[t][r][c]=-w[t][r][c]
                elif ((c%2) and (b<=8)):
                    w[t][r][c]=-w[t][r][c]
                if toBin:
                    w[t][r][c]=tern_to_bin(w[t][r][c])
    return w

def map_weights(w):
    ww = []
    for t in range(len(w)):
        w_t = []
        for r in range(len(w[0])):
            w_p = []
            w_m = []
            for c in range(len(w[0][0])):
                if w[t][r][c]==1:
                    w_p.append(1)
                    w_m.append(0)
                elif w[t][r][c]==-1:
                    w_p.append(0)
                    w_m.append(1)
                else:
                    w_p.append(0)
                    w_m.append(0)
            w_t.append(w_p)
            w_t.append(w_m)
        ww.append(w_t)
    return ww

def flatten_list(l):
    if not (type(l[0]) == list):
        return l
    else:
        flat_l = []
        for e in l:
            for i in e:
                flat_l.append(i)
        return flatten_list(flat_l)

# def _padd_C(w):
#     K  = len(w[0][0])
#     C  = int(len(w[0])/9)
#     FX = 3
#     FY = 3
#     #padding along C in case less than 1152 rows are copied. We want always 1152 rows copied inside the analog accelerator
#     wt = []
#     if (C<128):
#         wg=[]
#         print("Need for padding along C detected...")
#         for i in range(FX*FY):
#             for c in range(128):
#                 wk = []
#                 for k in range(K):
#                     if (c<C):
#                         wk.append(w[0][i*C+c][k])
#                     else:
#                         wk.append(0)
#                 wg.append(wk)
#         wt.append(wg)
#     else:
#         wt = w
#     return wt

def _padd_C(w):
    K  = len(w[0][0])
    C  = int(len(w[0]))
    #padding along C in case less than 1152 rows are copied. We want always 1152 rows copied inside the analog accelerator
    wt = []
    if (C%288!=0):
        wg=[]
        print("Need for padding along C detected...")
        for c in range(int((C+287)/288)*288):
            wk = []
            for k in range(K):
                if (c<C):
                    wk.append(w[0][c][k])
                else:
                    wk.append(0)
            wg.append(wk)
        wt.append(wg)
    else:
        wt = w
    return wt

# def _padd_K(w):
#     K  = len(w[0][0])
#     C  = int(len(w[0])/9)
#     FX = 3
#     FY = 3
#     wt = []
#     if (K<512):
#         wg=[]
#         print("Need for padding along K detected...")
#         for c in range(C*FX*FY):    #total amount of lines
#             wk = []
#             for i in range(512):     #single line parallelism on 16 * 16 bits. Multiple of 512 weights are copied.
#                 if (i<K):
#                     wk.append(w[0][c][i])
#                 else:
#                     wk.append(0)
#             wg.append(wk)
#         wt.append(wg)
#     else:
#         wt = w
#     return wt

def _padd_K(w):
    K  = len(w[0][0])
    C  = int(len(w[0]))
    wt = []
    if (K%128!=0):
        wg=[]
        print("Need for padding along K detected...")
        for c in range(C):    #total amount of lines
            wk = []
            for i in range(int((K+127)/128)*128):     #single line parallelism on 16 * 16 bits. Multiple of 512 weights are copied.
                if (i<K):
                    wk.append(w[0][c][i])
                else:
                    wk.append(0)
            wg.append(wk)
        wt.append(wg)
    else:
        wt = w
    return wt

def pad(w, C=True, K=True):
    if C:
        w = _padd_C(w)
    if K:
        w = _padd_K(w)
    return w

if __name__ == '__main__':
    n_rows = DIANA_AiMC.n_rows 
    n_cols = DIANA_AiMC.n_cols
    w = gen_weights(400, 100)
    w_list = pad(w, True, True)
    w_list = mirror_rows(w_list)
    w_list = flip_weights(w_list, False)
    w_list = map_weights(w_list)
    w_list = flatten_list(w_list)