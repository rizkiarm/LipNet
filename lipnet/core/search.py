import numpy as np

# Source: https://github.com/bshillingford/ctc-beam-search

def joint(char, y, input_dist, alphabet, B, Pr, pb_):
    if y != '' and char == y[-1]:
        out = input_dist[alphabet.index(char)] + pb_[B.index(y)]
    else:
        out = input_dist[alphabet.index(char)] + Pr[B.index(y)]
    return out

def ctc_beamsearch(input_dist, alphabet='abcdefghijklmnopqrstuvwxyz -', k=200):
    # input_dist should be numpy matrix
    # beamsize is k
    T = input_dist.shape[0]
    B = ['']
    Pr = [0]
    pnb_ = [-1e10]
    pb_ = [0]
    for t in range(T):
        # get k most probable sequences in B
        # print 't is ' + str(t)
        B_new = []
        Pr_new = []
        pnb_new = []
        pb_new = []
        ind = np.argsort(Pr)[::-1]
        B_ = [B[i] for i in ind[:k]]
        for y in B_:
            # print 'y is ' + y
            if y != '':
                pnb = pnb_[B.index(y)] + input_dist[t, alphabet.index(y[-1])]
                if y[:-1] in B_:
                    pnb = np.logaddexp(pnb, joint(y[-1], y[:-1], input_dist[t], alphabet, B, Pr, pb_))
            else:
                pnb = -1e10

            pb = Pr[B.index(y)] + input_dist[t, alphabet.index('-')]

            B_new += [y]
            Pr_new += [np.logaddexp(pnb, pb)]
            pnb_new += [pnb]
            pb_new += [pb]
            for c in alphabet[1:]:
                # print 'c is ' + c
                pb = -1e10
                pnb = joint(c, y, input_dist[t], alphabet, B, Pr, pb_)
                pb_new += [pb]
                pnb_new += [pnb]
                Pr_new += [np.logaddexp(pnb, pb)]
                B_new += [y + c]
        B = B_new;
        Pr = Pr_new;
        pnb_ = pnb_new;
        pb_ = pb_new;

    # out_ind = np.argmax([Pr[i]/len(B[i]) if len(B[i]) > 0 else -1e10 for i in range(len(B))])
    out_ind = np.argmax([Pr[i] for i in range(len(B))])
    return B[out_ind]