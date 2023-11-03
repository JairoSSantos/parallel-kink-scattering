import numpy as np

def coefficients(M, N, alpha, sym: bool=True):
    assert len(alpha) == N + 1
    delta = np.full((M + 1, N + 1, N + 1), sp.Rational(0) if sym else 0)
    one = sp.Rational(1) if sym else 1
    delta[0, 0, 0] = one
    c1 = one
    for n in range(1, N+1):
        c2 = one
        for nu in range(n):
            c3 = alpha[n] - alpha[nu]
            c2 *= c3
            for m in range(min(n, M) + 1):
                delta[m, n, nu] = (alpha[n]*delta[m, n-1, nu] - m*delta[m-1, n-1, nu])/c3
        for m in range(min(n, M) + 1):
            delta[m, n, n] = (c1/c2)*(m*delta[m-1, n-1, n-1] - alpha[n-1]*delta[m, n-1, n-1])
        c1 = c2
    return delta[M, N]
