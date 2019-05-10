import numpy as np
from scipy import stats
import Base_define_function as bf


def thompson_base (N, T, p, c, r, u):
    t=0
    q=1-p
    ne=np.ones(T+1)
    na=np.ones(T+1)
    ci=np.ones((T+1,N))
    fi=np.ones((T+1,N))
    u_prim = np.ones((T + 1, N))
    q_prim = np.ones(T + 1)
    S_star = bf.Order(r, u, c, q)
    regret = np.zeros(T)
    payoff_optimal = bf.Payoff(r, u, c, q, S_star)
    totalpayoff = np.zeros(T)
     while t < T:
        for i in range(N):
            u_prim[t][i] = stats.beta(ci[t][i], fi[t][i]).rvs(1)
        q_prim[t] = stats.beta(ne[t], na[t]).rvs(1)
        S = bf.Order(r, u_prim[t], c, q_prim[t])
        fb, q_flag = bf.Feedback(u, q, S)  # fb: user feedback,0 not select, 1 select; q_flag: whether abandon or not
        if len(S) == 0:
            continue

        kt = len(fb)
        if kt > 1:
            for i in range(kt - 1):
                fi[t][S[i]] = fi[t][S[i]] + 1
                ne[t] = ne[t] + 1
        if fb[kt - 1] == 1:
            ci[t][S[kt - 1]] = ci[t][S[kt - 1]] + 1
        else:
            fi[t][S[kt - 1]] = fi[t][S[kt - 1]] + 1
            if q_flag == 0:
                ne[t] = ne[t] + 1
            else:
                na[t] = na[t] + 1
        for i in range(N):
            ci[t + 1][i] = ci[t][i]
            fi[t + 1][i] = fi[t][i]
        ne[t + 1] = ne[t]
        na[t + 1] = na[t]
        payoff_real = bf.Payoff(r, u, c, q, S)
        regret[t] = payoff_optimal - payoff_real
        totalpayoff[t] = totalpayoff[t - 1] + regret[t]
        t = t + 1
    return totalpayoff
