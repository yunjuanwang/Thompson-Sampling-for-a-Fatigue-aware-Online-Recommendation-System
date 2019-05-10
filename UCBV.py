import numpy as np
from scipy import stats
import Base_define_function as bf


def ucbv(N, T, p, c, r, u, b):
    t = 1
    q = 1-p
    ne = np.ones(T+1)
    na = np.ones(T+1)
    ci = np.ones((T+1, N))
    fi = np.ones((T+1, N))
    u_ucb = np.ones((T + 1, N))
    q_ucb = np.ones(T + 1)
    S_star = bf.Order(r, u, c, q)
    assert len(S_star) > 0
    regret = np.zeros(T)
    payoff_optimal = bf.Payoff(r, u, c, q, S_star)
    totalpayoff = np.zeros(T)

    while t < T:
        S = bf.Order(r, u_ucb[t - 1], c, q_ucb[t - 1])
        if len(S) == 0:
            continue

        for i in range(N):
            ci[t][i] = ci[t - 1][i]
            fi[t][i] = fi[t - 1][i]
            u_ucb[t][i] = u_ucb[t - 1][i]
        ne[t] = ne[t - 1]
        na[t] = na[t - 1]

        fb, q_flag = bf.Feedback(u, q, S)  # fb: user feedback,0 not select, 1 select; q_flag: whether abandon or not
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
        for i in range(kt):
            u_ucb[t][S[i]] = ci[t][S[i]] / (ci[t][S[i]] + fi[t][S[i]]) + np.sqrt(
                2 * np.var(u_ucb[t-1]) * np.log(t) / (ci[t][S[i]] + fi[t][S[i]])) + np.log(t) / (ci[t][S[i]] + fi[t][S[i]])
            if u_ucb[t][S[i]] > 1:
                u_ucb[t][S[i]] = 1
        q_ucb[t] = ne[t] / (ne[t] + na[t]) + np.sqrt(2 * np.log(t) / (ne[t] + na[t]))
        if q_ucb[t] > 1:
            q_ucb[t] = 1
        payoff_real = bf.Payoff(r, u, c, q, S)
        regret[t] = payoff_optimal - payoff_real
        totalpayoff[t] = totalpayoff[t - 1] + regret[t]
        t = t + 1
    return totalpayoff
