import numpy as np
from scipy import stats
import Base_define_function as bf


def TSgaussian_base(N, T, p, c, r, u, R, alpha, beta):
    t = 0
    q = 1 - p
    ci = np.ones((T + 1, N))
    fi = np.ones((T + 1, N))
    ne = np.ones(T + 1)
    na = np.ones(T + 1)
    u_prim = np.ones((T + 1, N))
    q_prim = np.ones(T + 1)

    uhat = np.ones(N)
    qhat = 1
    sigmau = np.ones(N)
    sigmaq = 1
    ur = np.ones((N, R))
    qr = np.ones(R)

    S_star = bf.Order(r, u, c, q)
    regret = np.zeros(T)
    payoff_optimal = bf.Payoff(r, u, c, q, S_star)
    totalpayoff = np.zeros(T)
    while t < T:
        theta = np.random.normal(0, 1, R)
        for j in range(R):
            for i in range(N):
                uhat[i] = ci[t][i] / (ci[t][i] + fi[t][i])
                sigmau[i] = np.sqrt(alpha * uhat[i] * (1 - uhat[i]) / (ci[t][i] + fi[t][i] + 1)) + np.sqrt(
                    beta / (ci[t][i] + fi[t][i]))
                ur[i][j] = uhat[i] + theta[j] * sigmau[i]

            qhat = ne[t] / (ne[t] + na[t])
            sigmaq = np.sqrt(alpha * qhat * (1 - qhat) / (ne[t] + na[t] + 1)) + np.sqrt(beta / (ne[t] + na[t]))
            qr[j] = qhat + theta[j] * sigmaq

        for i in range(N):
            u_prim[t][i] = np.max(ur[i])
        q_prim[t] = np.max(qr)

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
