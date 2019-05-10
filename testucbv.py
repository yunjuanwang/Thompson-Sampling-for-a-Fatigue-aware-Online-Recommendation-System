import numpy as np
import pickle
from Thompsonbasedpy import thompson_base
from ucbbasedpy import ucb_base
from TSgaussian import TSgaussian_base
from UCBV import ucbv
import matplotlib.pyplot as plt
import os
import multiprocessing

N = 30
T = 100000
p = 0.1
c = 0.5
# R = 20
# alpha = 1
# beta = 2
# r = np.random.uniform(0, 1, N)   # reward
# u = np.random.uniform(0, 1, N)   # real probability of messages being selected


# [0,0.1]
r=[0.93063581, 0.59056804, 0.20004313, 0.38358707, 0.62820822, 0.61043812,
   0.17145896, 0.41702337, 0.29217008, 0.27526696, 0.73458688, 0.59549223,
   0.82045492, 0.34658352, 0.83268829, 0.691289,   0.26994209, 0.68752154,
   0.44007455, 0.40297208, 0.32021075, 0.7014262,  0.33391845, 0.81181029,
   0.22438217, 0.43780068, 0.56431515, 0.7487486,  0.50357076, 0.07523947]

u=[0.04312021, 0.07882069, 0.00092667, 0.04843211, 0.0546653,  0.04166339,
 0.08620413, 0.00709681, 0.08568388, 0.07943192, 0.08284139, 0.00838911,
 0.01219895, 0.08100727, 0.02344025, 0.01677291, 0.08026796, 0.07774915,
 0.02655784, 0.00624131, 0.0697308,  0.01291121, 0.03568038, 0.07548042,
 0.09600775, 0.02421222, 0.08759905, 0.02293794, 0.05272037, 0.07922301]

b = 0.1


def ucbvtest_1(N, T, p, c, r, u, b):
    payoff = ucbv(N, T, p, c, r, u, b)
    f = open('result/ucbvdata_0.1_1.txt', 'wb')
    pickle.dump(payoff, f)
    f.close()


def ucbvtest_2(N, T, p, c, r, u, b):
    payoff = ucbv(N, T, p, c, r, u, b)
    f = open('result/ucbvdata_0.1_2.txt', 'wb')
    pickle.dump(payoff, f)
    f.close()


def ucbvtest_3(N, T, p, c, r, u, b):
    payoff = ucbv(N, T, p, c, r, u, b)
    f = open('result/ucbvdata_0.1_3.txt', 'wb')
    pickle.dump(payoff, f)
    f.close()


def ucbvtest_4(N, T, p, c, r, u, b):
    payoff = ucbv(N, T, p, c, r, u, b)
    f = open('result/ucbvdata_0.1_4.txt', 'wb')
    pickle.dump(payoff, f)
    f.close()


def ucbvtest_5(N, T, p, c, r, u, b):
    payoff = ucbv(N, T, p, c, r, u, b)
    f = open('result/ucbvdata_0.1_5.txt', 'wb')
    pickle.dump(payoff, f)
    f.close()


def ucbvtest_6(N, T, p, c, r, u, b):
    payoff = ucbv(N, T, p, c, r, u, b)
    f = open('result/ucbvdata_0.1_6.txt', 'wb')
    pickle.dump(payoff, f)
    f.close()


def ucbvtest_7(N, T, p, c, r, u, b):
    payoff = ucbv(N, T, p, c, r, u, b)
    f = open('result/ucbvdata_0.1_7.txt', 'wb')
    pickle.dump(payoff, f)
    f.close()


def ucbvtest_8(N, T, p, c, r, u, b):
    payoff = ucbv(N, T, p, c, r, u, b)
    f = open('result/ucbvdata_0.1_8.txt', 'wb')
    pickle.dump(payoff, f)
    f.close()


def ucbvtest_9(N, T, p, c, r, u, b):
    payoff = ucbv(N, T, p, c, r, u, b)
    f = open('result/ucbvdata_0.1_9.txt', 'wb')
    pickle.dump(payoff, f)
    f.close()


def ucbvtest_10(N, T, p, c, r, u, b):
    payoff = ucbv(N, T, p, c, r, u, b)
    f = open('result/ucbvdata_0.1_10.txt', 'wb')
    pickle.dump(payoff, f)
    f.close()


# ucbvtest_1(N, T, p, c, r, u, b)
# ucbvtest_2(N, T, p, c, r, u, b)
# ucbvtest_3(N, T, p, c, r, u, b)
# ucbvtest_4(N, T, p, c, r, u, b)
# ucbvtest_5(N, T, p, c, r, u, b)
# ucbvtest_6(N, T, p, c, r, u, b)
# ucbvtest_7(N, T, p, c, r, u, b)
# ucbvtest_8(N, T, p, c, r, u, b)
# ucbvtest_9(N, T, p, c, r, u, b)
# ucbvtest_10(N, T, p, c, r, u, b)

plt.rcParams['font.family'] 	= 'serif'
plt.rcParams['font.size'] 		= 22
plt.rcParams['axes.labelsize'] 	= 22
plt.rcParams['figure.figsize'] = (9, 6)

f = open('result/ucbdata_1_1.txt','rb')
totalpayoff1 = pickle.load(f)
f.close()
t=np.linspace(0,T-1,T)
plt.plot(t,totalpayoff1)

f = open('result/ucbdata_1_2.txt','rb')
totalpayoff2 = pickle.load(f)
f.close()
t=np.linspace(0,T-1,T)
plt.plot(t,totalpayoff2)

f = open('result/ucbdata_1_3.txt','rb')
totalpayoff3 = pickle.load(f)
f.close()
t=np.linspace(0,T-1,T)
plt.plot(t,totalpayoff3)

f = open('result/ucbdata_1_4.txt','rb')
totalpayoff4 = pickle.load(f)
f.close()
t=np.linspace(0,T-1,T)
plt.plot(t,totalpayoff4)

f = open('result/ucbdata_1_5.txt','rb')
totalpayoff5 = pickle.load(f)
f.close()
t=np.linspace(0,T-1,T)
plt.plot(t,totalpayoff5)

f = open('result/ucbdata_1_6.txt','rb')
totalpayoff6 = pickle.load(f)
f.close()
t=np.linspace(0,T-1,T)
plt.plot(t,totalpayoff6)

f = open('result/ucbdata_1_7.txt','rb')
totalpayoff7 = pickle.load(f)
f.close()
t=np.linspace(0,T-1,T)
plt.plot(t,totalpayoff7)

f = open('result/ucbdata_1_8.txt','rb')
totalpayoff8 = pickle.load(f)
f.close()
t=np.linspace(0,T-1,T)
plt.plot(t,totalpayoff8)

f = open('result/ucbdata_1_9.txt','rb')
totalpayoff9 = pickle.load(f)
f.close()
t=np.linspace(0,T-1,T)
plt.plot(t,totalpayoff9)

f = open('result/ucbdata_1_10.txt','rb')
totalpayoff10 = pickle.load(f)
f.close()
t=np.linspace(0,T-1,T)
plt.plot(t,totalpayoff10)

plt.xlabel('T'); plt.ylabel('Regret');
plt.ylim((0, 400))
plt.savefig("compucb.eps")
plt.show()