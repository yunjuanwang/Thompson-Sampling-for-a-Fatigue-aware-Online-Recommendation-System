import numpy as np
import pickle
import matplotlib.pyplot as plt


T = 100000
plt.rcParams['font.family'] 	= 'serif'
plt.rcParams['font.size'] 		= 22
plt.rcParams['axes.labelsize'] 	= 22
plt.rcParams['figure.figsize'] = (9, 6)


for i in range(1, 11):
    f = open('result/data_0.5_'+str(i)+'.txt','rb')
    totalpayoff = pickle.load(f)
    f.close()
    t=np.linspace(0,T-1,T)
    plt.plot(t,totalpayoff)


plt.xlabel('T'); plt.ylabel('Regret');
plt.ylim((0, 400))
# plt.figure(figsize=(8,4))
plt.savefig("chart/algo0_5.eps")
plt.show()

