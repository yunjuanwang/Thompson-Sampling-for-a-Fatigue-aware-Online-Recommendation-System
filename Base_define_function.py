import numpy as np
from scipy import stats


def Order(reward, u_select, cost, q_not_abandon):# find the optimal order of sequential message
    num=len(reward)
    theta=list()
    p_abandon=1-q_not_abandon
    for i in range(num):        th=(reward[i]*u_select[i]-cost*p_abandon*(1-u_select[i]))/(1-q_not_abandon*(1-u_select[i]))
        theta.append(th)
    theta_order_indices = sorted(range(len(theta)), key=lambda k: theta[k],reverse=1)
    message_index_order = []
    for i in range(num):
        if theta[theta_order_indices[i]] > 0:
            message_index_order.append(theta_order_indices[i])
        else:
            break
    return message_index_order


def P_i_S(u_select, q_not_abandon, message_index_order, index):
    if index==0:
        return u_select[message_index_order[index]]
    else:
        u_prod=1
        for i in range(index):
            u_prod=u_prod*(1-u_select[message_index_order[i]])
        p_i_S=np.power(q_not_abandon,index)*u_prod*u_select[message_index_order[index]]
        return p_i_S


def Feedback(u_select, q_not_abandon, message_index_order):
    user_choice=list()
    user_abandon=0  # 0 means not abandon, 1 means abandon
    for i in range(len(message_index_order)):
        select=stats.bernoulli(u_select[message_index_order[i]]).rvs(1)
        if select==1:
            user_choice.append(1)
            return user_choice,user_abandon
        else:
            user_choice.append(0)
            user_abandon=stats.bernoulli(1-q_not_abandon).rvs(1)
            if user_abandon==1:
                return user_choice,user_abandon
    return user_choice,user_abandon


def P_a_S(u_select, q_not_abandon, sequential_message_index):
    m=len(sequential_message_index)
    summ=0
    for i in range(m):
        u_prod=1
        for j in range(i+1):
            u_prod=u_prod*(1-u_select[sequential_message_index[j]])
        summ=summ+np.power(q_not_abandon,i)*(1-q_not_abandon)*u_prod
    return summ


def Payoff(reward, u_select, cost,q_not_abandon,sequential_message_index):
    m=len(sequential_message_index)
    if m==0:
        return 0
    summ=0
    for i in range(m): summ=summ+P_i_S(u_select,q_not_abandon,sequential_message_index,i)*reward[sequential_message_index[i]]
    payoff=summ-cost*P_a_S(u_select,q_not_abandon,sequential_message_index)
    return payoff
