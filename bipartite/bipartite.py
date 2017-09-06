import numpy as np
import pandas as pd
import math
from sklearn.metrics import average_precision_score

def count_dict_filter(dic_str):
    d_dic = dic_str
    new_dict = {}
    for key in d_dic:
        if d_dic[key] > 5:
            new_dict[key] = d_dic[key]
          
    return new_dict

def FindSusUsers(U,c,sus_P,P,D,s,p):
    # U and sus_P are set, c is a dictionary
    sus_U = []
    alpha = {}

   
    for j in sus_P:
        for i in P[j]:
            if (abs(c[j] - D[i][j])/c[j] <= s):
                if i in alpha:
                    alpha[i] += 1
                else:
                    alpha[i] = 1
    sum_alpha = 0
    for i in alpha:
        if alpha[i] >= p * len(sus_P): 
            sum_alpha += alpha[i]
            sus_U.append(i)
            
    return (sus_U, sum_alpha)

def InitClusterCentre(u_dict,s):
    selected_ifa = []
    cluster_list = []
    current_cluster = []
    while len(selected_ifa) != len(u_dict):
        unselected_ifa = [u for u in u_dict if u not in selected_ifa]
        if len(current_cluster) == 0:
            current_cluster.append(u_dict[unselected_ifa[0]])
        else:
            for user in u_dict:
                avg_tmp = (sum(current_cluster) + u_dict[user]) / (len(current_cluster) + 1)
                if user not in selected_ifa:
                    if (abs(u_dict[user] - avg_tmp)/ avg_tmp <= s): #or (abs(u_dict[user] - avg_tmp)<=10):
                        current_cluster.append(u_dict[user])
                        selected_ifa.append(user)

            #print(current_cluster)
            if len(current_cluster) > 3:
                cluster_list.append(np.mean(current_cluster))
                current_cluster = []
            else:
                current_cluster = []
    return cluster_list

c_test = {}
p_test = []

def Init(P_init,s):
    c = {}
    P = []
#    k = 0
    for j in P_init:
#        k+=1
#        if k%50 ==0:
#            print(k) 
        j_cluster = InitClusterCentre(P_init[j],s)
        
        if len(j_cluster) > 0:
            P.append(j)
            c[j] = j_cluster

            p_test.append(j)
            c_test[j] = j_cluster
    return (c,P)   

def co_clustering(n,m,s,p,df,truth):
    
    ifa_dict = {}
    tmp = df.groupby(['iplong','partnerid']).count()['id']
    for c,col in tmp.iteritems():
        user = c[0]
        publisher = c[1]
        if user not in ifa_dict:
            ifa_dict[user] = {publisher: col}
        else:
            if publisher not in ifa_dict[user]:
                ifa_dict[user][publisher] = col
            else:
                ifa_dict[user][publisher] += col    
    filtered_ifa_dict = {}
    for i in ifa_dict:
        bundle_dic_tmp = count_dict_filter(ifa_dict[i])
        if len(bundle_dic_tmp) != 0:
            filtered_ifa_dict[i] = bundle_dic_tmp
            
    bundle_dict = {}
    bundle_test = df.groupby(['partnerid','iplong']).count()['id']
    for c,col in bundle_test.iteritems():   
        publisher = c[0]
        user = c[1]
        if publisher not in bundle_dict:
            bundle_dict[publisher] = {user: col}
        else:
            if user not in bundle_dict[publisher]:
                bundle_dict[publisher][user] = col
            else:
                bundle_dict[publisher][user] += col
        filtered_bundle_dict = {}
    for i in bundle_dict:
        ifa_dic_tmp = count_dict_filter(bundle_dict[i])
        if len(ifa_dic_tmp) != 0:
            filtered_bundle_dict[i] = ifa_dic_tmp
            
    P_init = filtered_bundle_dict
    U = filtered_ifa_dict
    D = filtered_ifa_dict

    c_init,sus_p_init = Init(P_init,s)
    
    cluster_test = []
    selected_seed = []
    
    for seed_pub in sus_p_init:
        selected_seed.append(seed_pub)
        #print(seed_pub)
        #count +=1
        #if count % 50 == 0:
        #    print(count)
        for c in c_init[seed_pub]:
            c_tmp = {seed_pub:c}
            sus_P = [seed_pub]
            (sus_u_tmp,a) = FindSusUsers(U,c_tmp,sus_P,P_init,D,s,p)
            pub_to_add = []
            for u in sus_u_tmp:
                pub_to_add += filtered_ifa_dict[u].keys()

            pub_to_add =  [i for i in (set(pub_to_add) - set(selected_seed)) if (pub_to_add.count(i) > p*len(sus_u_tmp)) and (i in sus_p_init)]

            for j in pub_to_add:   

                sus_P_tmp = sus_P + [j]
                #updata_c: choose selectes sus users to estimate the c for new added pub
                c_collect = [filtered_ifa_dict[i][j] for i in sus_u_tmp if j in filtered_ifa_dict[i].keys()]
                c_tmp_tmp = c_tmp.copy()
                c_tmp_tmp[j] = np.mean(c_collect)
                #print(c_tmp_tmp)
                #print(sus_P_tmp)
                (sus_u_tmp_tmp,a_tmp) = FindSusUsers(U,c_tmp_tmp,sus_P_tmp,P_init,D,s,p)

                if a_tmp >= a or (len(sus_P) == 1 and a_tmp > n):
                    sus_P = sus_P_tmp
                    a = a_tmp
                    sus_u_tmp = sus_u_tmp_tmp
                    c_tmp = c_tmp_tmp

            if len(sus_P) >= m:
                #selected_seed += sus_P
                cluster_test.append((sus_u_tmp, sus_P))
                
    merged_sus_cluster = {}
    all_sus_pub = []
    for (u,pub) in cluster_test:
        all_sus_pub += pub
        pub = tuple(pub)
        if pub in merged_sus_cluster:
            merged_sus_cluster[pub] = list(set(merged_sus_cluster[pub] + u))
        else:
            merged_sus_cluster[pub] = u
    all_sus_pub_count = { i:all_sus_pub.count(i) for i in set(all_sus_pub)}
                
    #evaluation
    fraud_df = truth[truth['status'] == 'Fraud']
    fraud_list = fraud_df['partnerid'].tolist()
    
    index = list(bundle_dict.keys())
    
    y_truth = np.zeros((len(index),2))
    for i in range(len(index)):
        if index[i] in fraud_list:
            y_truth[i] = [1,0]
        else:
            y_truth[i] = [0,1]

    y_pred = np.zeros((len(index),2))
    for i in range(len(index)):
        if index[i] in all_sus_pub_count.keys():
            y_pred[i] = [1,0]
        else:
            y_pred[i] = [0,1]
            
    ap = average_precision_score(y_truth, y_pred) 
    print('Average precision:' + str(ap))