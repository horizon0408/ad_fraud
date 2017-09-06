import numpy as np
import pandas as pd
from scipy import spatial
import math
from graph import Graph
from sklearn.metrics import average_precision_score
import time

def count_dict_filter(dic_str):
    d_dic = dic_str
    new_dict = {}
    for key in d_dic:
        if d_dic[key] > 5:
            new_dict[key] = d_dic[key]
          
    return new_dict
def cos_distance(dic_a,dic_b):    
    row = list(set(list(dic_a.keys()) + list(dic_b.keys())))
    
    row_a = []
    row_b = []
    for ifa in row:
        if ifa in dic_a.keys():
            row_a.append(dic_a[ifa])
        else:
            row_a.append(0)
        if ifa in dic_b.keys():
            row_b.append(dic_b[ifa])
        else:
            row_b.append(0)    
    return 1 - spatial.distance.cosine(row_a,row_b)

def build_graph(final_rdd):
    res_cos = []
    k = 0

    for i,item in enumerate(final_rdd):
        publisher_a = item[0]
        dict_a = item[1]
        ifa_list_a = dict_a.keys()

        for j in range(i+1,len(final_rdd)):
#             k += 1
#             if k%50000 == 0:
#                 print (k) 
            publisher_b = final_rdd[j][0]
            dict_b = final_rdd[j][1]
            ifa_list_b = dict_b.keys()

            inter = set(ifa_list_a) & set(ifa_list_b)

            if len(inter) > 5:
                cos_similarity = cos_distance(dict_a, dict_b)
                res_cos.append( (publisher_a, publisher_b, cos_similarity)  )
                
    cos_df = pd.DataFrame([(i[0],i[1],i[2]) for i in res_cos], columns=['publisher_a', 'publisher_b', 'similarity'])
    
    return cos_df

def isflag(ttt):
    last_user = None
    last_url = None
    last_time = None
    last_agent = None
    count = 0
    flag_count = 0
    for c,col in ttt.items():
        user = c[0]
        url = c[1]
        agent = c[2]
        time = c[-1]
        if user == last_user and url == last_url and agent == last_agent and time - last_time <= 2:
            count += col
        else:
            if count >= 5:
                flag_count += 1
            count = col
        last_user, last_url, last_agent,last_time = user, url, agent,time

    return flag_count

def get_flags(df):
    
    tmp = df.groupby(['partnerid']).count()['id']
    bundle_count = tmp.to_dict()
    bundle_count = {i:bundle_count[i] for i in bundle_count if bundle_count[i]>20}
    filtered_bundle_list = list(bundle_count.keys())
    
#    flag_list = {}
    flag_list = []
#     k = 0
    for i in filtered_bundle_list:
#         k+=1
#         if k%50 ==0:
#             print(k)
        tt = df[df['partnerid'] == i]
        tt['timeat'] = tt['timeat'].apply(lambda a :time.mktime(time.strptime(a,'%Y-%m-%d %H:%M:%S.0')) )
        ttt = tt.groupby(['iplong','referer','agent','timeat']).count()['id']
        f = isflag(ttt)
        if f > 0:
#             flag_list[i] = f
            flag_list.append(i)
        
        return flag_list

def BP(graph_df,df):
    V = list(set(graph_df['publisher_a'].tolist()) | set(graph_df['publisher_b'].tolist()))
    S = np.zeros((len(V),len(V)))
    for index, row  in graph_df.iterrows():
        S[V.index(row['publisher_a'])][V.index(row['publisher_b'])] = row['similarity']
        S[V.index(row['publisher_b'])][V.index(row['publisher_a'])] = row['similarity']
    
    sus_p = get_flags(df)
    sus_V_index = [V.index(i) for i in sus_p if i in V]
    
    G = Graph()
    
    for i,pub in enumerate(V):
        nodename = 'pub' + str(i) 
        locals()[nodename] = G.addVarNode(nodename, 2)
        nodepotname = 'P' + str(i)
        if i in sus_V_index:
            locals()[nodepotname] = np.array([[0.7], [0.3]])
        else:
            locals()[nodepotname] = np.array([[0.4], [0.6]])

    for i,pub in enumerate(V):
        nodename = 'pub' + str(i) 
        nodepotname = 'P' + str(i)
        G.addFacNode(eval(nodepotname), eval(nodename))

    for i in range(len(V)):
        nodename_a = 'pub' + str(i) 

        for j in range(i+1, len(V)):
            nodename_b = 'pub' + str(j)
            similarity = S[i][j]
            if similarity != 0:
                #add edge factor
                facname = 'PP' + str(i) +'_'+ str(j)

                #locals()[facname] = similarity * np.array([[0.6, 0.4], [0.4, 0.6]])
                locals()[facname] =  np.array([[similarity*0.8, similarity*0.2], [(1-similarity)*0.4, (1-similarity)*0.6]])
                G.addFacNode(eval(facname), eval('pub'+str(i)), eval('pub'+str(j)))
    marg = G.marginals()
    
    return marg,V

def BP_on_covisitation(df,truth):
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
        
    final_rdd = []
    for i in filtered_bundle_dict:
        final_rdd.append((i,filtered_bundle_dict[i]))
    
    graph_df = build_graph(final_rdd)
    graph_df = graph_df[graph_df['similarity'] > 0.05]
    
    marg,V = BP(graph_df,df)
    
    #evaluate
    index = list(bundle_dict.keys())
    
    y_pred = np.zeros((len(index),2))
    for i in range(len(index)):
        if index[i] in V:
            V_i = V.index(index[i])
            #y_pred[i] = [marg['pub'+str(V_i)][0][0],marg['pub'+str(V_i)][1][0]]
            if marg['pub'+str(V_i)][0][0] > marg['pub'+str(V_i)][1][0]:
                y_pred[i] = [1,0]
            else:
                y_pred[i] = [1,0]
        else:
            y_pred[i] = [0,1]
            
    fraud_list = truth[truth['status'] == 'Fraud']['partnerid'].tolist()
    y_truth = np.zeros((len(index),2))
    for i in range(len(index) ):
        if index[i] in fraud_list:
            y_truth[i] = [1,0]
        else:
            y_truth[i] = [0,1]
    
    ap = average_precision_score(y_truth, y_pred)
    print('Average precision:' + str(ap))