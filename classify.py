import pandas as pd
import numpy as np
import ast
simi_score=pd.read_csv('./experiments/ner_manual_both_sens_similarity_score_sw.csv',sep='\t')
qrels="./qrels/misinfo-qrels.2aspects.useful-credible"

qids = np.unique(simi_score.qid.values)
cred_score = []

simi_score['rank']=simi_score['rank'].astype(int)
for qid in qids:
    qid_simi_score = simi_score.loc[(simi_score['qid'] == qid) & (simi_score['rank']<100)].sort_values(by=['rank'])
    docnos_index=np.unique(qid_simi_score.docno.values,return_index=True)[1]
    docnos=[qid_simi_score.docno.values[index] for index in sorted(docnos_index)]
    docnos=list(set(docnos))
    for docno in docnos:
        docs_specific=qid_simi_score.loc[qid_simi_score['docno']==docno]
        docs_score = []
        journs_index=np.unique(docs_specific.j_docno.values,return_index=True)[1]
        journs=[docs_specific.j_docno.values[j_index] for j_index in sorted(journs_index)]
        for jour in journs[:1]:
            jours_specifics=docs_specific.loc[docs_specific['j_docno']==jour].head(1)
            for ii,rows in jours_specifics.iterrows():
                sentences=ast.literal_eval(rows['scores'])
                #sentences=rows['scores']
                if len(sentences)>0:
                    for sentence in sentences[:1]:
                        docs_score.append(sentence[1])
                else:
                    docs_score.append(0)
        cred_score.append([qid,docno,np.mean(docs_score, axis=0)])


qrels_df=pd.read_csv(qrels,sep=' ',header=None,names=['qid','Q0','docno','top','cred'])
qrels_df=qrels_df[['qid','docno','cred']]
cred_score_df=pd.DataFrame(cred_score,columns=['qid','docno','cred'])
inner=qrels_df.merge(cred_score_df,on=['qid','docno'])

from sklearn.metrics import roc_curve, auc

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


# Add prediction probability to dataframe

# Find optimal probability threshold
threshold = Find_Optimal_Cutoff(inner['cred_x'].values, inner['cred_y'].values)
print(threshold)

# Find prediction to the dataframe applying threshold
inner['cred_y_pred'] = inner['cred_y'].map(lambda x: 1 if x > threshold[0] else 0)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(inner['cred_x'],inner['cred_y_pred']))

from sklearn.metrics import f1_score
print(f1_score(inner['cred_x'], inner['cred_y_pred']))

from sklearn.metrics import fowlkes_mallows_score
print(fowlkes_mallows_score(inner['cred_x'],inner['cred_y_pred']))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(inner['cred_x'],inner['cred_y']))

