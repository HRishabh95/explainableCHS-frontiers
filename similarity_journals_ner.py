import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('GPL/trec-covid-v2-msmarco-distilbert-gpl')
import numpy as np
from transformers import pipeline
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForTokenClassification
ner_model='d4data/biomedical-ner-all'
tokenizer = AutoTokenizer.from_pretrained(ner_model)
model_ner = AutoModelForTokenClassification.from_pretrained(ner_model)
pipe = pipeline("ner", model=model_ner, tokenizer=tokenizer, aggregation_strategy="simple") # pass device=0 if using gpu

import errno
import os

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        # possibly handle other errno cases here, otherwise finally:
        else:
            raise

import re
import string
PUNCTUATIONS = string.punctuation.replace('.','')


def remove_punctuation(text):
  trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
  return text.translate(trans)

def remove_whitespaces(text):
    return " ".join(text.split())

def clean_en_text(text):
  """
  text
  """
  text = re.sub(r"[^A-Za-z0-9(),.!?\'`]", " ", text)
  #text= re.sub(r'\d+', '',text)
  text = remove_punctuation(text)
  text = remove_whitespaces(text)
  return text.strip().lower()

def get_entity_name(entities):
    if len(entities)>0:
        return_word=[]
        for entity in entities:
            if entity['entity_group']=='Medication':
                if entity['word'] not in return_word:
                    return_word.append(entity['word'])
        if len(return_word)>0:

            return return_word[0]
        else:
            return None
    else:
        return None

def get_medication_query(texts):
    texts=texts.replace('covid 19','')
    texts=" ".join(texts.split(" ")[:-2])
    return texts



def get_score_n(journal_dfs,docs_dfs,root_path,top_n=10,d_top=100):
    qids = np.unique(docs_dfs.qid.values)
    similarity = []
    for qid in qids:
        print(qid)
        journal_tops = journal_dfs.loc[journal_dfs['qid'] == qid].sort_values(by=['rank']).head(top_n)
        docs_tops = docs_dfs.loc[docs_dfs['qid'] == qid].sort_values(by=['rank'])
        for ii, doc_rows in docs_tops.iterrows():
            doc_sens = [(i.split('\t')[0],float(i.split('\t')[1])) for i in doc_rows['top_sens'].split(",") if len(i)>0]
            doc_sens_sorted=sorted(doc_sens, key=lambda t: t[1], reverse=True)
            for doc_sen in doc_sens_sorted:
                sens_evi=''
                if doc_sen[-1]>0.4:
                    docs_sen_entity_name = get_entity_name(pipe(doc_sen[0]))
                    doc_sen_vec = model.encode(doc_sen[0])
                    for jj, journal_rows in journal_tops.iterrows():
                        texts = journal_rows['text'].split('.')
                        for sen in texts:
                            if len(sen.split(" "))>5:
                                jou_sen_vec=model.encode(sen)
                                simi = cosine_similarity([doc_sen_vec, jou_sen_vec])[0][1]
                                journal_sen_entity_name = get_entity_name(pipe(sen))
                                if docs_sen_entity_name and journal_sen_entity_name:
                                    if docs_sen_entity_name == journal_sen_entity_name:
                                        sens_evi += '%s\t %s\t %s,'%(doc_sen[0],sen,simi)
                                    elif SequenceMatcher(None, docs_sen_entity_name,
                                                         journal_sen_entity_name).ratio() > 0.9:
                                        sens_evi += '%s\t %s\t %s,'%(doc_sen[0],sen,float(simi)*0.3)
                                    else:
                                        sens_evi += '%s\t %s\t %s,'%(doc_sen[0],sen,float(simi)*0.15)
                                else:
                                    sens_evi += '%s\t %s\t %s,' % (doc_sen[0], sen, float(simi) * 0.05)
                        journal_evi_sorted = sorted([('%s\t %s' % (i.split('\t')[0], i.split('\t')[1]), float(i.split('\t')[-1])) for i
                                        in sens_evi.split(",") if len(i) > 0], key=lambda t:t[1],reverse=True)[:10]

                        similarity.append([qid, doc_rows['docno'], journal_rows['docno'],journal_evi_sorted,doc_rows['rank']])

    similarity_df = pd.DataFrame(similarity, columns=['qid', 'docno', 'j_docno', 'scores','rank'])
    similarity_path=f'''{root_path}experiments'''
    mkdir_p(similarity_path)
    similarity_df.to_csv('%s/ner_manual_both_sens_similarity_score_sw.csv'%similarity_path, index=None, sep='\t')
    return similarity_df

#load dfs
root_path='./'
docs_path="%sdocs/docs_all_top_sen_ner_manual.csv"%root_path
journal_path="%sdocs/journal_wnum_top_30.csv"%root_path
journal_dfs=pd.read_csv(journal_path,sep='\t')
docs_dfs=pd.read_csv(docs_path,sep=';')

docs_dfs.dropna(inplace=True)
simi_score=get_score_n(journal_dfs,docs_dfs,root_path)
