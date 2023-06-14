import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('GPL/trec-covid-v2-msmarco-distilbert-gpl')
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

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

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

def get_vectors(dfs):
    vecs = []
    sens_all=[]
    for ii, rows in dfs.iterrows():
        print(ii)
        chuck_vecs = []
        texts = clean_en_text(rows['text'])
        simis=[]
        sens=''
        if texts:
            texts=split_into_sentences(texts)
            for i in range(0, len(texts)):
                c_text=clean_en_text(texts[i])
                sen_embeddings = model.encode(c_text)
                query_embeddings = model.encode(rows['query'])
                query_sen_entity_name = get_entity_name(pipe(rows['query']))
                arti_sen_entity_name = get_entity_name(pipe(c_text))
                simi=cosine_similarity([query_embeddings,sen_embeddings])[0][1]
                if query_sen_entity_name and arti_sen_entity_name:
                    if query_sen_entity_name==arti_sen_entity_name:
                        sens+=texts[i]+"\t %s,"%(float(simi))
                        simi=simi
                    elif SequenceMatcher(None, query_sen_entity_name, arti_sen_entity_name).ratio()>0.9:
                        sens += texts[i] + "\t %s," % (float(simi)*SequenceMatcher(None, query_sen_entity_name, arti_sen_entity_name).ratio()*0.3)
                        simi=simi*0.1
                    else:
                        sens += texts[i] + "\t %s," % (float(simi) * 0.01)
                        simi=simi*0.1
                else:
                    query_sen_entity_name=get_medication_query(rows['query'])
                    if query_sen_entity_name in texts[i]:
                        sens+=texts[i]+"\t %s,"%(float(simi))
                        simi=simi
                    else:
                        sens += texts[i] + "\t %s," % (float(simi) * 0.1)
                        simi=simi*0.1
                simis.append([texts[i],simi])
                chuck_vecs.append(sen_embeddings*simi)
            sens_all.append([rows['qid'],rows['docno'],sens])
    sens=pd.DataFrame(sens_all,columns=['qid','docno','top_sens'])
    dfs_final=pd.merge(dfs,sens,on=['qid','docno'])
    return dfs_final



docs_dfs=pd.read_csv("./docs/gen_docs_top_100.csv",sep='\t')
docs_dfs_vec=get_vectors(docs_dfs)
docs_dfs_vec.to_csv('./docs/gen_docs_func_all_top_sen_ner_manual_covid_bert.csv', index=None, sep=';')