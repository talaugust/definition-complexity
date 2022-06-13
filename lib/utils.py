# extra functions for analysis of sci_comm notebooks


import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.formula.api import ols
import re
# from sklearn.metrics.pairwise import cosine_similarity
# from nltk.tokenize import  word_tokenize, sent_tokenize


DOI_PATTERN = r'\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?![\"&\'<>])\S)+)\b'
DOI_REGEX = re.compile(DOI_PATTERN)




# function for extracting different top answer types 
# ascending=False means that taking the top number is the largest (least negative a lot of the time)
def get_top(df, col, ascending=False, n=1):
    return df.sort_values(by=col, ascending=ascending).iloc[0:0+n]

def make_q_doc(q, doc):
    return "question: {} context: {}".format(q, doc)

def make_gpt_doc(q, doc, definition, testing=False):
    if testing:
        return "<context>{}<question>{}<definition>".format(doc, q)
    else:
        return "<context>{}<question>{}<definition>{}".format(doc, q, definition)

# function for constraining each generation to start with 'Term is'
def make_term_start(question):
    q_regex = re.compile('(What is \(are\) )|(Do you have information about )|(\?)')

    term = re.sub(q_regex, '', question)
    term_start = '{} is '.format(term)
    
    return term_start

def make_decoder_inputs(tokenizer, question):

    term_start = make_term_start(question)
    
    decoder_inputs = tokenizer([term_start], max_length=24, return_tensors='pt', add_special_tokens=True)
    decoder_input_ids = decoder_inputs['input_ids'][:1, :-1] # strip of eos token
    
    return decoder_input_ids


# first get rouge, to just make sure these responses are reasonable
def calc_rouge(df, rouge_metric, ref_col, response_col):
    scores = rouge_metric.compute(
        predictions=df[response_col], references=df[ref_col],
        rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
#         rouge_types=['rougeL'],

        use_agregator=True, use_stemmer=False
    )
    rows = []
    measure_mapping = {'F':lambda x: x.mid.fmeasure}

    for t in scores.keys():
        for measure in ['F']:
            rows.append({'score_type':t, 'score_measure':measure, 'score':measure_mapping[measure](scores[t])})
    
    return pd.DataFrame(rows)


# I just use this too much to not have it here
def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


##### Extractive Baseline Utils 
def get_tfidf(query, doc_sentences, vectorizer, n_results=7):

    docs_tfidf = vectorizer.fit_transform(doc_sentences)
    query_tfidf = vectorizer.transform([query])

    cosine_similarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()

    sentence_similarity = list(zip(cosine_similarities, doc_sentences))
    
    return sorted(sentence_similarity, reverse=True, key=lambda tup: tup[0])[:n_results]



def get_sentences(support_doc):
    return flatten_list([sent_tokenize(passage) for passage in support_doc.split('<P>')])


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
        
def combine_chunks(chunks):
    return [' '.join(c) for c in chunks]


def get_and_chunk_sentences(support_doc, n=5):
    
    chunk_list = []
    passages = support_doc.split('<P>')
    
    for p in passages:
        sentences = sent_tokenize(p)
        chunk_list.extend(combine_chunks(chunks(sentences, n)))
        
    return chunk_list





def find_doi_links(links, regex=DOI_REGEX):
    return list(filter(regex.search, links))


# because some people wrote it weird or put it in the tags
def is_int_annotation(annotation):
    text = annotation.text
    tags = annotation.tags # while tags technically is a list, it takes long to evaluate it as such, so just search it like a string
    return ('INT' in text) or ('INT' in tags)


def run_logit(f, df, display_summary=False):
    logitfit = smf.logit(formula = str(f), data=df, missing = 'drop').fit()
    print('---------------------------------------') 
    print(f, 'AIC:', logitfit.aic)
    if display_summary:
        display(logitfit.summary2()) 
    print('---------------------------------------') 
    return logitfit
    
def run_ols(f, df, display_summary=False):
    results = ols(f, data=df, missing = 'drop').fit()
    print('---------------------------------------') 
    print(f, 'AIC:', results.aic, 'Cohen\'s F2:', cohen_f2(results.rsquared_adj))
    if display_summary:
        display(results.summary()) 
    print('---------------------------------------') 
    return results

def run_mixed_effects(f, df, groups, display_summary=False):
    results = smf.mixedlm(f, data=df, missing = 'drop', groups=groups).fit()
    print('---------------------------------------') 
    print(f, 'AIC:', results.aic, 'Cohen\'s F2:', cohen_f2(results.rsquared_adj))
    if display_summary:
        display(results.summary()) 
    print('---------------------------------------') 
    return results

# from: https://www.danielsoper.com/statcalc/calculator.aspx?id=5
def cohen_f2(r_squared):
    return r_squared / (1 - r_squared)