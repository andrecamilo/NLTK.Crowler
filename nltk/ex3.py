# Introdução ao Processamento de Linguagem Natural (PLN) Usando Python
# Professor Fernando Vieira da Silva MSc.
# Problema de Classificação
# Neste tutorial vamos trabalhar com um exemplo prático de problema de classificação de texto. O objetivo é identificar uma sentença como escrita "formal" ou "informal".
# 1. Obtendo o corpus
# Para simplificar o problema, vamos continuar utilizando o corpus Gutenberg como textos formais e vamos usar mensagens de chat do corpus nps_chat como textos informais.
# Antes de tudo, vamos baixar o corpus nps_chat:

# In [1]:
import nltk
nltk.download('nps_chat')
[nltk_data] Downloading package nps_chat to
[nltk_data]     /home/datascience/nltk_data...
[nltk_data]   Unzipping corpora/nps_chat.zip.
# Out[1]:
# True

# In [6]:
from nltk.corpus import nps_chat
print(nps_chat.fileids())
# ['10-19-20s_706posts.xml', '10-19-30s_705posts.xml', '10-19-40s_686posts.xml', '10-19-adults_706posts.xml', '10-24-40s_706posts.xml', '10-26-teens_706posts.xml', '11-06-adults_706posts.xml', '11-08-20s_705posts.xml', '11-08-40s_706posts.xml', '11-08-adults_705posts.xml', '11-08-teens_706posts.xml', '11-09-20s_706posts.xml', '11-09-40s_706posts.xml', '11-09-adults_706posts.xml', '11-09-teens_706posts.xml']
# Agora vamos ler os dois corpus e armazenar as sentenças em uma mesma ndarray. Perceba que também teremos uma ndarray para indicar se o texto é formal ou não. Começamos armazenando o corpus em lists. Vamos usar apenas 500 elementos de cada, para fins didáticos.

# In [42]:
import nltk
x_data_nps = []
for fileid in nltk.corpus.nps_chat.fileids():
    x_data_nps.extend([post.text for post in nps_chat.xml_posts(fileid)])
y_data_nps = [0] * len(x_data_nps)
x_data_gut = []
for fileid in nltk.corpus.gutenberg.fileids():
    x_data_gut.extend([' '.join(sent) for sent in nltk.corpus.gutenberg.sents(fileid)])    
y_data_gut = [1] * len(x_data_gut)
x_data_full = x_data_nps[:500] + x_data_gut[:500]
print(len(x_data_full))
y_data_full = y_data_nps[:500] + y_data_gut[:500]
print(len(y_data_full))
# 1000
# 1000
# Em seguida, transformamos essas listas em ndarrays, para usarmos nas etapas de pré-processamento que já conhecemos.

# In [43]:
import numpy as np
x_data = np.array(x_data_full, dtype=object)
#x_data = np.array(x_data_full)
print(x_data.shape)
y_data = np.array(y_data_full)
print(y_data.shape)
# (1000,)
# (1000,)

# 2. Dividindo em datasets de treino e teste
# Para que a pesquisa seja confiável, precisamos avaliar os resultados em um dataset de teste. Por isso, vamos dividir os dados aleatoriamente, deixando 80% para treino e o demais para testar os resultados em breve.
# In [44]:
train_indexes = np.random.rand(len(x_data)) < 0.80
print(len(train_indexes))
print(train_indexes[:10])
# 1000
# [False  True  True False  True  True  True  True False  True]
# In [45]:
x_data_train = x_data[train_indexes]
y_data_train = y_data[train_indexes]
print(len(x_data_train))
print(len(y_data_train))
# 808
# 808

# In [46]:
x_data_test = x_data[~train_indexes]
y_data_test = y_data[~train_indexes]
print(len(x_data_test))
print(len(y_data_test))
# 192
# 192

# 3. Treinando o classificador
# Para tokenização, vamos usar a mesma função do tutorial anterior:
# In [47]:
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import wordnet
stopwords_list = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
def my_tokenizer(doc):
    words = word_tokenize(doc)   
    pos_tags = pos_tag(words)    
    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]    
    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]    
    lemmas = []
    for w in non_punctuation:
        if w[1].startswith('J'):
            pos = wordnet.ADJ
        elif w[1].startswith('V'):
            pos = wordnet.VERB
        elif w[1].startswith('N'):
            pos = wordnet.NOUN
        elif w[1].startswith('R'):
            pos = wordnet.ADV
        else:
            pos = wordnet.NOUN        
        lemmas.append(lemmatizer.lemmatize(w[0], pos))
    return lemmas
# Mas agora vamos criar um pipeline contendo o vetorizador TF-IDF, o SVD para redução de atributos e um algoritmo de classificação. Mas antes, vamos encapsular nosso algoritmo para escolher o número de dimensões para o SVD em uma classe que pode ser utilizada com o pipeline:

# In [48]:
from sklearn.decomposition import TruncatedSVD
class SVDDimSelect(object):
    def fit(self, X, y=None):               
        self.svd_transformer = TruncatedSVD(n_components=X.shape[1]/2)
        self.svd_transformer.fit(X)
        
        cummulative_variance = 0.0
        k = 0
        for var in sorted(self.svd_transformer.explained_variance_ratio_)[::-1]:
            cummulative_variance += var
            if cummulative_variance >= 0.5:
                break
            else:
                k += 1
                
        self.svd_transformer = TruncatedSVD(n_components=k)
        return self.svd_transformer.fit(X)
    
    def transform(self, X, Y=None):
        return self.svd_transformer.transform(X)
        
    def get_params(self, deep=True):
        return {}
# Finalmente podemos criar nosso pipeline:

# In [49]:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')
my_pipeline = Pipeline([('tfidf', TfidfVectorizer(tokenizer=my_tokenizer)),\
                       ('svd', SVDDimSelect()), \
                       ('clf', clf)])

# Estamos quase lá... Agora vamos criar um objeto RandomizedSearchCV que fará a seleção de hiper-parâmetros do nosso classificador (aka. parâmetros que não são aprendidos durante o treinamento). Essa etapa é importante para obtermos a melhor configuração do algoritmo de classificação. Para economizar tempo de treinamento, vamos usar um algoritmo simples o K nearest neighbors (KNN).

# In [50]:
from sklearn.grid_search import RandomizedSearchCV
import scipy
par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}
hyperpar_selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20)
# E agora vamos treinar nosso algoritmo, usando o pipeline com seleção de atributos:

# In [51]:
#print(hyperpar_selector)
hyperpar_selector.fit(X=x_data_train, y=y_data_train)

RandomizedSearchCV(cv=3, error_score='raise',
          estimator=Pipeline(steps=[('tfidf', TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,
 ...wski',
           metric_params=None, n_jobs=1, n_neighbors=10, p=2,
           weights='uniform'))]),
          fit_params={}, iid=True, n_iter=20, n_jobs=1,
          param_distributions={'clf__weights': ['uniform', 'distance'], 'clf__n_neighbors': range(1, 60)},
          pre_dispatch='2*n_jobs', random_state=None, refit=True,
          scoring='accuracy', verbose=0)

# In [52]:
print("Best score: %0.3f" % hyperpar_selector.best_score_)
print("Best parameters set:")
best_parameters = hyperpar_selector.best_estimator_.get_params()
for param_name in sorted(par.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# /home/datascience/anaconda3/lib/python3.5/site-packages/sklearn/utils/extmath.py:218: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
#   R = random_state.normal(size=(A.shape[1], size))
# /home/datascience/anaconda3/lib/python3.5/site-packages/sklearn/utils/extmath.py:317: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
#   return V[:n_components, :].T, s[:n_components], U[:, :n_components].T
# /home/datascience/anaconda3/lib/python3.5/site-packages/sklearn/utils/extmath.py:218: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future


# 4. Testando o classificador
# Agora vamos usar o classificador com o nosso dataset de testes, e observar os resultados:
# In [53]:

from sklearn.metrics import *
y_pred = hyperpar_selector.predict(x_data_test)
print(accuracy_score(y_data_test, y_pred))
# 0.697916666667

# 5. Serializando o modelo
# In [54]:
import pickle
string_obj = pickle.dumps(hyperpar_selector)

# In [55]:
model_file = open('model.pkl', 'wb')
model_file.write(string_obj)

model_file.close()
# 6. Abrindo e usando um modelo salvo 
# In [56]:
model_file = open('model.pkl', 'rb')
model_content = model_file.read()
obj_classifier = pickle.loads(model_content)
model_file.close()
res = obj_classifier.predict(["what's up bro?"])
print(res)
# [0]
# In [57]:
res = obj_classifier.predict(x_data_test)
print(accuracy_score(y_data_test, res))
# 0.697916666667

# In [66]:
res = obj_classifier.predict(x_data_test)
print(res)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
#  1 1 0 1 1 0 1 0 0 0 0 1 0 0 0 1 0 1 0 0 1 0 0 0 1 1 0 0 0 0 0 1 0 0 1 0 0
#  1 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 0 1 0 1 1 0 1 0 0 1 0 0 0 1 0 1 0 0 1 0 0
#  0 1 0 0 1 1 0]

# In [67]:
formal = [x_data_test[i] for i in range(len(res)) if res[i] == 1]
for txt in formal:
    print("%s\n" % txt)    

















res2 = obj_classifier.predict(["Emma spared no exertions to maintain this happier flow of ideas , and hoped , by the help of backgammon , to get her father tolerably through the evening , and be attacked by no regrets but her own"])

print(res2)