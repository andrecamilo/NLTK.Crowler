# Introdução ao Processamento de Linguagem Natural (PLN) Usando Python
# Professor Fernando Vieira da Silva MSc.
# Técnicas para Pré-Processamento
# Vamos avaliar as técnicas mais comuns para prepararmos o texto para usar com algoritmos de aprendizado de máquina logo mais.

# Como estudo de caso, vamos usar o texto de Hamlet, encontrado no corpus Gutenberg do pacote NLTK

# 1. Baixando o corpus Gutenberg

# In [8]:
import nltk

nltk.download("gutenberg")
# [nltk_data] Downloading package gutenberg to
# [nltk_data]     /home/datascience/nltk_data...
# [nltk_data]   Unzipping corpora/gutenberg.zip.
# Out[8]:
# True
# 2. Exibindo o texto "Hamlet"

# In [13]:
hamlet_raw = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')
print(hamlet_raw[:1000])
# [The Tragedie of Hamlet by William Shakespeare 1599]


# Actus Primus. Scoena Prima.

# Enter Barnardo and Francisco two Centinels.

#   Barnardo. Who's there?
#   Fran. Nay answer me: Stand & vnfold
# your selfe

#    Bar. Long liue the King

#    Fran. Barnardo?
#   Bar. He

#    Fran. You come most carefully vpon your houre

#    Bar. 'Tis now strook twelue, get thee to bed Francisco

#    Fran. For this releefe much thankes: 'Tis bitter cold,
# And I am sicke at heart

#    Barn. Haue you had quiet Guard?
#   Fran. Not a Mouse stirring

#    Barn. Well, goodnight. If you do meet Horatio and
# Marcellus, the Riuals of my Watch, bid them make hast.
# Enter Horatio and Marcellus.

#   Fran. I thinke I heare them. Stand: who's there?
#   Hor. Friends to this ground

#    Mar. And Leige-men to the Dane

#    Fran. Giue you good night

#    Mar. O farwel honest Soldier, who hath relieu'd you?
#   Fra. Barnardo ha's my place: giue you goodnight.

# Exit Fran.

#   Mar. Holla Barnardo

#    Bar. Say, what is Horatio there?
#   Hor. A peece of
# 3. Segmentação de sentenças e tokenização de palavras

# In [17]:
from nltk.tokenize import sent_tokenize

sentences = sent_tokenize(hamlet_raw)

print(sentences[:10])
# ['[The Tragedie of Hamlet by William Shakespeare 1599]\n\n\nActus Primus.', 'Scoena Prima.', 'Enter Barnardo and Francisco two Centinels.', 'Barnardo.', "Who's there?", 'Fran.', 'Nay answer me: Stand & vnfold\nyour selfe\n\n   Bar.', 'Long liue the King\n\n   Fran.', 'Barnardo?', 'Bar.']
# In [75]:
from nltk.tokenize import word_tokenize

words = word_tokenize(sentences[0])

print(words)
# ['[', 'The', 'Tragedie', 'of', 'Hamlet', 'by', 'William', 'Shakespeare', '1599', ']', 'Actus', 'Primus', '.']
# 4. Removendo stopwords e pontuação

# In [19]:
from nltk.corpus import stopwords

stopwords_list = stopwords.words('english')

print(stopwords_list)
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
# In [76]:
non_stopwords = [w for w in words if not w.lower() in stopwords_list]
print(non_stopwords)
# ['[', 'Tragedie', 'Hamlet', 'William', 'Shakespeare', '1599', ']', 'Actus', 'Primus', '.']
# In [25]:
import string
punctuation = string.punctuation
print(punctuation)
# !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
# In [77]:
# non_punctuation = [w for w in non_stopwords if not w in punctuation]

print(non_punctuation)
# ['Tragedie', 'Hamlet', 'William', 'Shakespeare', '1599', 'Actus', 'Primus']
# 5. Part of Speech (POS) Tags

# In [78]:
from nltk import pos_tag

pos_tags = pos_tag(words)

print(pos_tags)
# [('[', 'IN'), ('The', 'DT'), ('Tragedie', 'NNP'), ('of', 'IN'), ('Hamlet', 'NNP'), ('by', 'IN'), ('William', 'NNP'), ('Shakespeare', 'NNP'), ('1599', 'CD'), (']', 'NNP'), ('Actus', 'NNP'), ('Primus', 'NNP'), ('.', '.')]
# As tags indicam a classificação sintática de cada palavra no texto. Ver https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html para uma lista completa

# 6. Stemming e Lemmatization

# Stemming permite obter a "raiz" da palavra, removendo sufixos, por exemplo.

# In [86]:
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')

sample_sentence = "He has already gone"
sample_words = word_tokenize(sample_sentence)

stems = [stemmer.stem(w) for w in sample_words]

print(stems)
# ['he', 'has', 'alreadi', 'gone']
# Já lemmatization vai além de somente remover sufixos, obtendo a raiz linguística da palavra. Vamos usar as tags POS obtidas anteriormente para otimizar o lemmatizer.

# In [87]:
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

pos_tags = nltk.pos_tag(sample_words)

lemmas = []
for w in pos_tags:
    if w[1].startswith('J'):
        pos_tag = wordnet.ADJ
    elif w[1].startswith('V'):
        pos_tag = wordnet.VERB
    elif w[1].startswith('N'):
        pos_tag = wordnet.NOUN
    elif w[1].startswith('R'):
        pos_tag = wordnet.ADV
    else:
        pos_tag = wordnet.NOUN
        
    lemmas.append(lemmatizer.lemmatize(w[0], pos_tag))
    
print(lemmas)
# ['He', 'have', 'already', 'go']
# 7. N-gramas

# Além da técnica de Bag-of-Words, outra opção é utilizar n-gramas (onde "n" pode variar)

# In [88]:
non_punctuation = [w for w in words if not w.lower() in punctuation]

n_grams_3 = ["%s %s %s"%(non_punctuation[i], non_punctuation[i+1], non_punctuation[i+2]) for i in range(0, len(non_punctuation)-2)]

print(n_grams_3)
# ['The Tragedie of', 'Tragedie of Hamlet', 'of Hamlet by', 'Hamlet by William', 'by William Shakespeare', 'William Shakespeare 1599', 'Shakespeare 1599 Actus', '1599 Actus Primus']
# Também podemos usar a classe CountVectorizer, do scikit-learn:

# In [45]:
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(ngram_range=(3,3))

import numpy as np

arr = np.array([sentences[0]])

print(arr)

n_gram_counts = count_vect.fit_transform(arr)

print(n_gram_counts)

print(count_vect.vocabulary_)
# ['[The Tragedie of Hamlet by William Shakespeare 1599]\n\n\nActus Primus.']
#   (0, 5)	1
#   (0, 6)	1
#   (0, 3)	1
#   (0, 2)	1
#   (0, 1)	1
#   (0, 7)	1
#   (0, 4)	1
#   (0, 0)	1
# {'the tragedie of': 5, 'hamlet by william': 2, 'william shakespeare 1599': 7, 'of hamlet by': 3, 'tragedie of hamlet': 6, 'by william shakespeare': 1, '1599 actus primus': 0, 'shakespeare 1599 actus': 4}
# Agora, vamos contar os n-grams (no nosso caso, trigramas) de todas as sentenças do texto:

# In [54]:
arr = np.array(sentences)

n_gram_counts = count_vect.fit_transform(arr)

print(n_gram_counts[:20])

print([k for k in count_vect.vocabulary_.keys()][:20])
#   (0, 18380)	1
#   (0, 20473)	1
#   (0, 13004)	1
#   (0, 6525)	1
#   (0, 2993)	1
#   (0, 22196)	1
#   (0, 15662)	1
#   (0, 0)	1
#   (2, 4728)	1
#   (2, 1884)	1
#   (2, 661)	1
#   (2, 5712)	1
#   (6, 12125)	1
#   (6, 1269)	1
#   (6, 11049)	1
#   (6, 16697)	1
#   (6, 20823)	1
#   (6, 23416)	1
#   (7, 10278)	1
#   (7, 10238)	1
#   (7, 17966)	1
#   (11, 22884)	1
#   (11, 3373)	1
#   (11, 11461)	1
#   (11, 3125)	1
#   :	:
#   (13, 474)	1
#   (13, 376)	1
#   (13, 15974)	1
#   (13, 1728)	1
#   (14, 6885)	1
#   (14, 22949)	1
#   (14, 6494)	1
#   (16, 12610)	1
#   (16, 11592)	1
#   (18, 8261)	1
#   (18, 22902)	1
#   (18, 4207)	1
#   (18, 11162)	1
#   (18, 7983)	1
#   (18, 819)	1
#   (18, 10805)	1
#   (18, 18247)	1
#   (18, 15131)	1
#   (18, 13169)	1
#   (18, 12059)	1
#   (18, 21211)	1
#   (18, 2339)	1
#   (18, 18603)	1
#   (19, 7983)	1
#   (19, 4738)	1
['vnweeded garden that', 'stronger then either', 'to th court', 'walke in death', 'leysure but wilt', 'this physicke but', 'her paint an', 'neyther hauing the', 'cruell onely to', 'fall cursing like', 'norwey ouercome with', 'rose of the', 'my honor lord', 'sonne as twere', 'killes my father', 'fat vs and', 'maiesties might by', 'winnowed opinions and', 'that you vouchsafe', 'lady whilst this']
# In [6]:
from nltk import word_tokenize

frase = 'o cachorro correu atrás do gato'


ngrams = ["%s %s %s" % (nltk.word_tokenize(frase)[i], \
                      nltk.word_tokenize(frase)[i+1], \
                      nltk.word_tokenize(frase)[i+2]) \
          for i in range(len(nltk.word_tokenize(frase))-2)]

print(ngrams)
# ['o cachorro correu', 'cachorro correu atrás', 'correu atrás do', 'atrás do gato']