# Introdução ao Processamento de Linguagem Natural (PLN) Usando Python
# Professor Fernando Vieira da Silva MSc.
# Técnicas para Pré-Processamento - Parte 2
# Uma vez que o texto já foi devidamente tratado, removendo stopwords e pontuações, e aplicando stemming ou lemmatization, agora precisamos contar a frequência das palavras (ou n-grams) que utilizaremos em seguida como atributos para as técnicas de aprendizado de máquina.

# 1. TF-IDF (Term Frequency - Inverse Document Frequency)

# Term Frequency: um termo que aparece muito em um documento, tende a ser um termo importante. Em resumo, divide-se o número de vezes em que um termo apareceu pelo maior número de vezes em que algum outro termo apareceu no documento.

# tfwd = fwd / mwd

# onde:
# fwd é o número de vezes em que o termo w aparece no documento d.
# mwd é o maior valor de fwd obtido para algum termo do documento d

# Inverse Document Frequency: um termo que aparece em poucos documentos pode ser um bom descriminante. Obtem-se dividindo o número de documentos pelo número de documentos em que o termo aparece.

# idfw = log2(n / nw)

# onde:
# n é o número de documentos no corpus nw é o número de documentos em que o termo w aparece.

# Na prática, usa-se:

# tf-idf = tfwd * idfw

# Podemos calcular o TF-IDF de um corpus usando o pacote scikit-learn. Primeiramente, vamos abrir novamente o texto de Hamlet e armazenar as sentenças em uma ndarray do numpy (como se cada sentença fosse um documento do corpus):

# In [2]:
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize

hamlet_raw = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')

sents = sent_tokenize(hamlet_raw)

hamlet_np = np.array(sents)

print(hamlet_np.shape)
# (2355,)
# Agora vamos definir uma função para tokenização pelo scikit-learn.

# In [3]:
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
# E essa função será chamada pelo objeto TfidfVectorizer

# In [4]:
from sklearn.feature_extraction.text import TfidfVectorizer

hamlet_raw = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')

sents = sent_tokenize(hamlet_raw)

hamlet_np = np.array(sents)

tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)

tfs = tfidf_vectorizer.fit_transform(hamlet_np)

print(tfs.shape)
# (2355, 4305)
# In [28]:
print([k for k in tfidf_vectorizer.vocabulary_.keys()][:20])
# ['roast', 'spred', 'stoupe', 'play', 'cheff', 'vpo', 'amaz', 'beggers', 'opprest', 'rag', 'staires', 'whipt', 'turneth', 'foreknow', 'auouch', 'germaine', 'rend', 'dishonour', 'bleede', 'among']
# In [7]:
print(tfs[:50,:50])
#   (0, 17)	0.40984054295
#   (4, 5)	1.0
#   (12, 6)	0.255695651027
#   (13, 6)	0.241831885737
#   (22, 5)	0.58657679876
#   (27, 0)	0.226773205328
#   (29, 5)	0.284581755506
#   (37, 0)	0.236027784598
#   (37, 5)	0.264020196593
#   (39, 6)	0.138754555877
#   (40, 13)	0.352972670444
#   (41, 25)	0.314756261552
#   (43, 5)	0.144160249461
#   (46, 5)	0.322996303502
# 2. TF-IDF de N-gramas

# Opcionalmente, podemos obter os atributos tf-idf de n-grams, combinando as classes CountVectorizer e TfidfTransformer. Em nosso exemplo, vamos utilizar apenas trigramas:

# In [8]:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer(ngram_range=(3,3))

n_gram_counts = count_vect.fit_transform(hamlet_np)

tfidf_transformer = TfidfTransformer()

tfs_ngrams = tfidf_transformer.fit_transform(n_gram_counts)

print(tfs_ngrams.shape)
# (2355, 23483)
# 3. Redução de Dimensionalidade

# A transformação do corpus em atributos contendo as frequências TF-IDF em geral resultará numa ndarray bastante esparsa, ou seja, com muitas dimensões. Porém, além de isso tornar o treinamento de algoritmos mais demorado e custoso (computacionalmente falando), muitas dessas dimensões provavelmente são pouco representativas ou mesmo podem causar ruído durante o treinamento. Para resolver esse problema, podemos aplicar uma técnica de redução de dimensionalidade simples chamada Singular Value Decomposition (SVD).

# Essa técnica transformará os vetores da matriz original, rotacionando e escalando-os, resultando em novas representações. A redução de dimensionalidade é feita ao manter apenas as k dimensões mais representativas que escolhermos. Outra vantagem dessa técnica é que as dimensões originais são, de certa forma, "combinadas", o que resulta em uma nova forma de representar a combinação de termos. No contexto de PLN, essa técnica é conhecida como Latent Semantic Analysis (LSA)

# In [9]:
from sklearn.decomposition import TruncatedSVD

svd_transformer = TruncatedSVD(n_components=1000)

svd_transformer.fit(tfs)

print(sorted(svd_transformer.explained_variance_ratio_)[::-1][:30])
[0.043191439648641985, 0.018619648885290274, 0.015991826821457531, 0.015984996712032233, 0.01374433331875188, 0.011771580238263369, 0.0096502136679698872, 0.0095168074249540862, 0.0095050502041718905, 0.0094605420641294159, 0.0093174536752550132, 0.0092700425653072342, 0.0077114068911877344, 0.0069839976227591954, 0.0066546888854241548, 0.0064195285763126407, 0.0058853020300632126, 0.0053722779003061915, 0.0051581711906632521, 0.0049312579564470939, 0.0048675271120150803, 0.0047897360657304577, 0.0044624928648614361, 0.0043399578651715717, 0.0042969022436285731, 0.0041605617283622925, 0.0040394806462038481, 0.0039137333697165681, 0.0038585289999530435, 0.0038298284367979185]
# Agora vamos manter as dimensões até que a variância acumulada seja maior ou igual a 0.50.

# In [45]:
cummulative_variance = 0.0
k = 0
for var in sorted(svd_transformer.explained_variance_ratio_)[::-1]:
    cummulative_variance += var
    if cummulative_variance >= 0.5:
        break
    else:
        k += 1
        
print(k)
# 143
# Transformarmos novamente, mas desta vez com o número de k componentes que obtemos anteriormente.

# In [48]:
svd_transformer = TruncatedSVD(n_components=k)
svd_data = svd_transformer.fit_transform(tfs)
print(sorted(svd_transformer.explained_variance_ratio_)[::-1])
# [0.043191439648826289, 0.01861964888652147, 0.015991826835527356, 0.015984996710452587, 0.013744333334787466, 0.011771580244008924, 0.0096502136852790223, 0.009516807505667994, 0.0095050503834325385, 0.0094605423989449471, 0.0093174536840222762, 0.009270042723600538, 0.0077114073827614536, 0.0069839970437553236, 0.006654689038930575, 0.0064195307095592548, 0.0058853027737397496, 0.0053722773814358629, 0.0051581709690178513, 0.0049312580891063152, 0.0048675262334294039, 0.0047897364422535612, 0.0044625005241406192, 0.0043399605163811986, 0.0042968981982684333, 0.0041605573117342188, 0.0040394781087619638, 0.003913724919905005, 0.0038585228295508495, 0.0038298238387394429, 0.0037136224146103636, 0.003663400117768618, 0.0035824603355119099, 0.0035617267544038435, 0.0035351536998920238, 0.0034284434150083074, 0.0033450623521941829, 0.0032429636522826938, 0.003234138650533474, 0.0031762289053079997, 0.0031226961994305809, 0.0030422387153360387, 0.0029313662750773835, 0.0029182937754962499, 0.0028876792670607106, 0.0028503807516150666, 0.0028175253324609849, 0.0027991666972033546, 0.0027526264158192742, 0.0027258824851357846, 0.002716623681286506, 0.0026895580989004602, 0.0026505650097487696, 0.002596249222812254, 0.0025798007460042633, 0.0025426282658134584, 0.0025134467802293507, 0.0024692505898046748, 0.0024557682135757765, 0.0024448662551789517, 0.0024314366936469749, 0.0023811303294640524, 0.0023651150899766573, 0.0023213895646213662, 0.0023010143493181145, 0.0022749212984855944, 0.0022641822178114035, 0.0022373228141946435, 0.0022336218953467779, 0.0022232887762415081, 0.0022053184001151471, 0.0021747147489168807, 0.0021681950149381995, 0.0021356155573569982, 0.0021143251465712727, 0.0020868432498339956, 0.0020675776193300934, 0.0020398074964539923, 0.0020225884972872284, 0.0019941103682858063, 0.0019809576757169513, 0.0019786113668748711, 0.0019512644431889433, 0.0019351311876070769, 0.0019149441576676563, 0.001908474156321975, 0.0018805758953483636, 0.0018698990257254699, 0.0018604604425385893, 0.0018532853852128307, 0.001830908297771725, 0.0018126963458854435, 0.0017871449168582844, 0.001774123309069921, 0.0017573593054305918, 0.0017401103937694788, 0.0017159774549544043, 0.0017065360766679745, 0.001676012353486526, 0.0016550649502964337, 0.0016467024312047359, 0.0016374850165145293, 0.0016222148700155024, 0.0016107514466028609, 0.0015987818099820325, 0.0015666796762709152, 0.0015551885470345193, 0.0015480194178913228, 0.0015164725133708665, 0.0015112359739928008, 0.0014996302713715718, 0.001479130633463171, 0.0014681018540691059, 0.0014546077816446703, 0.001449749340834723, 0.0014351120155622554, 0.0014291059119088808, 0.0014167577905362608, 0.0014107312481184927, 0.0014035769339539007, 0.0013877790344961677, 0.001374368210299276, 0.0013705549111900717, 0.0013648157485219713, 0.0013552401014710958, 0.0013385816079103203, 0.0013266544318097739, 0.0013131808475499379, 0.0013045561190482696, 0.0012958550936225631, 0.0012800462280110104, 0.0012709172103269385, 0.0012666690580737038, 0.0012564580719857548, 0.0012515237175182875, 0.0012406487650874862, 0.0012266025095226901, 0.0012112357208175894, 0.0012079912601163408, 0.001187337654069036, 0.0011848617719568896, 0.0011810343473862112, 0.0011664874882136428]
# In [50]:
print(svd_data.shape)
# (2355, 143)