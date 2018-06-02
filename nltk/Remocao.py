# from nltk.corpus import stopwords
# nltk.download('stopwords')  

# stopwords.words('portuguese')
# clean_tokens = tokens[:]
# sr = stopwords.words('portuguese')
# for token in tokens:
#     if token in stopwords.words('portuguese'):
#         clean_tokens.remove(token)

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')                                                                                                                                                               
stopwords_list = stopwords.words('portuguese')
print(stopwords_list)