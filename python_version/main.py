import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer 
import pickle
from nltk.tokenize import RegexpTokenizer

#reading datasets
df1 = pd.read_csv('/Users/as.tarlan02/Desktop/articles1.csv')
df2 = pd.read_csv('/Users/as.tarlan02/Desktop/articles2.csv')
df3 = pd.read_csv('/Users/as.tarlan02/Desktop/articles3.csv')

#merging into one dataframe
frames = [df1, df2, df3]
df = pd.concat(frames)

#caching dataframe
#pickle.dump(df, open('df.pickle', 'wb'))
#df = pickle.load(open('df.pickle', 'rb'))

#preprocessing function for input data
def preprocessing(text):
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer('\w+')
    word_list = tokenizer.tokenize(text.lower())
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output

preprocessed_data = list()

for paragraph in df['content']:
    preprocessed_data.append(preprocessing(paragraph))

#using tf-idf vectorization on preprocessed data
vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
vectors = vectorizer.fit_transform(preprocessed_data)

#caching the vectorized data
#vectors = pickle.load(open('vectors.pickle', 'rb'))
#vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

while True:
    query = input() #asking user for a query
    preprocessed_query = preprocessing(query) #preprocessing the user's query
    query_vector = vectorizer.transform([preprocessed_query])
    
    result = df[['id','title']]
    result['cossim'] = (vectors * query_vector.T).todense()
    result.sort_values(by=['cossim'], ascending=False, inplace=True)
    result['title'] = result['title']

    # result.index = result['id']
    print(result[['id', 'title']].head().to_string(index=False))