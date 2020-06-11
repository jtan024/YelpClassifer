#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Run in terminal or command prompt
# python3 -m spacy download en
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
from spacy.lang.en import English
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('reviews_100k.csv', error_bad_lines=False, engine='python')
df = df.drop(columns=['user_id', 'review_id', 'votes.cool', 'business_id', 'votes.funny', 'stars', 'votes.useful', 'date', 'type'], axis=1)
df = df.dropna(subset=['text'])
data = df.text.values.tolist()# Remove Emails
data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]# Remove new line characters
data = [re.sub(r'\s+', ' ', sent) for sent in data]# Remove distracting single quotes
data = [re.sub(r"\'", "", sent) for sent in data]


# In[8]:


##pprint(data[:1])
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data_words = list(sent_to_words(data))
##print(data_words[:1])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): #'NOUN', 'ADJ', 'VERB', 'ADV'
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out
# Initialize spacy ‘en’ model, keeping only tagger component (for efficiency)
# Run in terminal: python -m spacy download en
# Initialize spacy ‘en’ model, keeping only tagger component (for efficiency)
# Run in terminal: python -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'VERB']) #select noun and verb
print(data_lemmatized[:2])

vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,
# minimum reqd occurences of a word 
                             stop_words='english',             
# remove stop words
                             lowercase=True,                   
# convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  
# num chars > 3
                             # max_features=50000,             
# max number of uniq words    
                            )
                             
data_vectorized = vectorizer.fit_transform(data_lemmatized)
                             
                             # Build LDA Model
lda_model = LatentDirichletAllocation(n_components=20,               # Number of topics
                                      max_iter=10,               
# Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          
# Random state
                                      batch_size=128,            
# n docs in each learning iter
                                      evaluate_every = -1,       
# compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               
# Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(data_vectorized)
print(lda_model)  # Model attributes


# In[9]:


# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))# See model parameters
pprint(lda_model.get_params())


# In[14]:


# Create Document — Topic Matrix
lda_output = lda_model.transform(data_vectorized)# column names
topicnames = ['Topic' + str(i) for i in range(lda_model.n_components)]# index names
docnames = ['Doc' + str(i) for i in range(len(data))]# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic# Styling
def color_green(val):
 color = 'green' if val > .1 else 'black'
 return 'color: {col}'.format(col=color)
def make_bold(val):
 weight = 700 if val > .1 else 400
 return 'font-weight: {weight}'.format(weight=weight)# Apply Style
df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)

# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(lda_model.components_)# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames# View
df_topic_keywords.head()

# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=15)# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]

Topics = ["price","food","ambience","price","service", 
          "price", "service", "food", "food", "ambience", "food", "food", "none", "food", "food", "service", "service", "none", "food", "food"]
df_topic_keywords["Topics"]=Topics
df_topic_keywords


# In[56]:


# Define function to predict topic for a given text document.
nlp = spacy.load('en', disable=['parser', 'ner'])
def predict_topic(text, nlp=nlp):
    global sent_to_words
    global lemmatization# Step 1: Clean with simple_preprocess
    mytext_2 = list(sent_to_words(text))# Step 2: Lemmatize
    mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])# Step 3: Vectorize transform
    mytext_4 = vectorizer.transform(mytext_3)# Step 4: LDA Transform
    topic_probability_scores = lda_model.transform(mytext_4)
    topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), 1:14].values.tolist()
    
    # Step 5: Infer Topic
    infer_topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), -1]
    
    #topic_guess = df_topic_keywords.iloc[np.argmax(topic_probability_scores), Topics]
    return infer_topic, topic, topic_probability_scores# Predict the topic

#read from csv to array
import csv

mytext = []
actualTopic = []

with open('./Yelp.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            mytext.append(row[0])
            actualTopic.append(row[1])
            line_count += 1

infer_topic = []
topic = []
prob_scores = []

for i in range(len(mytext)):
    singleText = [""]
    singleText[0] = mytext[i]
    var1, var2, var3 = predict_topic(singleText)
    infer_topic.append(var1)
    topic.append(var2)
    prob_scores.append(var3)

# print(topic)
print(infer_topic)


# In[16]:


def apply_predict_topic(text):
 text = [text]
 infer_topic, topic, prob_scores = predict_topic(text = text)
 return(infer_topic)
df["Topic_key_word"]= df['text'].apply(apply_predict_topic)
df


# In[61]:


check_df = pd.DataFrame({'actual_label': actualTopic, 'prediction': infer_topic, 'review':mytext})
check_df


# In[65]:


from sklearn.metrics import accuracy_score, precision_score, recall_score
print('Accuracy score: ', accuracy_score(actualTopic, infer_topic))
print('Precision score: ', precision_score(actualTopic, infer_topic, average='micro'))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(actualTopic, infer_topic)
sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False,
xticklabels=['price', 'service', 'food', 'ambience', 'none'], yticklabels=['price', 'service', 'food', 'ambience', 'none'])
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[ ]:

# Predict the topic
while inp != 'quit':
    inp = input("Enter a review: ")
    print("Type 'quit' to exit ")
    arr = []
    arr.append(inp)
    infer_topic, topic, prob_scores = predict_topic(text = arr)
    print(infer_topic)


