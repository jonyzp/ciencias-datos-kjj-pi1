# %% [markdown]
# # Proyecto Integrador 1
# ---
# Presentado por:
# * Karla Orozco
# * Jonathan zapata
# * Juan Fernando Gallego

# %% [markdown]
# # Trusted -> Refined

# %% [markdown]
# ### Imports and Instances

# %%
# Import common libraries:
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# %%
# Import specific libraries:
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# %%
## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

# from pyspark.sql import SparkSession
# spark = SparkSession.builder.master("local[1]") \
#                     .appName('SparkByExamples.com') \
#                     .getOrCreate()

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# %% [markdown]
# ### Read data from S3

# %%
input_APPL_dyf = glueContext.create_dynamic_frame_from_options(
  connection_type = "s3",
  connection_options = {
    "paths": ["s3://pi1-kjj/trusted/AAPL/"]},
  format = "parquet")


# %%
## @type: DataSource
## @args: [database = "pi1-kjj-trusted", table_name = "company1_csv", transformation_ctx = "DataSource0"]
## @return: company_dynamicframe
## @inputs: []
company_dynamicframe = glueContext.create_dynamic_frame.from_catalog(
       database = "pi1-kjj-trusted",
       table_name = "company_csv",
       transformation_ctx = "company_dynamicframe")
# company_dynamicframe = spark.read.csv('raw/Company.csv', header='true', inferSchema='true')
# company_dynamicframe.printSchema()

company_dataframe = company_dynamicframe.toDF()

# company_dataframe = company_dynamicframe

# %%
# company_dataframe.show(n=6)

# %%

APPL_pd_df = input_APPL_dyf.toDF().toPandas()

#APPL_pd_df['test'] = 'testing'

# %% [markdown]
# ### Proporcion para cada sentimiento del total de tweets por dia
def n_sentiment(x):
  #tabla cruzada
  sentiment=pd.crosstab(x['post_date'],x['sentiment'])
  #calcular la sumatoria
  n_tweets=sentiment.apply(lambda x: np.sum(x), axis=1)
  n_tweets.name='n_tweets'
  n_tweets=n_tweets.to_frame()
  #convertir los valores en porcentaje del total
  sentiment=sentiment.apply(lambda x: x/np.sum(x), axis=1)
  #concatenar porcentajes con sumatoria
  sentiment=pd.concat([sentiment,n_tweets], axis=1)
  
  return sentiment

sentiment=n_sentiment(APPL_pd_df)
print('sentiment:')
sentiment.head(2)

# %% [markdown]
# ### Group by day by sentiment
def transform_data(x):
    #campos calculados
    agg=x.groupby(['post_date','sentiment']).agg(
        bow=('bow', lambda x: [item for item in x]),
        eng=('total_engagement', lambda x: [item for item in x]),
        sum_eng=('total_engagement',sum),
    ).unstack()
    #rellenar NaN con cero para 'sum_eng'
    agg.sum_eng=agg.sum_eng.fillna(0)
    #concatenar índice multiple en índice único
    agg.columns=agg.columns.map('_'.join)
    return agg

agg = transform_data(APPL_pd_df)

APPL_pd_df = pd.concat([agg,sentiment], axis=1)

print('APPL_pd_df:')
APPL_pd_df.head(2)

# %% [markdown]
# ### Dispersion

def dispersion_tweets_content(tweets_cont_list, tweets_eng_list):
    
    #return zero if list´s length is 0 or is nan object
    if ((isinstance(tweets_cont_list, list)) | (isinstance(tweets_eng_list, list))):
        
        #if lists has nan, remove it by its position
        pos_nan=[ind for ind, x in enumerate(tweets_cont_list) if str(x) == 'nan']
        tweets_cont_list=[x for ind, x in enumerate(tweets_cont_list) if not np.isin(ind,pos_nan)]
        tweets_eng_list=[x for ind, x in enumerate(tweets_eng_list) if not np.isin(ind,pos_nan)]
        
        if ((len(tweets_cont_list)==0) | (len(tweets_eng_list)==0)):
            return 0
    else:
        if ((pd.isna(tweets_cont_list)) | (pd.isna(tweets_eng_list))):
            return 0
    
    tfidf_matrix=TfidfVectorizer().fit_transform(tweets_cont_list)
    cosine_sim=linear_kernel(tfidf_matrix, tfidf_matrix) #cosine_similarity
    dispersion_sim=(1-cosine_sim)
    
    tweets_eng_array=np.array(tweets_eng_list)
    unos=np.ones((1,len(tweets_eng_array)))
    tweets_eng_matrix=tweets_eng_array.reshape(1,-1)*(unos.T@unos)
    np.fill_diagonal(tweets_eng_matrix,0)
    
    if np.sum(tweets_eng_matrix)==0:
        return 0
    
    return np.sum(dispersion_sim*tweets_eng_matrix)/np.sum(tweets_eng_matrix)

APPL_pd_df['dispersion_negative']=APPL_pd_df.apply(lambda x: dispersion_tweets_content(x.bow_negative, x.eng_negative), axis=1)
APPL_pd_df['dispersion_neutral']=APPL_pd_df.apply(lambda x: dispersion_tweets_content(x.bow_neutral, x.eng_neutral), axis=1)
APPL_pd_df['dispersion_positive']=APPL_pd_df.apply(lambda x: dispersion_tweets_content(x.bow_positive, x.eng_positive), axis=1)

APPL_pd_df = APPL_pd_df.assign(dispersion_tweets_content=(APPL_pd_df.dispersion_negative*APPL_pd_df.sum_eng_negative +
  APPL_pd_df.dispersion_neutral * APPL_pd_df.sum_eng_neutral +
  APPL_pd_df.dispersion_positive * APPL_pd_df.sum_eng_positive) /
  (APPL_pd_df.sum_eng_negative+APPL_pd_df.sum_eng_neutral+APPL_pd_df.sum_eng_positive))

APPL_pd_df.dispersion_tweets_content.describe()

APPL_pd_df = APPL_pd_df.drop(['bow_negative','bow_neutral','bow_positive',
                'eng_negative','eng_neutral','eng_positive',
                'dispersion_negative','dispersion_neutral','dispersion_positive'], axis=1)

# %% [markdown]
# ### Get change percentage

# mvv = company_dataframe.select('ticker_symbol').rdd.flatMap(lambda x: x).collect()
mvv = [row['ticker_symbol'] for row in company_dataframe.collect()]
print('mvv:')
print(mvv)
print('Downloading yfinance data:')
prices = yf.download(mvv, start="2014-12-31", end="2020-12-31")['Adj Close']
price=pd.DataFrame(index=pd.date_range(start="2014-12-31",end="2020-12-31"))
price.index.name='Date'
price = pd.concat([price,prices],axis=1)
price=price.fillna(method='ffill')
# print(price.isnull().sum().sum()==0)
# print(all(pd.date_range(start="2015-01-01",end="2020-12-31").isin(price.index)))
change_price=price.pct_change()

# print('change_price.index[:10]')
# print(change_price.index[:10])

# print('change_price.loc[change_price.index==2014-12-31].index:')
# print(change_price.loc[change_price.index=='2014-12-31'].index)

## Eliminar periodos fuera del rango de analisis:
change_price.drop(change_price.loc[change_price.index=='2014-12-31'].index, axis=0, inplace=True)

# Se establecerá los valores umbrales, como la mediana +/- n desviaciones estandar:

n_desv_est=2

umbrales_inf=change_price.median()-n_desv_est*np.sqrt(change_price.var())
umbrales_sup=change_price.median()+n_desv_est*np.sqrt(change_price.var())

Y_AAPL=change_price['AAPL'].apply(lambda x: 1 if x>umbrales_sup['AAPL'] else (2 if x<umbrales_inf['AAPL'] else 0)).to_frame()
#Cambiarle el tipo al índice
Y_AAPL.index=Y_AAPL.index.to_series().dt.date
#Renombrar la columna
Y_AAPL.rename(columns={'AAPL': 'Y'}, inplace=True)
#Seleccionar los periodos con datos
Y_AAPL=Y_AAPL.loc[APPL_pd_df.index]

APPL_pd_df = pd.concat([APPL_pd_df,Y_AAPL], axis=1)

# %%

# Pandas to Spark
df_spark = spark.createDataFrame(APPL_pd_df)

# %%

# Export data to S3:
from awsglue.dynamicframe import DynamicFrame
AAPL_dyf = DynamicFrame.fromDF(df_spark, glueContext, 'AAPL')
# partitioned_dataframe = inputDyf.toDF().repartition(1)
partitioned_dataframe = AAPL_dyf.repartition(1)

glueContext.write_dynamic_frame.from_options(
  frame = partitioned_dataframe,
  connection_type = 's3',
  connection_options = {'path': 's3://pi1-kjj/refined/AAPL'},
  format = 'csv')

print('End storing to S3')

# %%

# # https://towardsdatascience.com/sentiment-analysis-with-pyspark-bc8e83f80c35
# from pyspark.ml.feature import NGram, Tokenizer, StopWordsRemover
# from pyspark.ml.feature import HashingTF, IDF, StringIndexer, VectorAssembler
# from pyspark.ml import Pipeline
# from pyspark.ml.linalg import Vector

# indexer = StringIndexer(inputCol='category', outputCol='label')
# tokenizer = Tokenizer(inputCol="sentence", outputCol="sentence_tokens")
# remove_stop_words = StopWordsRemover(inputCol="sentence_tokens", outputCol="filtered")
# unigrammer = NGram(n=1, inputCol="filtered", outputCol="tokens") 
# hashingTF = HashingTF(inputCol="tokens", outputCol="hashed_tokens")
# idf = IDF(inputCol="hashed_tokens", outputCol="tf_idf_tokens")

# clean_up = VectorAssembler(inputCols=['tf_idf_tokens'], outputCol='features')

# data_prep_pipe = Pipeline(
#     stages=[indexer, tokenizer, remove_stop_words, unigrammer, hashingTF, idf, clean_up]
# )
# transformed = data_prep_pipe.fit(spark_df).transform(spark_df)
# clean_data = transformed.select(['label','features'])


# # %% [markdown]
# # 
# # Train the model
# # 

# # %%

# from pyspark.ml.classification import NaiveBayes
# nb = NaiveBayes()
# (training,testing) = clean_data.randomSplit([0.7,0.3], seed=12345)
# model = nb.fit(training)
# test_results = model.transform(testing)


# # %% [markdown]
# # Evaluate Model
# # 

# # %%

# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# acc_eval = MulticlassClassificationEvaluator()
# acc = acc_eval.evaluate(test_results)
# print("Accuracy of model at predicting label was: {}".format(acc))

# %%


# %%


# %%
# -*- coding: utf-8 -*-

# Import common libraries:
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
# Import specific libraries:
import yfinance as yf
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

###############CÓDIGO KARLA#############

#IMPORTAR LIBRERIAS
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
!pip install nltk

#CARGA DE ARCHIVOS
path_X1 = '/Users/karlaorozco/OneDrive_UniversidadEAFIT/Escritorio/INFORMACIÓN PERSONAL/1_MAESTRIA/SEMESTRE1/PI/Notebooks/Company_Tweet.csv'
path_X2 = '/Users/karlaorozco/OneDrive_UniversidadEAFIT/Escritorio/INFORMACIÓN PERSONAL/1_MAESTRIA/SEMESTRE1/PI/Notebooks/Company1.csv'
path_X3 = '/Users/karlaorozco/OneDrive_UniversidadEAFIT/Escritorio/INFORMACIÓN PERSONAL/1_MAESTRIA/SEMESTRE1/PI/Notebooks/Tweet.csv'
path_X4 = '/Users/karlaorozco/OneDrive_UniversidadEAFIT/Escritorio/INFORMACIÓN PERSONAL/1_MAESTRIA/SEMESTRE1/PI/Notebooks/CompanyValues.csv'

company_tweet = pd.read_csv(path_X1)
company = pd.read_csv(path_X2)
tweet = pd.read_csv(path_X3)
company_values = pd.read_csv(path_X4)

#CAMBIAR FECHA EN TABLA TWEET
tweet['post_date'] = pd.to_datetime(tweet['post_date'],unit='s')

#TOKENIZACIÓN DE 2000 TWEETS
tweet_head = tweet.head(2000).copy()
tweet_head['body_tokenized'] = tweet_head.apply(lambda row: nltk.word_tokenize(str(row['body'])), axis=1)
tweet_head['body_tokenized']

# STOPWORDS EN NLTK
stop_words_nltk = set(stopwords.words('english'))
print(f'Stopwords length: {len(stop_words_nltk)}')
print(f'Stopwords: {stop_words_nltk}')

#ELIMINACI´ÓN DE CARACTERES ESPECIALES 

# ELIMINAR tokens de long = 1
# ELIMINAR caracteres que no sean alfanumericos
# NUEVAMENTE ELIMINAR tokens de long = 1
# REMOVER tokens conformados solo por numeros, ya que no necesitamos buscar expresiones solo numéricas
# REMOVER stop words
refined_tokens_by_file = []
for (idx, tokens) in enumerate(tweet_head['body_tokenized']):
  tokens = [re.sub(r'[^A-Za-z0-9]+','',w) for w in tokens]
  # tokens=[word for word in tokens if word.isalpha()] si en vez de re.sub(r'[^A-Za-z0-9]+','',w) hace esto, que pasa?
  tokens = [w.lower() for w in tokens if len(w)>1]
  for i, w in reversed(list(enumerate(tokens))):
    try:
      # Preguntamos si el token es solo numerico y si si, lo eliminamos
      if (w.isnumeric()):
        tokens.pop(i)
    except:
      pass
  tokens = [w for w in tokens if w not in stop_words_nltk]
  refined_tokens_by_file.append(tokens)
  #fdist = nltk.FreqDist(refined_tokens_by_file)
  # extract top 20 words
  #topwords = fdist.most_common(20)
  #print(f"numero de palabras finales en {df['filename'][idx]} = {len(fdist)}")
  #print(f"Top 20: {topwords}")
  # graficar los 20 términos más frecuentes:
  #x,y = zip(*topwords)
  #plt.figure(figsize=(15,10))
  #plt.bar(x,y)
  #plt.xlabel("Word")
  #plt.ylabel("frecuency")
  #plt.xticks(rotation=90)
  #plt.show()
  #print('------------------------------------------')
tweet_head['body_tokenized'] = refined_tokens_by_file
tweet_head.head()

# LEMMA con NLTK

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_tokens_by_file = []
for (idx, tokens) in enumerate(refined_tokens_by_file):
  lemmatized_tokens = [wordnet_lemmatizer.lemmatize(w) for w in tokens ]
  lemmatized_tokens_by_file.append(lemmatized_tokens)
  #fdist = nltk.FreqDist(lemmatized_tokens)
  # extract top 20 words
  #topwords = fdist.most_common(20)
  #print(f"numero de palabras finales en {tweet_head['body_tokenized'][idx]} = {len(fdist)}")
  #print(f"Top 20: {topwords}")
  #x,y = zip(*topwords)
  #plt.figure(figsize=(15,10))
  #plt.bar(x,y)
  #plt.xlabel("Word")
  #plt.ylabel("frecuency")
  #plt.xticks(rotation=90)
  #plt.show()
  #print('------------------------------------------')

tweet_head['lemmatized_tokens'] = lemmatized_tokens_by_file
tweet_head.head()

#CREACIÓN DE COLUMNA LIST_LEMMA EN EL DF
tweet_head['list_lemma'] = tweet_head['lemmatized_tokens'].apply(lambda c: list(w for w in c if not any(x.isdigit() for x in w)))

#SE UNEN LOS TOKENS REFINADOS Y SE CREA UNA COLUMNA LLAMADA CLEAN_TWEET
tweet_head['clean_tweet'] = tweet_head['list_lemma'].apply(lambda c: ' '.join(c))

#ANÁLISIS DE SENTIMIENTOS
#LIBRERIAS
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
nltk.download('vader_lexicon')

#SE CREA COLUMNA CON LOS SCORES DE LOS SENTIMIENTOS Y SE LLEVA A LA COLUMNA SENTIMENT
tweet_head['score_vader'] = tweet_head['clean_tweet'].apply(lambda c: SentimentIntensityAnalyzer().polarity_scores(c)['compound'])

#SE ASIGNA CONDICIONES PARA DEFINIR SI ES NEGATIVO, NEUTRAO O POSITIVO
conditions = [
    (tweet_head["score_vader"] >= .05),
    (tweet_head["score_vader"] > -.05) & (tweet_head["score_vader"] < .05),
    (tweet_head["score_vader"] <= -.05),
]
choices = ['positive', 'neutral', 'negative']
tweet_head['sentiment'] = np.select(conditions, choices)
tweet_head['sentiment'].value_counts()

#TRANSFORMACI´ÓN DE SENTIMIENTOS EN ETIQUETAS [0,1,2]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
tweet_head['sentiment_code'] = le.fit_transform(tweet_head['sentiment'])
np.unique(tweet_head['sentiment_code'])

#ADICIONAR AL DF EMPRESAS DE LAS QUE HABLA CADA TWEET
company_tweet=company_tweet.loc[company_tweet['tweet_id'].isin(tweet_head['tweet_id'])]
ticker_symbol_group = company_tweet.groupby('tweet_id')['ticker_symbol'].unique()
#Renombrar y mostrar la serie
ticker_symbol_group.rename("ticker_symbol_group", inplace=True)
ticker_symbol_group.head(2)
#Realizar combinación mencionada entre la tabla agrupada "ticker_symbol_group" y "tweet" por la columna "tweet_id":
tweet_head = tweet_head.merge(ticker_symbol_group, how='left',left_on='tweet_id',right_on='tweet_id')

#################### ****POR EMPRESA**** ########################

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

#SEPARAR CADA EMPRESA CON SUS SENTIMIENTOS Y HACER LOS EMBEDDINGS

#********************* AAPL **************************
df_aapl_idx = tweet_head.ticker_symbol_group.apply(lambda a: 'AAPL' in a)
df_aapl= tweet_head[df_aapl_idx]

#NEGATIVE = 0
df_aapl_negative = df_aapl[df_aapl['sentiment_code']==0]
### JOIN ALL TWEETS NEGATIVE
df_aapl_negative_alltweets  = df_aapl_negative['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_aapl_negative = [df_aapl_negative_alltweets]
vocabulary_aapl_negative = np.unique(df_aapl_negative_alltweets.split(" "))
pipe_aapl_nega = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_aapl_negative)),
                  ('tfid', TfidfTransformer())]).fit(corpus_aapl_negative)
pipe_aapl_nega['count'].transform(corpus_aapl_negative).toarray()
pipe_aapl_nega['tfid'].idf_
pipe_aapl_nega.transform(corpus_aapl_negative).shape

#NEUTRAL = 1
df_aapl_neutral = df_aapl[df_aapl['sentiment_code']==1]
### JOIN ALL TWEETS NEGATIVE
df_aapl_neutral_alltweets  = df_aapl_neutral['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_aapl_neutral = [df_aapl_neutral_alltweets]
vocabulary_aapl_neutral = np.unique(df_aapl_neutral_alltweets.split(" "))
pipe_aapl_neu = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_aapl_neutral)),
                  ('tfid', TfidfTransformer())]).fit(corpus_aapl_neutral)
pipe_aapl_neu['count'].transform(corpus_aapl_neutral).toarray()
pipe_aapl_neu['tfid'].idf_
pipe_aapl_neu.transform(corpus_aapl_neutral).shape

#POSITIVE = 2
df_aapl_positive = df_aapl[df_aapl['sentiment_code']==2]
### JOIN ALL TWEETS NEUTRAL
df_aapl_positive_alltweets  = df_aapl_positive['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_aapl_positive = [df_aapl_positive_alltweets]
vocabulary_aapl_positive = np.unique(df_aapl_positive_alltweets.split(" "))
pipe_aapl_pos = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_aapl_positive)),
                  ('tfid', TfidfTransformer())]).fit(corpus_aapl_positive)
pipe_aapl_pos['count'].transform(corpus_aapl_positive).toarray()
pipe_aapl_pos['tfid'].idf_
pipe_aapl_pos.transform(corpus_aapl_positive).shape

#********************* TSLA **************************
df_tsla_idx = tweet_head.ticker_symbol_group.apply(lambda a: 'TSLA' in a)
df_tsla= tweet_head[df_tsla_idx]

#NEGATIVE = 0
df_tsla_negative = df_tsla[df_tsla['sentiment_code']==0]
### JOIN ALL TWEETS NEGATIVE
df_tsla_negative_alltweets  = df_tsla_negative['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_tsla_negative = [df_tsla_negative_alltweets]
vocabulary_tsla_negative = np.unique(df_tsla_negative_alltweets.split(" "))
pipe_tsla_nega = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_tsla_negative)),
                  ('tfid', TfidfTransformer())]).fit(corpus_tsla_negative)
pipe_tsla_nega['count'].transform(corpus_tsla_negative).toarray()
pipe_tsla_nega['tfid'].idf_
pipe_tsla_nega.transform(corpus_tsla_negative).shape

#NEUTRAL = 1
df_tsla_neutral = df_tsla[df_tsla['sentiment_code']==1]
### JOIN ALL TWEETS NEUTRAL
df_tsla_neutral_alltweets  = df_tsla_neutral['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_tsla_neutral = [df_tsla_neutral_alltweets]
vocabulary_tsla_neutral = np.unique(df_tsla_neutral_alltweets.split(" "))
pipe_tsla_neu = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_tsla_neutral)),
                  ('tfid', TfidfTransformer())]).fit(corpus_tsla_neutral)
pipe_tsla_neu['count'].transform(corpus_tsla_neutral).toarray()
pipe_tsla_neu['tfid'].idf_
pipe_tsla_neu.transform(corpus_tsla_neutral).shape

#POSITIVE = 2
df_tsla_positive = df_tsla[df_tsla['sentiment_code']==2]
### JOIN ALL TWEETS POSITIVE
df_tsla_positive_alltweets  = df_tsla_positive['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_tsla_pos = [df_tsla_positive_alltweets]
vocabulary_tsla_pos = np.unique(df_tsla_positive_alltweets.split(" "))
pipe_tsla_pos = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_tsla_pos)),
                  ('tfid', TfidfTransformer())]).fit(corpus_tsla_pos)
pipe_tsla_pos['count'].transform(corpus_tsla_pos).toarray()
pipe_tsla_pos['tfid'].idf_
pipe_tsla_pos.transform(corpus_tsla_pos).shape

#********************* AMZN **************************
df_amzn_idx = tweet_head.ticker_symbol_group.apply(lambda a: 'AMZN' in a)
df_amzn= tweet_head[df_amzn_idx]

#NEGATIVE = 0
df_amzn_negative = df_amzn[df_amzn['sentiment_code']==0]
### JOIN ALL TWEETS NEGATIVE
df_amzn_negative_alltweets  = df_amzn_negative['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_amzn_negative = [df_amzn_negative_alltweets]
vocabulary_amzn_negative = np.unique(df_amzn_negative_alltweets.split(" "))
pipe_amzn_nega = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_amzn_negative)),
                  ('tfid', TfidfTransformer())]).fit(corpus_amzn_negative)
pipe_amzn_nega['count'].transform(corpus_amzn_negative).toarray()
pipe_amzn_nega['tfid'].idf_
pipe_amzn_nega.transform(corpus_amzn_negative).shape

#NEUTRAL = 1
df_amzn_neutral = df_amzn[df_amzn['sentiment_code']==1]
### JOIN ALL TWEETS NEUTRAL
df_amzn_neutral_alltweets  = df_amzn_neutral['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_amzn_neutral = [df_amzn_neutral_alltweets]
vocabulary_amzn_neutral = np.unique(df_amzn_neutral_alltweets.split(" "))
pipe_amzn_neu = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_amzn_neutral)),
                  ('tfid', TfidfTransformer())]).fit(corpus_amzn_neutral)
pipe_amzn_neu['count'].transform(corpus_amzn_neutral).toarray()
pipe_amzn_neu['tfid'].idf_
pipe_amzn_neu.transform(corpus_amzn_neutral).shape

#POSITIVE = 2
df_amzn_positive = df_amzn[df_amzn['sentiment_code']==2]
### JOIN ALL TWEETS POSITIVE
df_amzn_positive_alltweets  = df_amzn_positive['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_amzn_pos = [df_amzn_positive_alltweets]
vocabulary_amzn_pos = np.unique(df_amzn_positive_alltweets.split(" "))
pipe_amzn_pos = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_amzn_pos)),
                  ('tfid', TfidfTransformer())]).fit(corpus_amzn_pos)
pipe_amzn_pos['count'].transform(corpus_amzn_pos).toarray()
pipe_amzn_pos['tfid'].idf_
pipe_amzn_pos.transform(corpus_amzn_pos).shape

#********************* GOOG **************************
df_goog_idx = tweet_head.ticker_symbol_group.apply(lambda a: 'GOOG' in a)
df_goog= tweet_head[df_goog_idx]

#NEGATIVE = 0
df_goog_negative = df_goog[df_goog['sentiment_code']==0]
### JOIN ALL TWEETS NEGATIVE
df_goog_negative_alltweets  = df_goog_negative['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_goog_negative = [df_goog_negative_alltweets]
vocabulary_goog_negative = np.unique(df_goog_negative_alltweets.split(" "))
pipe_goog_nega = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_goog_negative)),
                  ('tfid', TfidfTransformer())]).fit(corpus_goog_negative)
pipe_goog_nega['count'].transform(corpus_goog_negative).toarray()
pipe_goog_nega['tfid'].idf_
pipe_goog_nega.transform(corpus_goog_negative).shape

#NEUTRAL = 
df_goog_neutral = df_goog[df_goog['sentiment_code']==1]
### JOIN ALL TWEETS NEUTRAL
df_goog_neutral_alltweets  = df_goog_neutral['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_goog_neutral = [df_goog_neutral_alltweets]
vocabulary_goog_neutral = np.unique(df_goog_neutral_alltweets.split(" "))
pipe_goog_neu = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_goog_neutral)),
                  ('tfid', TfidfTransformer())]).fit(corpus_goog_neutral)
pipe_goog_neu['count'].transform(corpus_goog_neutral).toarray()
pipe_goog_neu['tfid'].idf_
pipe_goog_neu.transform(corpus_goog_neutral).shape

#POSITIVE = 2
df_goog_positive = df_goog[df_goog['sentiment_code']==2]
### JOIN ALL TWEETS POSITIVE
df_goog_positive_alltweets  = df_goog_positive['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_goog_pos = [df_goog_positive_alltweets]
vocabulary_goog_pos = np.unique(df_goog_positive_alltweets.split(" "))
pipe_goog_pos = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_goog_pos)),
                  ('tfid', TfidfTransformer())]).fit(corpus_goog_pos)
pipe_goog_pos['count'].transform(corpus_goog_pos).toarray()
pipe_goog_pos['tfid'].idf_
pipe_goog_pos.transform(corpus_goog_pos).shape

#********************* GOOGL **************************
df_googl_idx = tweet_head.ticker_symbol_group.apply(lambda a: 'GOOGL' in a)
df_googl= tweet_head[df_googl_idx]

#NEGATIVE = 0
df_googl_negative = df_googl[df_googl['sentiment_code']==0]
### JOIN ALL TWEETS NEGATIVE
df_googl_negative_alltweets  = df_googl_negative['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_googl_negative = [df_googl_negative_alltweets]
vocabulary_googl_negative = np.unique(df_googl_negative_alltweets.split(" "))
pipe_googl_nega = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_googl_negative)),
                  ('tfid', TfidfTransformer())]).fit(corpus_googl_negative)
pipe_googl_nega['count'].transform(corpus_googl_negative).toarray()
pipe_googl_nega['tfid'].idf_
pipe_googl_nega.transform(corpus_googl_negative).shape

#NEUTRAL = 1
df_googl_neutral = df_googl[df_googl['sentiment_code']==1]
### JOIN ALL TWEETS NEUTRAL
df_googl_neutral_alltweets  = df_googl_neutral['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_googl_neutral = [df_googl_neutral_alltweets]
vocabulary_googl_neutral = np.unique(df_googl_neutral_alltweets.split(" "))
pipe_googl_neu = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_googl_neutral)),
                  ('tfid', TfidfTransformer())]).fit(corpus_googl_neutral)
pipe_googl_neu['count'].transform(corpus_googl_neutral).toarray()
pipe_googl_neu['tfid'].idf_
pipe_googl_neu.transform(corpus_googl_neutral).shape

#POSITIVE = 2
df_googl_positive = df_googl[df_googl['sentiment_code']==2]
### JOIN ALL TWEETS POSITIVE
df_googl_positive_alltweets  = df_googl_positive['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_googl_pos = [df_googl_positive_alltweets]
vocabulary_googl_pos = np.unique(df_googl_positive_alltweets.split(" "))
pipe_googl_pos = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_googl_pos)),
                  ('tfid', TfidfTransformer())]).fit(corpus_googl_pos)
pipe_googl_pos['count'].transform(corpus_googl_pos).toarray()
pipe_googl_pos['tfid'].idf_
pipe_googl_pos.transform(corpus_googl_pos).shape

#********************* MSFT **************************
df_msft_idx = tweet_head.ticker_symbol_group.apply(lambda a: 'MSFT' in a)
df_msft = tweet_head[df_msft_idx]

#NEGATIVE = 0
df_msft_negative = df_msft[df_msft['sentiment_code']==0]
### JOIN ALL TWEETS NEGATIVE
df_msft_negative_alltweets  = df_msft_negative['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_msft_negative = [df_msft_negative_alltweets]
vocabulary_msft_negative = np.unique(df_msft_negative_alltweets.split(" "))
pipe_msft_nega = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_msft_negative)),
                  ('tfid', TfidfTransformer())]).fit(corpus_msft_negative)
pipe_msft_nega['count'].transform(corpus_msft_negative).toarray()
pipe_msft_nega['tfid'].idf_
pipe_msft_nega.transform(corpus_msft_negative).shape

#NEUTRAL = 1
df_msft_neutral = df_msft[df_msft['sentiment_code']==1]
### JOIN ALL TWEETS NEUTRAL
df_msft_neutral_alltweets  = df_msft_neutral['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_msft_neutral = [df_msft_neutral_alltweets]
vocabulary_msft_neutral = np.unique(df_msft_neutral_alltweets.split(" "))
pipe_msft_neu = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_msft_neutral)),
                  ('tfid', TfidfTransformer())]).fit(corpus_msft_neutral)
pipe_msft_neu['count'].transform(corpus_msft_neutral).toarray()
pipe_msft_neu['tfid'].idf_
pipe_msft_neu.transform(corpus_msft_neutral).shape

#POSITIVE = 2
df_msft_positive = df_msft[df_msft['sentiment_code']==2]
### JOIN ALL TWEETS POSITIVE
df_msft_positive_alltweets  = df_msft_positive['clean_tweet'].str.cat(sep = ' ')
#EMBEDDINGS
corpus_msft_pos = [df_msft_positive_alltweets]
vocabulary_msft_pos = np.unique(df_msft_positive_alltweets.split(" "))
pipe_msft_pos = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_msft_pos)),
                  ('tfid', TfidfTransformer())]).fit(corpus_msft_pos)
pipe_msft_pos['count'].transform(corpus_msft_pos).toarray()
pipe_msft_pos['tfid'].idf_
pipe_msft_pos.transform(corpus_msft_pos).shape



