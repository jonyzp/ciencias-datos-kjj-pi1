# %% [markdown]
# # Raw -> Trusted

# %% [markdown]
# ### Imports and Instances

# %%
# -*- coding: utf-8 -*-

# Import common libraries:
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job


# %%
# Import specific libraries:
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import nltk
# nltk.download('punkt') # word_tokenize
# nltk.download('stopwords') # word_tokenize
# nltk.download('words') # WordNetLemmatizer
# nltk.download('wordnet') # WordNetLemmatizer
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk.tokenize import TweetTokenizer
# tt = TweetTokenizer()


# %%

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

# from pyspark.sql import SparkSession
# spark = SparkSession.builder.master("local[1]") \
#                     .appName('SparkByExamples.com') \
#                     .getOrCreate()

# sc = SparkContext.getOrCreate()
# glueContext = GlueContext(sc)
# spark = glueContext.spark_session
# job = Job(glueContext)
# job.init(args['JOB_NAME'], args)

# %% [markdown]
# ### Read data from Glue Catalog

# %%
## @type: DataSource
## @args: [database = "pi1-kjj-trusted", table_name = "company1_csv", transformation_ctx = "DataSource0"]
## @return: company_dynamicframe
## @inputs: []
# company_dynamicframe = glueContext.create_dynamic_frame.from_catalog(
#        database = "pi1-kjj-trusted",
#        table_name = "company_csv",
#        transformation_ctx = "company_dynamicframe")
company_dynamicframe = spark.read.csv('raw/Company.csv', header='true', inferSchema='true')
company_dynamicframe.printSchema()

# company_tweet_dynamicframe = glueContext.create_dynamic_frame.from_catalog(
#        database = "pi1-kjj-trusted",
#        table_name = "company_tweet_csv")
company_tweet_dynamicframe = spark.read.csv('raw/Company_Tweet.csv', header='true', inferSchema='true')
company_tweet_dynamicframe.printSchema()

# tweet_dynamicframe = glueContext.create_dynamic_frame.from_catalog(
#        database = "pi1-kjj-trusted",
#        table_name = "tweet_csv")
tweet_dynamicframe = spark.read.csv('raw/Tweet.csv', header='true', inferSchema='true')
tweet_dynamicframe.printSchema()

# company_dataframe = company_dynamicframe.toDF()
# tweet_dataframe = tweet_dynamicframe.toDF()
# company_tweet_dataframe = company_tweet_dynamicframe.toDF()

company_dataframe = company_dynamicframe
tweet_dataframe = tweet_dynamicframe
company_tweet_dataframe = company_tweet_dynamicframe


# %% [markdown]
# ### Summarize total engagement:

# %%

# Summarize total engagement:
# tweet_dataframe['total_engagement'] = tweet_dataframe['comment_num'] + tweet_dataframe['retweet_num'] + tweet_dataframe['like_num']
tweet_dataframe = tweet_dataframe.withColumn('total_engagement', sum(tweet_dataframe[col] for col in ['comment_num', 'retweet_num', 'like_num']))

# Get iso date (Not epoch)
tweet_dataframe = tweet_dataframe.withColumnRenamed('post_date', 'post_datetime_epoch')
# tweet_dataframe['post_datetime'] = pd.to_datetime(tweet_dataframe['post_datetime_epoch'],unit='s')
tweet_dataframe.head(2)

# %% [markdown]
# ### Get only date from datetime

# %%

from pyspark.sql import functions as f
# https://stackoverflow.com/a/54340652
tweet_dataframe = tweet_dataframe.withColumn('post_datetime', f.to_timestamp(tweet_dataframe.post_datetime_epoch))
# tweet_dataframe['post_date'] = tweet_dataframe['post_datetime'].dt.date
tweet_dataframe = tweet_dataframe.withColumn('post_date', f.to_date(tweet_dataframe.post_datetime))
#tweet_dataframe = tweet_dataframe.withColumn('post_date', f.to_date(f.col('post_datetime')))


# %% [markdown]
# ### Get ticker symbol for each tweet

# %%
# Join company tweets with tweets dataframe:
# ticker_symbol_group = company_tweet_dataframe.groupBy('tweet_id')['ticker_symbol'].distinct()
# ticker_symbol_group.rename("ticker_symbol_group", inplace=True)
ticker_symbol_group = company_tweet_dataframe.groupBy('tweet_id').agg(
    f.collect_set(f.col('ticker_symbol')).alias('ticker_symbol_group')
)
# print(ticker_symbol_group.show(n=5))

tweet_dataframe = tweet_dataframe.join(ticker_symbol_group, tweet_dataframe['tweet_id'] == ticker_symbol_group['tweet_id'], 'left')
# print(tweet_dataframe.show(truncate=False, n=2))


# %% [markdown]
# ### Tokenize & Lemmatize

# %%
from pyspark.ml.feature import NGram, Tokenizer, StopWordsRemover
from pyspark.ml import Pipeline

# %%
#tweet_head['body_tokenized'] = tweet_head.apply(lambda row: nltk.word_tokenize(str(row['body'])), axis=1)
#tweet_dataframe = tweet_dataframe.withColumn('body_tokenized', nltk.word_tokenize(str(col)) for col in ['body'])
# tweet_dataframe = tweet_dataframe.withColumn('body_tokenized', tweet_dataframe.select('body').rdd.map(lambda x: 
#     TweetTokenizer(str(x))).collect()
# )
# print(tweet_dataframe.show(n=5))

# stop_words_nltk = set(stopwords.words('english'))

# https://towardsdatascience.com/sentiment-analysis-with-pyspark-bc8e83f80c35

tokenizer = Tokenizer(inputCol="body", outputCol="body_tokenized")
remove_stop_words = StopWordsRemover(inputCol="body_tokenized", outputCol="no_stopwords")
unigrammer = NGram(n=1, inputCol="no_stopwords", outputCol="ngrams") 


data_prep_pipe = Pipeline(
    stages=[tokenizer, remove_stop_words, unigrammer]
)


transformed = data_prep_pipe.fit(tweet_dataframe).transform(tweet_dataframe)

# %%
#transformed.select(['body', 'ngrams']).show()

transformed.printSchema()


# %%
from pyspark.sql.functions import concat_ws
transformed = transformed.withColumn('bow', concat_ws(' ', 'ngrams'))

transformed.select('bow').show()

# %% [markdown]
# ### Sentiment Analysis

# %%
import nltk
from pyspark.sql.functions import udf
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# %%
sentiment = udf(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
spark.udf.register('sentiment', sentiment)
transformed = transformed.withColumn('score_vader', sentiment('bow').cast('double'))

# %%
transformed.select('score_vader').show()

# %%
# conditions = [
#     (tweet["score_vader"] >= .05),
#     (tweet["score_vader"] > -.05) & (tweet["score_vader"] < .05),
#     (tweet["score_vader"] <= -.05),
# ]

# choices = ['positive', 'neutral', 'negative']
# tweet['sentiment'] = np.select(conditions, choices)

@udf
def scorevader_classifier(s):
  if (s['score_vader'] >= 0.05): return 'positive'
  elif s['score_vader'] <= -0.05: return 'neutral'
  else: return 'negative'

transformed = transformed.withColumn('sentiment', scorevader_classifier('score_vader').cast('double'))

# %% [markdown]
# ### Split data by ticker symbol

# %%
#transformed.show(n=2)
#transformed.rdd.takeSample(False, 1, seed=0)
transformed.sample(False, 0.1, seed=0).limit(1)


# %%
company_dataframe.show(n=6)

# %%
# def split_data(x, v_split):
#     #columna de listas de tickers
#     t=x['ticker_symbol_group']
#     #vector de boleanos
#     v_bool=t.apply(lambda tikers: np.isin(v_split,tikers))
#     #retornar datos filtrados
#     return x.loc[v_bool].copy()
from pyspark.sql.functions import array_contains, col

APPL = transformed \
  .filter(array_contains(col('ticker_symbol_group'), 'AAPL'))

APPL.show()

# %%
GOOG = transformed \
  .filter(array_contains(col('ticker_symbol_group'), 'GOOG'))

GOOG.show()


# %%


GOOGL = transformed \
  .filter(array_contains(col('ticker_symbol_group'), 'GOOGL'))

GOOGL.show()



# %%

AMZN = transformed \
  .filter(array_contains(col('ticker_symbol_group'), 'AMZN'))

AMZN.show()


# %%


TSLA = transformed \
  .filter(array_contains(col('ticker_symbol_group'), 'TSLA'))

TSLA.show()


# %%


MSFT = transformed \
  .filter(array_contains(col('ticker_symbol_group'), 'MSFT'))

MSFT.show()

# %% [markdown]
# ### Store to S3

# %%

# Export data to S3:
from awsglue.dynamicframe import DynamicFrame
tweet_dyf = DynamicFrame.fromDF(tweet_dataframe, glueContext, "tweet_parquet")

glueContext.write_dynamic_frame.from_options(
       frame = tweet_dyf,
       connection_type = "s3",
       connection_options = {"path": "s3://pi1-kjj/trusted/tweets"},
       format = "json")
print('End storing to S3')



# %%


# %%


# %%



