import re
import tweepy
import pandas as pd
from datetime import datetime
from textblob import TextBlob
from search_twitter_query import to_data_frame, limit_handled
import time

# Dhruv's Twitter API authentication
# consumer_key = 'QyPohWOnU5rWBj0p8eitISxZm'
# consumer_secret = 'Bic1EnxYGOnaZZaNiSz0xW0Kb3Y4RGVhqbfriVP3dgm2xdj2Ln'
# access_token = '383059399-Nv6Z94gW7fMReiyKLGKANTFYay14tFvrB1Ut8c9s'
# access_token_secret = 'BsM3FhhEkkBYe7q1LPTAHQa1HyZKmhjVkaxdA5hJefwGy'

# Yusuf's Twitter API authentication
# consumer_key = 'GyfyFJEkU6cyGBq0PPLjHlvz0'
# consumer_secret = 'q3ghkBA8i1qheGFFnpd5mmCmlAlrNIk02wqTqeoQ2gERHiwqLw'
# access_token = '855727868-h0MenCCakLLaz6engeaIm2mh77j3uoOnN5DIXV07'
# access_token_secret = 'c4TWPLTCmdx8ijhXS3gkH59Wcv8PGJ8BUFDTFXfT6hMiS'

# Matt's Twitter API authentication
consumer_key = '1iQ8rHkTabqFSEaIujAHqahTW'
consumer_secret = 'KxnRmci0bkWpoeFF6lSyR8lzlNmutwDIaMZ6vkXF5moh68vS6v'
access_token = '701880260678148096-GkJ6CXuCFLjQYKnQdYl13QZI0Lk49KU'
access_token_secret = 'N5rwNAW7Su20FEZW741R0T992vg2xGwfqBNZzNC5FdMal'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# test_tweet = TextBlob("'@Apple sucks!!', really? #iPhoneX")
# print(test_tweet.sentiment.subjectivity)
# print(test_tweet.sentiment.polarity)

# Create an empty list to store our tweets
tweets = []

# Search for tweets matching our query, and add them to our list
for tweet in limit_handled(tweepy.Cursor(api.search, q='@apple', lang='en', since='2017-11-26', until='2017-12-27').items(500)):
    tweets.append(tweet)

# Convert our list of tweets into a Pandas data frame
tweet_data_frame = to_data_frame(tweets)

print(tweet_data_frame)