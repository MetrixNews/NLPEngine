import csv
import codecs
import SqlHandler
import operator
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd

nltk.download('stopwords')

analyser = SentimentIntensityAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')

vectorizer = TfidfVectorizer(lowercase=True, analyzer='word', ngram_range=(1, 3), min_df=0.05, max_features=15000)

lexicon_words = []

x_fit_transform = []

articles_df = pd.DataFrame()

def parse_key_words(articles_text):
   global tfidf_dataframe
   try:
      x = vectorizer.fit_transform(articles_text)

      print(x)

      tfidf_dataframe = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())

      print(tfidf_dataframe)

   except ValueError:
      print('Transforming process skipped')

def sentiment_analyzer_score(text):
   return analyser.polarity_scores(text)

def parse_vader_score(score):
   com = score['compound'] #Compound
   
   if com >= 0.5:
      bias = 'pos'
   elif com > -0.5 and com < 0.5:
      bias = 'neu'
   else:
      bias = 'neg'

   working_article['vader_pos'] = score['pos']
   working_article['vader_neu'] = score['neu']
   working_article['vader_neg'] = score['neg']
   working_article['biasness'] = bias

def write_articles_to_csv(result):
   result.to_csv('data/articles.csv')

def get_field_names():
   fields = ['id', 'source_name', 'title', 'url', 'published', 'vader_pos', 'vader_neu', 'vader_neg',
   'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'topic', 'biasness', 'political_bias', 'topic']

   return fields

def get_lexicon_words():
   cursor = SqlHandler.my_db.cursor()
   cursor.execute("SELECT * FROM NRC")

   for row in cursor.fetchall():
      nrc = {"word":row[1], "anger":row[2], "anticipation":row[3], "disgust":row[4], "fear":row[5], "joy":row[6], "negative":row[7], 
             "positive":row[8], "sadness":row[9], "surprise":row[10], "trust":row[11]}

      lexicon_words.append(nrc)

def get_article_emotion(text):
   anger = 0
   anticipation = 0
   disgust = 0
   fear = 0
   joy = 0
   #negative = 0
   #positive = 0
   sadness = 0
   surprise = 0
   trust = 0

   #store scores for each emotion for each article in database

   words = tokenizer.tokenize(text)

   for word in words:
      nrc = next((lex for lex in lexicon_words if lex["word"] == word), None)

      if nrc != None:
         anger = anger + nrc["anger"]
         anticipation = anticipation + nrc["anticipation"]
         disgust = disgust + nrc["disgust"]
         fear = fear + nrc["fear"]
         joy = joy + nrc["joy"]
         #negative = negative + nrc["negative"]
         #positive = positive + nrc["positive"]
         sadness = sadness + nrc["sadness"]
         surprise = surprise + nrc["surprise"]
         trust = trust + nrc["trust"]
   
   emotions = {'anger':anger, 'anticipation':anticipation, 'disgust':disgust, 'fear':fear, 'joy':joy, 
   'sadness':sadness, 'surprise':surprise, 'trust':trust}

   working_article['anger'] = anger
   working_article['anticipation'] = anticipation
   working_article['disgust'] = disgust
   working_article['fear'] = fear
   working_article['joy'] = joy
   working_article['sadness'] = sadness
   working_article['surprise'] = surprise
   working_article['trust'] = trust

   result = max(emotions.items(), key=operator.itemgetter(1))[0]

   return result

def main():
   global working_article

   # CREATE NEWS CORPUS FOR TRAINING
   news_corpus = []
   articles_text = []

   stop_words = set(stopwords.words('english'))

   # with open() as csvfile:
   csv.field_size_limit(100000000)

   with open('../data/news_corpus.txt', 'r') as f:
      reader = csv.reader((line.replace('\0', '') for line in f), delimiter='\t')
      try:
         print("Scraping CSV...", end="", flush=True)
         for row in reader:
            #parse full text to pare minumum
            filtered_article = []
            tokenized = tokenizer.tokenize(str(row[5]).lower())

            for w in tokenized:
               if w not in stop_words:
                  filtered_article.append(w)

            #filtered_article = [w for w in tokenized if not w in stop_words] 
            filtered_article = ' '.join(map(str, filtered_article))

            #write contents to global array
            news_corpus.append((row[0], row[1], row[2], row[3], row[4], filtered_article, row[6], row[7]))
            
      except csv.Error as e:
         print(e)
   
   print("Done", flush=True)
   # Get lexicon words from database

   print("Getting Lexicon Words...", end="", flush=True)
   get_lexicon_words()
   print("Done", flush=True)
   print("Done")

   print("Creating Articles Dataframe...")
   articles_df = pd.DataFrame(columns = get_field_names())
   print("Done")

   #loop and process each article
   for a in news_corpus:
      if(len(a[5]) == 0):
         continue

      print("Processing Article ID " + str(int(a[0]) - 1) + "...", end="", flush=True)

      #reset working article
      working_article = {}

      #get article text
      text = a[5]
      articles_text.append(text)

      working_article['id'] = str(int(a[0]) - 1)
      working_article['source_name'] = a[1]
      working_article['title'] = a[2]
      working_article['url'] = a[3]
      working_article['published'] = a[4]
      working_article['topic'] = a[6]
      working_article['political_bias'] = a[7]

      #parse_key_words(text)
      score = sentiment_analyzer_score(text)
      parse_vader_score(score)
      get_article_emotion(text)

      articles_df = articles_df.append(working_article, ignore_index=True)

      print("Done", flush=True),

   print("Parsing Key Words...", end="", flush=True)
   parse_key_words(articles_text)
   print("Done", flush=True)
   print("Merging Dataframes", end="", flush=True)
   result = articles_df.merge(tfidf_dataframe, how='outer', left_index=True, right_index=True)
   print("Done", flush=True)
   #write all articles and header to CSV for processing
   print("Writing to CSV", end="", flush=True)
   write_articles_to_csv(result)
   print("Done", flush=True)

if __name__ == "__main__":
   main()

# VADER lEXICON(https: //github.com/nltk/nltk/blob/develop/nltk/sentiment/vader.py) --> USE TO PULL POS, NEG, NEU SENTIMENT SCORES

#NRC EMOTION lEXICON(http://saifmohammad.com/WebPages/AccessResource.htm) --> USE TO PULL EMOTION SCORES

#TF - IDF-- > SCIKIT - LEARN(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) --> USE TO PULL IMPORTANT WORDS AS FEATURES