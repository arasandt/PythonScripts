import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.stem.lancaster import LancasterStemmer
from nltk.wsd import lesk
from nltk.probability import FreqDist
from string import punctuation
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
text = 'Mary had a little lamb. Her fleece was white as snow. Lamb little '
sents = sent_tokenize(text)
#print(sents)
#words = word_tokenize(text) 
words = [word_tokenize(t) for t in sents]
#print(words)
customstopwords = set(stopwords.words('english') + list(punctuation))
#print(customstopwords)
wordsstop = [word for word in word_tokenize(text) if word not in customstopwords]
#print(wordsstop )
bm = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(wordsstop) # can do trigrams too.
#print(sorted(finder.ngram_fd.items()))

text2 = 'Mary closed closer in close'
st = LancasterStemmer() # reduces to root form.
stemw = [st.stem(i) for i in word_tokenize(text2)]
#print(set(stemw))
#print(nltk.pos_tag(word_tokenize(text2))) # part of speech tagging

for ss in wordnet.synsets('bass'):
    pass #print(ss,ss.definition())

sense1 = lesk(word_tokenize("Sing in a lower tone, along with the bass"),'bass')
#print(sense1,sense1.definition())

sense1 = lesk(word_tokenize("This sea bass was really hard"),'bass')
#print(sense1,sense1.definition())

freq = FreqDist(wordsstop)
print(freq.items())
