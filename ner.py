import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.simplefilter("ignore")

f1=open('modified_train.txt','r')
sentences=list()
tags=list()
entity=list()
sentence=''
tag=list()
ent=list()
B=0
I=0
O=0
tokens=list()
words=list()
ner=list()
for line in f1.readlines():
	l=line.split()
	if l==[]:
		sentences.append(sentence)
		sentence=''
		tags.append(tag)
		tag=[]
		entity.append(ent)
		ent=[]
	elif len(l)==4:
		continue
	else:
		sentence+=(l[0].lower())
		sentence+=' '
		tokens.append(l[0].lower())
		words.append(l[0].lower())
		#tag.append(l[1].split('-')[0])

		"""Assigning unique identifiers for the tags. 1 for B, 2 for I and 3 for O. Also counting the number of B,I,O tags """
		if l[1].split('-')[0]=='B':
			B+=1
			tag.append(1)
			ner.append(1)
		elif l[1].split('-')[0]=='I':
			I+=1
			tag.append(2)
			ner.append(2)
		elif l[1].split('-')[0]=='O':
			O+=1
			tag.append(3)
			ner.append(3)
		if len(l[1].split('-'))==2:
			ent.append(l[1].split('-')[1])
		else:
			ent.append('NIL')

f2=open('modified_test.txt','r')
test_words=list()
test_tags=list()
sentence1=list()
tag1=list()
add=True
for line in f2.readlines():
	l=line.split()
	if l==[]:
		if add:
			for i in range(len(sentence1)):
				test_words.append(sentence1[i])
				test_tags.append(tag1[i])
		sentence1=list()
		tag1=list()
	else:
		if l[0] not in words: 
			sentence1.append(l[0])
			if l[1].split('-')[0]=='B':
				tag1.append(1)
			elif l[1].split('-')[0]=='I':
				tag1.append(2)
			else:
				tag1.append(3)

sentences=sentences[1:]
tags=tags[1:]
entity=entity[1:]
#words=words[1:]
#ner=ner[1:]
tokens=list(set(tokens))

print("Number of sentences = ",len(sentences))
print("Number of B tags = ",B)
print("Number of I tags = ",I)
print("Number of O tags = ",O)

print("Number of unique tokens = ",len(tokens))

"""Extracting lemmas and POS tags for all the tokens in the dataset"""

from nltk.stem import WordNetLemmatizer

lemma=WordNetLemmatizer()
pos=list()
for i in range(len(sentences)):
	modified=''
	original=sentences[i].split()
	for s in original:
		modified+=lemma.lemmatize(s)
		modified+=' '
	sentences[i]=modified
	pos.append(nltk.pos_tag(nltk.word_tokenize(sentences[i])))

for i in range(len(tokens)):
	tokens[i]=lemma.lemmatize(tokens[i])

for i in range(len(words)):
	words[i]=lemma.lemmatize(words[i])

for i in range(len(test_words)):
	test_words[i]=lemma.lemmatize(test_words[i])

""" One hot encoding of the lemma and POS tags"""

input_lemmas=list()
for s in sentences:
	s1=s.split()
	temp=list()
	for x in s1:
		temp=[0]*len(tokens)
		temp[tokens.index(x)]=1
	input_lemmas.append(temp)

used_pos=list()

for p in pos:
	for word,ptag in p:
		if ptag not in used_pos:
			used_pos.append(ptag)

print("Number of POS tags used : ",len(used_pos))

input_pos=list()
for s in pos:
	sent=list()
	for word,ptag in s:
		temp=[0]*len(used_pos)
		temp[used_pos.index(ptag)]=1
		sent.append(temp)
	input_pos.append(sent)


""" Create model for NER """

import pandas as pd
from sklearn.naive_bayes import MultinomialNB

#X_train=pd.DataFrame({'Sentence' : input_lemmas, 'POS' : input_pos, 'NER' : tags})
X_train=pd.DataFrame({'Words' : words, 'Tag' : ner})
vect=TfidfVectorizer(max_features=50000, analyzer='word', ngram_range=(1, 1))
X=vect.fit_transform(X_train.Words)
model=MultinomialNB()
model.fit(X,X_train['Tag'])

""" Predict and checking for accuracy """

from sklearn.metrics import precision_recall_fscore_support
X_test=pd.DataFrame({'Words' : test_words, 'Tag' : test_tags})
X_t=vect.transform(X_test.Words)
pred=model.predict(X_t)
(precision,recall,fscore,support)=precision_recall_fscore_support(X_test['Tag'],pred,average="micro")
print("\n\nPrecision = ",precision,"\nRecall = ",recall,"\nF-Score = ",fscore)