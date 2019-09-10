# Importing the libraries
import numpy as np
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('restaurant_review.tsv', delimiter = '\t', quoting = 3)
dataset.dropna(inplace=True)
dataset.isnull().sum()
print(dataset['Liked'].value_counts())
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
corpus = []
for i in range(0, 1593):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    S_stemmer = SnowballStemmer(language='english')
    review = [S_stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
# Creating the Bag of Words model using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1600,ngram_range=(1, 1))
X = cv.fit_transform(corpus).toarray()
feature=cv.get_feature_names()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test)
# Accuracy, Precision and Recall
score1 = accuracy_score(y_test,y_pred1)
print("\n")
print("Multinomial NB Accuracy is ",round(score1*100,2),"%","\n")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)
print ("Confusion Matrix:\n",cm)

#  Precision and Recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score12 = precision_score(y_test,y_pred1)
score13= recall_score(y_test,y_pred1)
print("\n")
print("Precision is ",round(score12,2))
print("Recall is ",round(score13,2))


# Bernoulli NB
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB(alpha=0.8)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred2 = classifier.predict(X_test)
score21 = accuracy_score(y_test,y_pred2)

print("Bernoulli NB Accuracy is ",round(score21*100,2),"%","\n")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score22 = precision_score(y_test,y_pred2)
score23= recall_score(y_test,y_pred2)
print("\n")
print("Precision is ",round(score22,2))
print("Recall is ",round(score23,2))


# Logistic Regression
# Fitting Logistic Regression to the Training set
from sklearn import linear_model
classifier1 = linear_model.LogisticRegression(C=1.5)
classifier1.fit(X_train, y_train)

# Predicting the Test set results
y_pred3 = classifier1.predict(X_test)

# Accuracy
score31 = accuracy_score(y_test,y_pred3)
print("Logistic Regression Accuracy is: ",round(score31*100,2),"%","\n")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred3)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score32 = precision_score(y_test,y_pred3)
score33= recall_score(y_test,y_pred3)
print("\n")
print("Precision is ",round(score32,2))
print("Recall is ",round(score33,2))


#SVM
from sklearn.svm import LinearSVC
classifier = LinearSVC()
classifier.fit(X_train, y_train)
y_pred4 = classifier.predict(X_test)
# Accuracy
score41 = accuracy_score(y_test,y_pred4)
print("Support vector machine Accuracy is ",round(score41*100,2),"%","\n")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred4)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score42 = precision_score(y_test,y_pred4)
score43= recall_score(y_test,y_pred4)
print("\n")
print("Precision is ",round(score42,2))
print("Recall is ",round(score43,2))
#KNN
from sklearn.neighbors import KNeighborsClassifier
neighbors = KNeighborsClassifier(n_neighbors=5)
neighbors.fit(X_train, y_train)
y_pred5 = neighbors.predict(X_test)

score51=accuracy_score(y_test, y_pred5)
print('KNeighborsClassifier Accuracy:', round(score51*100,2),"%","\n")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred5)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score52 = precision_score(y_test,y_pred5)
score53= recall_score(y_test,y_pred5)
print("\n")
print("Precision is ",round(score52,2))
print("Recall is ",round(score53,2))

#decission Tree

classifier= DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred6=classifier.predict(X_test)

score61=accuracy_score(y_test, y_pred6)
print('DecisionTreeClassifier Accuracy: ', round(score61*100,2),"%","\n")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred6)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score62 = precision_score(y_test,y_pred6)
score63= recall_score(y_test,y_pred6)
print("\n")
print("Precision is ",round(score62,2))
print("Recall is ",round(score63,2))
#from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier())
model.fit(X_train, y_train)
y_pred7=model.predict(X_test)
score71=accuracy_score(y_test, y_pred7)
print('BaggingClassifier Accuracy:', round(score71*100,2),"%","\n")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred7)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score72 = precision_score(y_test,y_pred7)
score73= recall_score(y_test,y_pred3)
print("\n")
print("Precision is ",round(score72,2))
print("Recall is ",round(score73,2))

##AdaBoostClassifier

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(X_train, y_train)
y_pred8=model.predict(X_test)
score81=accuracy_score(y_test, y_pred8)
print('AdaBoostClassifier Accuracy: ', round(score81*100,2),"%","\n")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred8)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score82 = precision_score(y_test,y_pred8)
score83= recall_score(y_test,y_pred8)
print("\n")
print("Precision is ",round(score82,2))
print("Recall is ",round(score83,2))

models = [
    MultinomialNB(),
    LogisticRegression(C=1),
    LinearSVC(C=1),
    DecisionTreeClassifier(),
    BernoulliNB(alpha=0.8),
    KNeighborsClassifier(n_neighbors=5)

]

m_names = [m.__class__.__name__ for m in models]

models = list(zip(m_names, models))
final_model = VotingClassifier(estimators=models)

accs = []

final_model.fit(X_train, y_train)   
y_pred9 = final_model.predict(X_test)
accs.append(accuracy_score(y_test, y_pred9))
print("Voting Classifier")
print("-" * 30)
print("Avg. Accuracy: {:.2f}%".format(sum(accs) / len(accs) * 100))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred9)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score92 = precision_score(y_test,y_pred9)
score93= recall_score(y_test,y_pred9)
print("\n")
print("Precision is ",round(score92,2))
print("Recall is ",round(score93,2))


###############################

feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), classifier1.coef_[0]
    )
}
print ("Best_positive word : ")
for best_positive in sorted( feature_to_coef.items(),  key=lambda x: x[1], reverse=True)[:15]:
  print (best_positive)
print ("Best_negative word : ")  
for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1])[:15]:
  print (best_negative)





	 