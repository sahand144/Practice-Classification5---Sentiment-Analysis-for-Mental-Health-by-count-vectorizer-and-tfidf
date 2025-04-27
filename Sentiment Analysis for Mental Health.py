#Preprocessing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from string import punctuation
#ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Loading the dataset
path = r"D:\datasets\New To Work on 3\Sentiment Analysis for Mental Health\Sentiment Analysis for Mental Health.csv"
df = pd.read_csv(path)

#EDA Part
print(df.head())
dataset = df[['statement','status']]
print(dataset.head())
print(dataset.isna().sum())
print(dataset.info())
dataset.dropna(inplace=True)
#check if we have null value
print(dataset.isna().sum())
#Lets see if our target is balanced or not with visualization the result
print(dataset["status"].value_counts())
sns.countplot(data=dataset , x="status" , color='green' , stat='percent')
plt.show()
#the result show there is imbalance in target column , 
# if we did not get th result in the end we may use smote or under sampling for getting bettr model performance

#make sure no empty string in statement
dataset = dataset[dataset['statement'].str.strip() != '']

#now lets preprocess the text
import nltk

# Force download to your directory
nltk.download('punkt', download_dir='F:/Users/ELECOMP/nltk_data')
# Tell nltk to look into this directory
nltk.data.path.append('F:/Users/ELECOMP/nltk_data')

nltk.download('stopwords', download_dir='F:/Users/ELECOMP/nltk_data')
nltk.data.path.append('F:/Users/ELECOMP/nltk_data')


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return ' '.join(tokens)

#Apply preprocessing to statement column
dataset['Preprocessed_Statement'] = dataset['statement'].apply(preprocess_text)
print(dataset[['Preprocessed_Statement','statement']].head())

#Encode the Label column
encoder = LabelEncoder()
dataset['Encoded_Status'] = encoder.fit_transform(dataset['status'])
dataset['Encoded_Status'].sample(10)

# Display the mapping of labels to numbers
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Label Mapping:", label_mapping)

X = dataset['Preprocessed_Statement']
y = dataset['Encoded_Status']

#Split the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42 , stratify=y)
#stratify=y ensures the class distribution in the training and testing sets matches the original dataset, which is important for imbalanced datasets.
print(f'X_train.shape,X_test.shape,y_train.shape,y_test.shape : {X_train.shape,X_test.shape,y_train.shape,y_test.shape}')

#create or feature matrix with count vectorizer
count_vectorizer = CountVectorizer(ngram_range=(1,2),max_features=5000)
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

print("CountVectorizer Feature Matrix Shape (Train):", X_train_count.shape)
print("CountVectorizer Feature Matrix Shape (Test):", X_test_count.shape)
#we can see that the shape of the feature matrix is (n_samples, n_features) and we have 5000 features
# max_features=5000: Limits the vocabulary to the top 5,000 most frequent words to reduce dimensionality.
# ngram_range=(1, 2): Includes unigrams (single words) and bigrams (two-word phrases) to capture some context.

# create our feature matrix with tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("TfidfVectorizer Feature Matrix Shape (Train):", X_train_tfidf.shape)
print("TfidfVectorizer Feature Matrix Shape (Test):", X_test_tfidf.shape)
#we can see that the shape of the feature matrix is (n_samples, n_features) and we have 5000 features

# now lets create a function to train and evaluate the model and return the accuracy score
def train_and_evaluate_model(model,X_train,y_train,X_test,y_test, model_name,vectorizer_name):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    #Calculate the accuracy score
    accuracy = accuracy_score(y_test,y_pred)
    # Print results
    print(f"\n{model_name} with {vectorizer_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    #use heatmap to visualize the confusion matrix
    cm = confusion_matrix(y_test,y_pred)
    sns.heatmap(cm,annot=True,fmt='d',cmap='coolwarm',xticklabels=encoder.classes_,yticklabels=encoder.classes_)
    plt.show()
    return accuracy

#try models single by single
train_and_evaluate_model(LogisticRegression(max_iter=500),X_train_count,y_train,X_test_count,y_test,'Logistic Regression','Count Vectorizer')
train_and_evaluate_model(MultinomialNB(),X_train_count,y_train,X_test_count,y_test,'MultinomialNB','Count Vectorizer')
train_and_evaluate_model(DecisionTreeClassifier(),X_train_count,y_train,X_test_count,y_test,'Decision Tree','Count Vectorizer')
train_and_evaluate_model(RandomForestClassifier(n_estimators=100,random_state=42),X_train_count,y_train,X_test_count,y_test,'Random Forest','Count Vectorizer')

#now lets train with different models and see how they perform
models = [
    (LogisticRegression(max_iter=500) , 'Logistic Regression'),
    (MultinomialNB() , 'MultinomialNB'),
    (DecisionTreeClassifier() , 'Decision Tree'),
    (RandomForestClassifier(n_estimators=100,random_state=42) , 'Random Forest')
]

# Train and evaluate with CountVectorizer
print("Results with CountVectorizer:")
#train and evaluate the model
for model,model_name in models:
    train_and_evaluate_model(model,X_train_count,y_train,X_test_count,y_test,model_name,'Count Vectorizer')

# Train and evaluate with TfidfVectorizer
print("\nResults with TfidfVectorizer:")
for model, model_name in models:
    train_and_evaluate_model(model, X_train_tfidf, X_test_tfidf, y_train, y_test, model_name, "TfidfVectorizer")
    
#lets try to improve the model performance by using grid search cv
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

# Initialize GridSearchCV with logistic regression model
grid_search = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5, scoring='accuracy')

# Train and evaluate with CountVectorizer
print("Results with CountVectorizer:")
grid_search.fit(X_train_count, y_train)

# Print the best parameters and best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")

# Train and evaluate with TfidfVectorizer
print("\nResults with TfidfVectorizer:")
grid_search.fit(X_train_tfidf, y_train)

# Print the best parameters and best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")


# Define the parameter grid for MultinomialNB
param_grid = {
    'alpha': [0.01, 0.1, 1, 10],
    'fit_prior': [True, False]
}

# Initialize GridSearchCV with MultinomialNB
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')

# Train and evaluate with CountVectorizer
print("Results with CountVectorizer:")
grid_search.fit(X_train_count, y_train)

# Print the best parameters and best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")

# Train and evaluate with TfidfVectorizer
print("\nResults with TfidfVectorizer:")
grid_search.fit(X_train_tfidf, y_train)

# Print the best parameters and best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")
