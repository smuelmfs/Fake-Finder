import nltk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
import re

nltk.download('stopwords')
nltk.download('wordnet')

# Lê o arquivo CSV em um DataFrame do pandas
df = pd.read_csv('WELFake_Dataset.csv')

# Verificação inicial do DataFrame
print("Número de linhas antes do pré-processamento:", df.shape[0])

# Verifica valores nulos no DataFrame
print("Valores nulos no DataFrame:")
print(df.isnull().sum())

# Remove linhas nulas e duplicadas
df = df.dropna().drop_duplicates()

# Verificação após remoção de valores nulos e duplicados
print("Número de linhas após remoção de nulos e duplicados:", df.shape[0])

# Contagem dos rótulos
label_count = df['label'].value_counts()

# Visualização da distribuição dos rótulos
plt.bar(label_count.index, label_count)
plt.title('Distribuição de rótulos')
plt.xlabel('Rótulo')
plt.ylabel('Número de ocorrências')
plt.xticks([0, 1], ['Fake', 'Real'])
plt.show()

# Exibição percentual dos rótulos
print("Percentual de rótulos Fake:", 100 * label_count[0] / len(df['label']))
print("Percentual de rótulos Real:", 100 * label_count[1] / len(df['label']))

# Concatenação dos campos 'title' e 'text' para criar uma nova coluna 'news'
df['news'] = df['title'] + ' ' + df['text']

# Pré-processamento de texto
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Aplica o pré-processamento na coluna 'news'
df['processed_news'] = df['news'].apply(preprocess_text)

# Separação em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['processed_news'], df['label'], test_size=0.2, random_state=42)

# Vetorização usando Bag-of-Words
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Criação e treinamento do modelo Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)

# Avaliação do modelo com classification_report
nb_predictions = nb_model.predict(X_test_vectorized)
print("Classification Report for Naive Bayes Model:")
print(classification_report(y_test, nb_predictions))

# Avaliação do modelo com métricas individuais
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print("Precision:", precision_score(y_test, nb_predictions, average='weighted'))
print("Recall:", recall_score(y_test, nb_predictions, average='weighted'))
print("F1 Score:", f1_score(y_test, nb_predictions, average='weighted'))

# Cross-validation
scoring = {'accuracy': 'accuracy',
           'precision': 'precision',
           'recall': 'recall',
           'f1_score': 'f1'}

cv_results = cross_validate(nb_model, X_train_vectorized, y_train, scoring=scoring, cv=5)
print("Cross-validation results for Naive Bayes Model:")
for metric_name, result in cv_results.items():
    print(f"{metric_name}: {result.mean()} (±{result.std()})")

'''# Inicialização e treinamento de outros modelos (SVM, Random Forest, Decision Tree)
svm_clf = SVC()
svm_clf.fit(X_train_vectorized, y_train)

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train_vectorized, y_train)

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train_vectorized, y_train)

# Avaliação dos outros modelos
print("Avaliação de outros modelos:")
print("SVM Accuracy:", svm_clf.score(X_test_vectorized, y_test))
print("Random Forest Accuracy:", rf_clf.score(X_test_vectorized, y_test))
print("Decision Tree Accuracy:", dt_clf.score(X_test_vectorized, y_test))'''

# Ajuste de hiperparâmetros para o Naive Bayes Multinomial
parameters = {'alpha': [0.5, 1.0, 1.5]}
nb_clf = MultinomialNB()
grid_search = GridSearchCV(nb_clf, parameters, cv=5, n_jobs=-1)
grid_search.fit(X_train_vectorized, y_train)

best_nb_clf = grid_search.best_estimator_
print("Best Naive Bayes Multinomial Accuracy:", best_nb_clf.score(X_test_vectorized, y_test))

# Exemplo de oversampling para balancear classes
oversample = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = oversample.fit_resample(X_train_vectorized, y_train)

# Define top_n como um valor inteiro
top_n = 20

# Obter as palavras mais importantes com base nas contagens no vetorizador
feature_names = vectorizer.get_feature_names_out()

# Obter os índices para as palavras mais importantes positivas e negativas
top_positive_indices = best_nb_clf.feature_log_prob_[1].argsort()[-top_n:][::-1]
top_negative_indices = best_nb_clf.feature_log_prob_[0].argsort()[-top_n:][::-1]

# Mostra as palavras mais importantes positivas e seus coeficientes
print("Top Palavras Positivas:")
print([(feature_names[i], best_nb_clf.feature_log_prob_[1][i]) for i in top_positive_indices])

# Mostra as palavras mais importantes negativas e seus coeficientes
print("Top Palavras Negativas:")
print([(feature_names[i], best_nb_clf.feature_log_prob_[0][i]) for i in top_negative_indices])
'''ATÉ AQUI O CÓDIGO FUNCIONA DEPOIS DE MOSTRAR O TOP 20 DE PALAVRAS ELE PARECE ENTRAR EM UM LOOP'''

'''# Inicialização e treinamento do modelo SVM com kernel linear
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_vectorized, y_train)

# Avaliação do modelo
svm_linear_accuracy = svm_linear.score(X_test_vectorized, y_test)
print("Accuracy with Linear SVM:", svm_linear_accuracy)'''

# Função para pré-processamento de uma única notícia
def preprocess_single_news(news):
    news = re.sub(r'\b\d+\b', '', news)  # Remove números
    processed_news = preprocess_text(news)
    return processed_news

# Exemplo de uso da função para prever se uma notícia é falsa ou não
def predict_fake_news(model, vectorizer, news_vectorized):
    # Previsão usando o modelo treinado
    prediction = model.predict(news_vectorized)
    return prediction

# Solicita ao usuário inserir uma notícia
user_input_news = input("Insira a notícia a ser classificada como fake ou real: ")

# Pré-processa a notícia de entrada do usuário
processed_user_input_news = preprocess_single_news(user_input_news)

# Vetoriza a notícia pré-processada usando o mesmo vetorizador usado no treinamento
user_input_news_vectorized = vectorizer.transform([processed_user_input_news])

# Exemplo de uso da função para prever se uma notícia é falsa ou não
prediction = predict_fake_news(nb_model, vectorizer, user_input_news_vectorized)

if prediction == 0:
    print("\nA notícia é classificada como falsa.")
else:
    print("\nA notícia é classificada como real.")