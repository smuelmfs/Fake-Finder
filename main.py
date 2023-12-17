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
           'precision': 'precision_weighted',
           'recall': 'recall_weighted',
           'f1_score': 'f1_weighted'}
cv_results = cross_validate(nb_model, X_train_vectorized, y_train, scoring=scoring, cv=5)
print("Cross-validation results for Naive Bayes Model:")
for metric_name, result in cv_results.items():
    print(f"{metric_name}: {result.mean()} (±{result.std()})")

# Inicialização e treinamento de outros modelos (SVM, Random Forest, Decision Tree)
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
print("Decision Tree Accuracy:", dt_clf.score(X_test_vectorized, y_test))

# Ajuste de hiperparâmetros para o Naive Bayes Multinomial
parameters = {'alpha': [0.5, 1.0, 1.5]}
nb_clf = MultinomialNB()
grid_search = GridSearchCV(nb_clf, parameters)
grid_search.fit(X_train_vectorized, y_train)

best_nb_clf = grid_search.best_estimator_
print("Best Naive Bayes Multinomial Accuracy:", best_nb_clf.score(X_test_vectorized, y_test))

# Exemplo de oversampling para balancear classes
oversample = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = oversample.fit_resample(X_train_vectorized, y_train)

# Verificação das palavras mais importantes para o Naive Bayes Multinomial
feature_names = vectorizer.get_feature_names()
top_n = 10
top_positive = sorted(zip(best_nb_clf.coef_[0], feature_names), reverse=True)[:top_n]
top_negative = sorted(zip(best_nb_clf.coef_[0], feature_names))[:top_n]

print("Top Palavras Positivas:")
print([word for coef, word in top_positive])

print("Top Palavras Negativas:")
print([word for coef, word in top_negative])

# Inicialização e treinamento do modelo SVM com kernel linear
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_vectorized, y_train)

# Avaliação do modelo
svm_linear_accuracy = svm_linear.score(X_test_vectorized, y_test)
print("Accuracy with Linear SVM:", svm_linear_accuracy)


# Função para pré-processamento de uma única notícia
def preprocess_single_news(news):
    news = re.sub(r'\b\d+\b', '', news)  # Remove números
    processed_news = preprocess_text(news)
    return processed_news


# Função para prever se uma notícia é fake ou não
def predict_fake_news(model, vectorizer, news):
    # Pré-processamento da notícia
    processed_news = preprocess_single_news(news)

    # Vetorização usando o mesmo vetorizador usado no treinamento
    news_vectorized = vectorizer.transform([processed_news])

    # Previsão usando o modelo treinado
    prediction = model.predict(news_vectorized)

    return prediction[0]

# Exemplo de uso da função para prever se uma notícia é fake ou não
sample_news = "Um policial penal de 44 anos reagiu a uma tentativa de assalto na tarde desta quinta-feira, 14, na rua Raio do Sol, no bairro Sapopemba, na Zona Leste de São Paulo. O agente chegava de moto em casa quando foi abordado por dois homens, que também estavam de moto. Ao perceber a presença dos assaltantes, ele atirou – um jovem de 20 anos morreu, enquanto o outro, um adolescente, foi detido após ser baleado. Após controlar a situação, o policial gravou um vídeo relatando o ocorrido. “Minha moto está parada na porta de casa. Quando eu estava chegando, vi que eles estavam me seguindo. Aí desci, parei na porte e na hora que eu desci da moto, me armei. Eles começaram a gritar ‘perdeu’ na minha direção. Eu já esperava que eles iam me abordar. Aí atirei e fiz o revide. Eles tentaram fugir, mas o piloto está caído. O outro está dizendo que foi baleado e dominado. Já liguei no 190 e estou esperando o apoio”, disse. Em contato com a reportagem do site da Jovem Pan, a SSP (Secretaria de Segurança Pública) informou que o caso foi registrado como tentativa de roubo de veículo e morte decorrente à intervenção policial no 69° DP (Teotônio Vilela), que solicitou assessoramento ao DHPP."
prediction = predict_fake_news(best_nb_clf, vectorizer, sample_news)

if prediction == 0:
    print("A notícia é classificada como fake.")
else:
    print("A notícia é classificada como real.")