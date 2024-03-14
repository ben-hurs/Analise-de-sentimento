import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")

# Carregando os stop words
df_stop = pd.read_csv('stop_words.csv')

# Carregando o modelo a partir do arquivo
modelo = joblib.load('modelo.pkl')

frase = input("Digite uma frase: ")

# Ajustar o CountVectorizer
countvec = CountVectorizer(ngram_range=(1, 2))
countvec.fit(df_stop['Frase Limpa'])

# Vetorização da frase de teste usando o mesmo CountVectorizer ajustado
frase_vetorizada = countvec.transform([frase])

# Fazer a previsão
mapa_sentimentos = {0: 'Felicidade', 1: 'Tristeza', 2: 'Raiva'}
classe_prevista = modelo.predict(frase_vetorizada)
sentimento_previsto = mapa_sentimentos[classe_prevista[0]]
print(frase, "->", sentimento_previsto)

