import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf

# Carregando os stop words
df_stop = pd.read_csv('stop_words_novo.csv')

# Carregando o modelo a partir do arquivo
modelo = joblib.load('modelo_redes_neurais.pkl')

# Mapeamento das classes
mapa_classes = {
    0: 'Felicidade',
    1: 'Tristeza',
    2: 'Raiva'
}

# Ajustar o CountVectorizer
countvec = CountVectorizer(ngram_range=(1, 2))
countvec.fit(df_stop['Frase'])

# Função de previsão eficiente usando TensorFlow
def prever_classe(frase_vetorizada):
    return modelo.predict(frase_vetorizada, verbose=0)

while True:
    frase = input("Digite uma frase (ou 'exit' para sair): ")
    if frase.lower() == 'exit':
        print("Programa encerrado.")
        break

    # Vetorização da frase de teste usando o mesmo CountVectorizer ajustado
    frase_vetorizada = countvec.transform([frase])

    # Converter a matriz esparsa em um tensor
    frase_vetorizada_tensor = tf.convert_to_tensor(frase_vetorizada.toarray())

    # Fazer a previsão
    classe_prevista_prob = prever_classe(frase_vetorizada_tensor)
    classe_prevista = tf.argmax(classe_prevista_prob, axis=1).numpy()[0]
    classe_prevista_nome = mapa_classes[classe_prevista]
    print(frase, "->", classe_prevista_nome)
