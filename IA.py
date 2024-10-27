#Ejemplo de prueba:
#These Rules, made under the Animal Diseases Act, regulate the importation of animals and introduce restrictions on the movement of animals in Kenya and other measures for the control of animal diseases. Importation of animals shall be done only through prescribed ports and under a licence granted by the Director of Veterinary Services. Specified certificates shall be required in respect of animals to be imported. The Rules also provide for tests, treatment and quarantine for imported cattle and other animals. Animals from Tanzania or Uganda may be imported subject to such restrictions and requirements as the Director may, from time to time, direct. No cattle, swine, sheep, goats or captive wild animal shall be moved within a restricted area, i.e. any of the areas described in the First Schedule, except under a licence issued under these Rules. Various other rules regarding the movement of animals are prescribed for purposes of the control of diseases. Other provisions concern the notification of infected areas, orders that may be made in infected areas by the Director of Veterinary Services, a veterinary officer or inspector the detention of animals, the branding of animals, the disinfecting of animals, places and vehicles, the handling of stray animals, the movement of material from infected areas, etc. 


import numpy as np
from gensim.models import Word2Vec
import pandas as pd

# Cargar el DataFrame (suponiendo que el archivo es un CSV)
df = pd.read_excel("Corpus-Agro.xlsx")

# Cargar el modelo Word2Vec
modelo_w2v = Word2Vec.load("modelo_word2vec.model")

# Funciones que has definido anteriormente
def vectorizar_resumen(resumen):
    lematizar_resumen = str(resumen).split()
    resumen_tokenizado = [modelo_w2v.wv[resumen_token] for resumen_token in lematizar_resumen if resumen_token in modelo_w2v.wv]
    return np.mean(resumen_tokenizado, axis=0)

def vectorizar_consulta(consulta):
    lematizar_consulta = str(consulta).split()
    consulta_tokenizada = [modelo_w2v.wv[consulta_token] for consulta_token in lematizar_consulta if consulta_token in modelo_w2v.wv]
    return np.mean(consulta_tokenizada, axis=0)

def buscador(consulta, n=5):
    vector_consulta = vectorizar_consulta(consulta)
    # Aquí deberías calcular la similitud con los resúmenes
    # Asegúrate de que el DataFrame df esté definido y tenga los resúmenes vectorizados
    df['Similitud'] = df['Vector_resumen'].apply(lambda x: 1 - cosine(vector_consulta, x))
    return df.nlargest(n, 'Similitud')

# Ejecución del programa
consulta = input("Ingrese su consulta: ")
resultados = buscador(consulta, 7)
print(resultados)

