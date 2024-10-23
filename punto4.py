import numpy as np
# punto 1
def amplitudDeTransicion(vector1,vector2):
    # Normalizar vector 1
    normaVector1 = np.linalg.norm(vector1)
    vector1 = vector1/normaVector1
    # Normalizar vector2
    normaVector2 = np.linalg.norm(vector2)
    vector2 = vector2/normaVector2
    # Solo se calculara la amplitud de transicion en caso de que los vectores sean del mismo tamaño
    if len(vector1) != len(vector2) :
        return "error de longitud entre los dos vectores"
    else:
        amplitudDeTransicion = np.dot(np.conjugate(vector2.T),vector1)
        return amplitudDeTransicion

# punto 2
def calculaMedia(matriz,vectorKet):
    esHermitania = np.array_equal(matriz,np.conjugate(np.transpose(matriz)))
    # Normalizar matriz
    normaMatriz = np.linalg.norm(matriz)
    matriz = matriz/normaMatriz

    vectorKet = np.array([[1],[0]])
    # Normalizar vector
    normaVector = np.linalg.norm(vectorKet)
    vectorKet = vectorKet/normaVector

    # Solo se calculara la media y la varianza en caso de que la matriz sea hermitania
    if not esHermitania:
        return "La matriz no es hermitania"
    else:
        # Calculo de la media
        valorEsperado = np.dot(np.conjugate(matriz.T),vectorKet)
        valorEsperado = np.dot(np.conjugate(valorEsperado.T),vectorKet)
        return valorEsperado
        


def calculaVarianza(matriz,vectorKet):
    esHermitania = np.array_equal(matriz,np.conjugate(np.transpose(matriz)))
    # Normalizar matriz
    normaMatriz = np.linalg.norm(matriz)
    matriz = matriz/normaMatriz

    vectorKet = np.array([[1],[0]])
    # Normalizar vector
    normaVector = np.linalg.norm(vectorKet)
    vectorKet = vectorKet/normaVector
    if not esHermitania:
        return "La matriz no es hermitania"
    else:
        # Solo se calculara la media y la varianza en caso de que la matriz sea hermitania
        # Calculo de la varianza
        # primero se realiza el calculo del omega
        longitud = len(vectorKet)
        matrizIdentidad = np.eye(int(longitud))
        valorEsperado = calculaMedia(matriz,vectorKet)
        operador1 = matriz
        operador2 = valorEsperado* matrizIdentidad
        operadorDelta = operador1 - operador2
            
        # Ahora seguiremos la formula para hallar la varianza
        operador = np.dot(operadorDelta, operadorDelta)
        varianza = np.dot(np.conjugate(operador.T),vectorKet)
        varianza = np.dot(np.conjugate(varianza.T),vectorKet)
        return varianza
    
#punto 3
def calculoTransitoDeVectoresPropios(matriz,vectorKet):
    # Tercer punto
    # Primero se calcularan los valores propios del observable.
    valoresPropios, vectoresPropios = np.linalg.eig(matriz)
    probabilidades=[]
    # Ahora se calculara la probabilidad de que el vectorKet pueda transitar a alguno de los vectores propios despues de la observacion
    for rep in range (len(vectoresPropios)):
        probabilidad = np.dot(np.conjugate(vectoresPropios[rep].T),vectorKet)
        probabilidad = probabilidad**2
        probabilidades.append(probabilidad)
        
    return probabilidades,valoresPropios,vectoresPropios

#punto 4
def estadoFinal(estadoInicial,matrices):
    numeroDeMatrices = len(matrices)
    matrices = map(np.array,matrices) # Se le aplicara la funcion np.array a todas las matrices
    for rep in range(numeroDeMatrices):
        estadoInicial = np.dot(estadoInicial,matrices[rep])
        
    return estadoInicial

