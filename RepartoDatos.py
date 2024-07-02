import numpy as np

class RepartoDatos:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def dividir_dataset(self, N,random_state=None):

        if random_state is not None:
            np.random.seed(random_state)


        indices = list(self.X.index)
        np.random.shuffle(indices)

        X_random = self.X.reindex(indices)
        y_random = self.y.reindex(indices)

        tamano_parte = len(X_random) // N

        partes_X = []
        partes_y = []
        inicio = 0

        for i in range(N):
            fin = inicio + tamano_parte if i < N - 1 else None
            parte_X = X_random.iloc[inicio:fin].copy().reset_index(drop=True)
            parte_y = y_random.iloc[inicio:fin].copy().reset_index(drop=True)
            partes_X.append(parte_X)
            partes_y.append(parte_y)
            inicio = fin

        return partes_X, partes_y #da una lista con las matrices de X, y otra con las de y
    
    def dividir_etiquetas(self,datos_clientes, etiquetas_por_cliente,aleatorio=False,random_state=None):
        
        if random_state is not None:
            np.random.seed(random_state)

        etiquetas_asignadas = []
        indice_inicio = 0
        
        if aleatorio:
            etiquetas_disponibles = list(range(len(datos_clientes[0].columns)))
            np.random.shuffle(etiquetas_disponibles)
        else:
            etiquetas_disponibles = list(range(len(datos_clientes[0].columns)))


        for num_etiquetas, cliente in zip(etiquetas_por_cliente, datos_clientes):
            
            if aleatorio:
                etiquetas_cliente = cliente.iloc[:, etiquetas_disponibles[indice_inicio:indice_inicio+num_etiquetas]]
            else:
                etiquetas_cliente = cliente.iloc[:, indice_inicio:indice_inicio+num_etiquetas]
            
            etiquetas_asignadas.append(etiquetas_cliente)
            
            indice_inicio += num_etiquetas
        
        return etiquetas_asignadas

