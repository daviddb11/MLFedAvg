from keras import Sequential, Input
from keras.layers import Dense
from keras.models import clone_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras.initializers import RandomNormal
class MLFedAvg:
    def __init__(self, neurons,rho=1,S=10):
        self.neurons = neurons #neuronas  core
        self.rho=rho #porcentaje de clientes en cada ronda
        self.S=S #epochs de cada cliente


    def model_core(self):
        core = Sequential()
        core.add(Input(shape=(self.neurons[0],)))
        for n in range(len(self.neurons) - 2):
            core.add(Dense(self.neurons[n + 1], activation='relu'))
        core.add(Dense(self.neurons[-1], activation='relu'))
        core.summary()
        return core

    def create_client_model(self,core, yCs, show_summ=True):
        clientes_etiquetas = {}
        for i, yC in enumerate(yCs, start=1):
            cliente = f"Cliente {i}"
            etiqueta = yC.shape[1]
            clientes_etiquetas[cliente] = etiqueta

        modelos_clientes = {}
        
        for client, label in clientes_etiquetas.items():
            client_model = clone_model(core)
            
            client_model.set_weights(core.get_weights())
            
            client_model.add(Dense(label, activation='sigmoid'))
            
            modelos_clientes[client] = client_model

        if show_summ:
            for cliente, modelo in modelos_clientes.items():
                print("Modelo para {}:".format(cliente))
                modelo.summary()
                print("\n\n")

        return modelos_clientes
    
    def compilar_modelos(self, modelos_clientes, lr=0.001,loss="binary_crossentropy",metrics=['accuracy']):


        for cliente, modelo in modelos_clientes.items():
            opt = keras.optimizers.Adam(learning_rate=lr)
            modelo.compile(optimizer=opt, loss=loss, metrics=metrics)
            modelos_clientes[cliente] = modelo

        return modelos_clientes
    
    def entrenar_modelos(self, modelos_clientes, X_train_list, y_train_list, callbacks, draw=True,
                        metrica_pintar='accuracy', val_split=0.2):
        modelos_entrenados = {}
        metricas_entrenamiento = {}
        metricas_validacion = {}
        early_stopped=[]
        metricas_val_loss = {} 
        

        for i, (nombre_modelo, modelo) in enumerate(modelos_clientes.items()):
            print(f"Entrenando modelo {i+1} de {len(modelos_clientes)}: {nombre_modelo}")

            X_train_cliente = X_train_list[i]
            y_train_cliente = y_train_list[i]
            y_train_cliente = tf.cast(y_train_cliente, tf.float32)

            historia_entrenamiento = modelo.fit(X_train_cliente, y_train_cliente,
                                                batch_size=30,
                                                epochs=self.S,
                                                validation_split=val_split,
                                                verbose=2, callbacks=callbacks)
            
            early_stopped.append(callbacks[0].stopped_epoch > 0)

            if draw:
                plt.plot(historia_entrenamiento.history[metrica_pintar], label='train')
                plt.plot(historia_entrenamiento.history[f'val_{metrica_pintar}'], label='validation')
                plt.title(f'{metrica_pintar.capitalize()} del modelo: {nombre_modelo}')
                plt.xlabel('epoch')
                plt.ylabel(metrica_pintar)
                plt.legend()
                plt.show()

            metricas_entrenamiento[nombre_modelo] = historia_entrenamiento.history[metrica_pintar]
            metricas_validacion[nombre_modelo] = historia_entrenamiento.history[f'val_{metrica_pintar}']
            metricas_val_loss[nombre_modelo] = historia_entrenamiento.history['val_loss'] 

            modelos_entrenados[nombre_modelo] = modelo

        return modelos_entrenados, metricas_entrenamiento, metricas_validacion, metricas_val_loss, early_stopped


    
    def extraer_pesos_intermedios(self,diccionario_modelos):
        pesos_modelos = {}
        for cliente, modelo in diccionario_modelos.items():
            pesos_modelo = []
            for capa in modelo.layers[:-1]:
                pesos_capa = capa.get_weights()
                pesos_modelo.append(pesos_capa)
            pesos_modelos[cliente] = pesos_modelo
        return pesos_modelos
    
    def extraer_pesos_ultima_capa(self, diccionario_modelos):
        pesos_ultima_capa = {}
        for cliente, modelo in diccionario_modelos.items():
            ultima_capa = modelo.layers[-1]
            pesos_ultima_capa[cliente] = ultima_capa.get_weights()
        return pesos_ultima_capa

    
    def agregar_pesos(self,diccionario_modelos, instancias_por_cliente, etiquetas_por_cliente,agg=1):
        total_instancias = sum(instancias_por_cliente)
        total_etiquetas = sum(etiquetas_por_cliente)

        if agg==0:
                factores_ponderacion = [1/len(instancias_por_cliente)]*len(instancias_por_cliente) 

        elif agg==1:
                factores_ponderacion = [instancia / total_instancias for instancia in instancias_por_cliente]
        elif agg==2:
                factores_ponderacion = [instancia / total_instancias * etiqueta / total_etiquetas for 
                                        instancia, etiqueta in zip(instancias_por_cliente, etiquetas_por_cliente)]
                factores_ponderacion=[factor /sum(factores_ponderacion) for factor in factores_ponderacion]
        else:  
                factores_ponderacion = [instancia / total_instancias for instancia in instancias_por_cliente]
                factores_ponderacion = [1 / x for x in factores_ponderacion]
                factores_ponderacion = [factor /sum(factores_ponderacion) for factor in factores_ponderacion]


        pesos_combinados = {}
        clientes = list(diccionario_modelos.keys())
        num_capas = len(diccionario_modelos[clientes[0]])
        
        for i in range(num_capas):
            pesos_combinados_capa = []
            for j in range(len(diccionario_modelos[clientes[0]][i])):
                peso_combinado = sum(diccionario_modelos[cliente][i][j] * factores_ponderacion[indice] for indice, cliente in enumerate(clientes))
                pesos_combinados_capa.append(peso_combinado)
            pesos_combinados[f'Capa_{i+1}'] = pesos_combinados_capa
        
        return pesos_combinados
    
    def aplicar_pesos_a_modelo(self,modelo, pesos_combinados):
        for i, (capa, pesos) in enumerate(pesos_combinados.items()):
            modelo.layers[i].set_weights(pesos)
        return modelo
    
    def aplicar_pesos_ultima_capa(self,diccionario_modelos, pesos_ultima_capa):
        for nombre_modelo, modelo in diccionario_modelos.items():
            if nombre_modelo in pesos_ultima_capa:
                ultima_capa = modelo.layers[-1]
                ultimos_pesos = pesos_ultima_capa[nombre_modelo]
                ultima_capa.set_weights(ultimos_pesos)
        return diccionario_modelos