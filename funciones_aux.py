from Metricas import Metrica
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

metricas=Metrica()

def eliminar_ceros(X_list, y_list):
    X_sin_ceros = []
    y_sin_ceros = []

    for X, y in zip(X_list, y_list):
        filas_ceros = (y == 0).all(axis=1)
        y_filtrado = y[~filas_ceros]
        X_filtrado = X[~filas_ceros]
        X_sin_ceros.append(X_filtrado)
        y_sin_ceros.append(y_filtrado)

    return X_sin_ceros, y_sin_ceros

def obtener_etiquetas_test_global(y_train, y_test):
    y_test_filtrado = []  
    
    
    for df_train in y_train:
        etiquetas = df_train.columns.tolist()  
        df_test_filtrado = y_test[0][etiquetas]  
        y_test_filtrado.append(df_test_filtrado)  
    
    return y_test_filtrado

def evaluar_modelos(modelos, X_test, y_test_filtrado):
    resultados_dict = {"Cliente": [], "Subset Accuracy":[], "Hamming Loss (inv)": [], "F1 Micro": [], "F1 Macro": [], "JS": []}

    if len(X_test)==1:
        
        y_test_filtrado_copy = y_test_filtrado.copy()  

        for cliente, modelo in modelos.items():
            modelo.compile(metrics=["accuracy", metricas.HammingLoss(), metricas.f1_micro(), metricas.f1_macro(), metricas.JS()])
            
            y_test_cliente = y_test_filtrado_copy.pop(0)  
            
            resultados = modelo.evaluate(X_test[0], y_test_cliente.astype(np.float32), verbose=0)
            
            resultados_dict["Cliente"].append(cliente)
            resultados_dict["Subset Accuracy"].append(resultados[1])
            resultados_dict["Hamming Loss (inv)"].append(1 - resultados[2])
            resultados_dict["F1 Micro"].append(resultados[3])
            resultados_dict["F1 Macro"].append(resultados[4])
            resultados_dict["JS"].append(resultados[5])
        
        df_resultados = pd.DataFrame(resultados_dict)

        df_resultados = pd.DataFrame(resultados_dict)

        medias_dict = {
            "Cliente": ["Media"],
            "Subset Accuracy": [df_resultados["Subset Accuracy"].mean()],
            "Hamming Loss (inv)": [df_resultados["Hamming Loss (inv)"].mean()],
            "F1 Micro": [df_resultados["F1 Micro"].mean()],
            "F1 Macro": [df_resultados["F1 Macro"].mean()],
            "JS": [df_resultados["JS"].mean()]
        }

        df_medias = pd.DataFrame(medias_dict)
        df_resultados = pd.concat([df_resultados, df_medias], ignore_index=True)

        
        print("----------------------------------------------------------------------------")
        print(df_resultados.to_string(index=False))
        print("----------------------------------------------------------------------------")
        print()
        
    else:

        for modelo in modelos.values():
            modelo.compile(metrics=["accuracy", metricas.HammingLoss(), metricas.f1_micro(), metricas.f1_macro(), metricas.JS()])

        for i, (cliente, modelo) in enumerate(modelos.items()):
            resultados = modelo.evaluate(X_test[i], y_test_filtrado[i].astype(np.float32), verbose=0)

            resultados_dict["Cliente"].append(cliente)
            resultados_dict["Subset Accuracy"].append(resultados[1])
            resultados_dict["Hamming Loss (inv)"].append(1 - resultados[2])
            resultados_dict["F1 Micro"].append(resultados[3])
            resultados_dict["F1 Macro"].append(resultados[4])
            resultados_dict["JS"].append(resultados[5])

        df_resultados = pd.DataFrame(resultados_dict)

        medias_dict = {
            "Cliente": ["Media"],
            "Subset Accuracy": [df_resultados["Subset Accuracy"].mean()],
            "Hamming Loss (inv)": [df_resultados["Hamming Loss (inv)"].mean()],
            "F1 Micro": [df_resultados["F1 Micro"].mean()],
            "F1 Macro": [df_resultados["F1 Macro"].mean()],
            "JS": [df_resultados["JS"].mean()]
        }

        df_medias = pd.DataFrame(medias_dict)
        df_resultados = pd.concat([df_resultados, df_medias], ignore_index=True)

        print("----------------------------------------------------------------------------")
        print(df_resultados.to_string(index=False))
        print("----------------------------------------------------------------------------")




def dibujar_graficas(ultimos_accuracy_entrenamiento, ultimos_accuracy_validacion, primeros_accuracy_entrenamiento=None, primeros_accuracy_validacion=None):
    clientes = sorted(set(cliente for cliente, _ in ultimos_accuracy_entrenamiento))
    for cliente in clientes:
        plt.figure(figsize=(9, 4))
        plt.title(cliente)
        valores_validacion = None
        for label, datos_entrenamiento, datos_validacion in [('Últimos Entrenamiento', ultimos_accuracy_entrenamiento, None), 
                                                             ('Últimos Validación', ultimos_accuracy_validacion, None),
                                                             ('Primeros Entrenamiento', primeros_accuracy_entrenamiento, None), 
                                                             ('Primeros Validación', primeros_accuracy_validacion, None)]:
            if datos_entrenamiento is not None:
                valores_entrenamiento = [acc for cli, acc in datos_entrenamiento if cli == cliente]
                plt.plot(range(1, len(valores_entrenamiento) + 1), valores_entrenamiento, label=label)
            if datos_validacion is not None:
                valores_validacion = [acc for cli, acc in datos_validacion if cli == cliente]
                plt.plot(range(1, len(valores_validacion) + 1), valores_validacion, label=label)
        if valores_validacion is not None:
            plt.xticks(range(1, len(valores_validacion) + 1))
        else:
            plt.xticks(range(1, len(valores_entrenamiento) + 1))
        plt.xlabel('Ronda Federada')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right', fontsize='small', fancybox=True, framealpha=0.8)

        plt.show()


def train_test_split_list(X_list, y_list, test_size=0.2, random_state=None):
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for X, y in zip(X_list, y_list):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    return X_train_list, y_train_list, X_test_list, y_test_list

def cuenta_instancias(df_list):
    instance_counts = []

    for df in df_list:
        instance_counts.append(len(df))

    return instance_counts

def evalua_modelo_unico(modelo,X_test,y_test):

    modelo.compile(metrics=["accuracy", metricas.HammingLoss(), metricas.f1_micro(), metricas.f1_macro(), metricas.JS()])

    resultados_dict = {"Subset Accuracy":[], "Hamming Loss (inv)": [], "F1 Micro": [], "F1 Macro": [], "JS": []}
                
    resultados = modelo.evaluate(X_test[0], y_test[0].astype(np.float32), verbose=0)
                
    resultados_dict["Subset Accuracy"].append(resultados[1])
    resultados_dict["Hamming Loss (inv)"].append(1 - resultados[2])
    resultados_dict["F1 Micro"].append(resultados[3])
    resultados_dict["F1 Macro"].append(resultados[4])
    resultados_dict["JS"].append(resultados[5])
    
    df_resultados = pd.DataFrame(resultados_dict)
    print("EVALUACIÓN MODELO ÚNICO CON CONJUNTO DE TEST GLOBAL")
    print("----------------------------------------------------------------------------")
    print(df_resultados.to_string(index=False))
    print("----------------------------------------------------------------------------")
    print()




def ruido_lista(dfs, noise_list, seed=None):
    if seed is not None:
        np.random.seed(seed)
    noisy_dataframes = []
    for df, noise_level in zip(dfs, noise_list):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        noise = np.abs(np.random.normal(0, noise_level, df[numeric_cols].shape))
        df_noisy = df.copy()
        df_noisy[numeric_cols] = df[numeric_cols] + noise
        noisy_dataframes.append(df_noisy)
    return noisy_dataframes

def ruido(dataframes, noise_level,seed=None):
    if seed is not None:
        np.random.seed(seed)
    noisy_dataframes = []
    for df in dataframes:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        noise = np.random.normal(0, noise_level, df[numeric_cols].shape)
        df_noisy = df.copy()
        df_noisy[numeric_cols] = df[numeric_cols] + noise
        noisy_dataframes.append(df_noisy)
    return noisy_dataframes

def lista_aleatoria(N,a,b,seed):
    random.seed(seed)
    return [random.uniform(a,b) for _ in range(N)]
