# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

#Simular datos usando cholesky
def simulate_data(df, n_simul):
    row,col=df.shape #Dimensiones
    x=df.to_numpy() #Datos como arreglo
    unos=np.ones((row,1)) #Vector de unos
    x_barra=(1/row)*(x.T@unos) #vector de medias
    x_centrado=x-unos@x_barra.T #Matriz de centrado
    sigma=(1/(row-1))*(x_centrado.T@x_centrado) #cov
    try:
        print('trying Usual covariance matrix \n')
        L=np.linalg.cholesky(sigma) #cholesky
        print('Success using usual covariance matriz')
        print('Usual covariance matriz determinant:', np.linalg.det(sigma))
        print('Usual covariance matriz condition number:', np.linalg.cond(sigma), '\n')
    except:
        print('Warning using usual covariance matrix')
        print('Usual covariance matriz determinant:', np.linalg.det(sigma))
        print('Usual covariance matriz condition number:', np.linalg.cond(sigma), '\n')
        print('trying Shrinkage estimator covariance matrix \n')
        eig,vect=np.linalg.eig(sigma)
        i=0
        while np.linalg.det(sigma)<15:
            sigma=sigma+np.eye(col)*(i*abs(min(eig)))
            i+=1
            if i==50:
                break
        #from sklearn.covariance import LedoitWolf
        #sigma=LedoitWolf().fit(x_centrado).covariance_
        L=np.linalg.cholesky(sigma) #cholesky
        print('Success using Shrinkage estimator covariance matriz')
        print('Increasing: '+str(i)+' times min abs eigenvalue, resulting in an increment of : '+str(i*abs(min(eig)))+'\n')
        print('Shrinkage estimator covariance matrix determinant:', np.linalg.det(sigma))
        print('Shrinkage estimator covariance matrix condition number:', np.linalg.cond(sigma), '\n')
    Ruido=np.random.randn(col,n_simul) #Datos ruido blanco
    aux=L@Ruido
    datos=aux.T #datos simulados, no media de originales
    df_datos=pd.DataFrame(datos, columns=df.columns)
    
    row_simul,col_simul=df_datos.shape #Dimensiones
    x_simul = df_datos.to_numpy() #Datos como arreglo
    unos_simul = np.ones((row_simul,1)) #Vector de unos
    x_barra_simul = (1/row_simul)*(x_simul.T@unos_simul)
    
    x_centrado_simul = x_simul - unos_simul@x_barra_simul.T #Matriz de centrado simulada
    datos2 = x_centrado_simul + unos_simul@x_barra.T #Agregar la media de los originales
    df_datos2=pd.DataFrame(datos2, columns=df.columns)
    
    return df_datos2


#Balancear datos por clase
def balance_data(datos, col_cat, n):
    data=datos.copy()
    for i in np.unique(data[col_cat]):
        df=data.loc[data[col_cat]==i].drop(col_cat, axis=1).copy()
        if len(df)>n:
            print('n can´t be less than length of some categorie')
            break
        if len(df)==n:
            pass
        else:
            n_simul=n-len(df)
            df_simul=simulate_data(df, n_simul)
            df_simul[col_cat]=i
            data=data.append(df_simul, ignore_index=True)
    return data


#Método de identificación de datos raros multivariante: proxy
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

#Porcentaje de valores a remover siempre y cuando esten más alejados de los bigotes
porc_remove=.1
#Número de proyecciones a realizar en cada iteración
N=10000

def kur_simul_montecarlo(data_standard, porc_remove=porc_remove, N=N):
    #Cantidad de valores a remover
    remove= int(np.ceil(len(data_standard)*porc_remove))
    print('Cantidad esperada de valores a remover '+str(remove))

    #Iniciar vector de posiciones removidas
    pos_removed=[]
    #Crear serie para extrar el índice de la posición en cada iteración
    #a medida que van saliendo valores
    pos_serie=pd.Series(range(len(data_standard)))

    while len(pos_removed)<remove:
        #Dimensiones de los datos
        row,col=data_standard.shape
        #Vectores de proyección
        V=np.random.randn(col,N)
        #Vectores de proyección normalizados
        u=V/np.linalg.norm(V, axis=0)
        #Proyectar datos
        P=data_standard@u
        #Calcular kurtosis
        K=kurtosis(P, axis=0)
        #vector de máxima kurtosis
        I=np.where(K==max(K))[0][0]
        #proyección de máxima kurtosis
        P_max=data_standard@u[:,I]

        #percentiles extremos
        q1,q3=np.percentile(P_max, [25,75])
        #Rango intercualtilico
        iqr=q3-q1
        #Bigotes
        BS=q3+1.5*iqr
        BI=q1-1.5*iqr

        #vector con las posiciones de los puntos más alejados en P_max
        far=np.where((P_max<BI)|(P_max>BS))[0]
        if len(far)>0:
            #puntos más alejados en P_max, ordenarlos de forma descendente
            points_far=pd.Series(abs(P_max[far]),far).sort_values(ascending=False)
            #posición a remover del valor más alejado de los bigotes
            pos_remove=points_far.iloc[:1].index.values
            #Agregar posición removida en el vector "pos_removed"
            pos_removed.append(pos_serie.index[pos_remove])
            #vector booleano de posiciones a mantener
            bool_to_keep=~np.isin(list(range(row)),pos_remove)
            #Remover valor más alejado de los bigotes
            data_standard=data_standard[bool_to_keep,:]
            pos_serie=pos_serie[bool_to_keep]
        else:
            #Resultado
            print('No hay puntos más alejados de los bigotes.')
            break

    #Resultado
    print('Han sido retirados: '+str(len(pos_removed))+' registros, correspondientes al: '+
          str(round(len(pos_removed)/(row+len(pos_removed))*100,3))+'% de los datos originales.\n')
    #vector booleano de posiciones a mantener
    bool_to_keep=~np.isin(list(range(row)),pos_removed)

    #Mostrar última proyección de máxima curtosis
    fig,ax=plt.subplots()
    plt.boxplot(data_standard@u[:,I])
    
    return data_standard


#kur_simul_montecarlo por clase
from sklearn import preprocessing

def kur_per_class(datos, col_class):
    dt=pd.DataFrame(columns=datos.columns)
    for i in np.unique(datos[col_class]):
        scaler = preprocessing.StandardScaler()
        #Estandarizar los datos a limpiar
        df_standard=scaler.fit_transform(datos.loc[datos[col_class]==i].drop([col_class], axis=1).values)
        #limpiar datos
        df_standard=kur_simul_montecarlo(df_standard)
        #Desestandarizar los datos limpios, sin col_class
        temp=pd.DataFrame(scaler.inverse_transform(df_standard),columns=datos.columns[datos.columns!=col_class])
        #Calcular categoria col_class respectiva
        temp[col_class]=int(i)
        #Concatenarlos
        dt=dt.append(temp, ignore_index=True)
    return dt