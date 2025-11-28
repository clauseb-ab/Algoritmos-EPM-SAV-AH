import networkx as nx
import numpy as np
import random


#Funcion para crear grafo con signos
def create_signed_graph(n, p, probability_negative_edges):
    # Crea un grafo Erdos-Renyi con n vértices y probabilidad p de tener aristas
    G = nx.erdos_renyi_graph(n, p)
    
    # Asigna signos a las aristas basado en la probabilidad de aristas negativas
    for (u, v) in G.edges():
        # Asigna un signo positivo o negativo basado en probabilidad
        sign = random.choices([1, -1], weights=[1 - probability_negative_edges, probability_negative_edges])[0]
        G[u][v]['sign'] = sign
    
    return G

#Funcion para creal grafo de intervalo unitario positivo
def generar_grafo_intervalo_unitario(n, epsilon):
    # Generar los intervalos unitarios con un control de densidad epsilon
    centros = []
    for _ in range(n):
        a = random.uniform(0, 1)  # Ajusta el rango según el valor de epsilon,cuanto menor es epsilon, mayor es la probabilidad de solapamiento
        centros.append(a)
    # Crear un grafo vacío
    G = nx.Graph()
    
    # Añadir nodos al grafo
    for i in range(n):
        G.add_node(i)
    
    # Verificar pares de nodos que se solapan
    for u in range(n):
        for v in range(u + 1, n):
            # Verificar si los intervalos se solapan
            if abs(centros[u]-centros[v])<=epsilon:
                # Agregar arista si hay solapamiento
                G.add_edge(u, v, sign=+1)
    return G

#funcion para agregar aristas negativas de forma aleatoria a un grafo
def agregar_aristas_negativas(G,probability_negative_edges):
    nodos = list(G.nodes)
    
    for u in range(len(nodos)):
        for v in range(u + 1, len(nodos)):
            # Verificar que no exista una arista positiva
            if not G.has_edge(u, v):
                # Agregar una arista negativa con una probabilidad dada
                sign = random.choices([0, -1], weights=[1 - probability_negative_edges, probability_negative_edges])[0]
                if sign==-1:    
                    G.add_edge(u, v, sign=-1)
                    
    
    return G

#funcion para crear matriz de adyacencia de un grafo con signos
def graph_to_signed_adjacency_matrix(G):
    # Inicializa una matriz de adyacencia con ceros
    n = G.number_of_nodes()
    adj_matrix = np.zeros((n, n))
    
    # Llena la matriz con +1 para aristas positivas y -1 para aristas negativas
    for u, v, data in G.edges(data=True):
        adj_matrix[u, v] = data['sign']
        adj_matrix[v, u] = data['sign']  # Asegurar simetría en el grafo no dirigido
    
    return adj_matrix

#Fijar semilla 
random.seed(123)