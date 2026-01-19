import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

#funcion para crear una matriz de adyacencia
def create_random_adjacency_matrix(n, prob_positive=0.3, prob_negative=0.3):
    adj_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            rand_val = np.random.rand()
            if rand_val < prob_positive:
                value = 1
            elif rand_val < prob_positive + prob_negative:
                value = -1
            else:
                value = 0
            adj_matrix[i, j] = value
            adj_matrix[j, i] = value
    
    return adj_matrix


#funcion para crear un grafo a partir de una matriz de adyacencia
def create_weighted_graph(adj_matrix):
    # Crear un grafo vacío
    G = nx.Graph()
    
    # Obtener el número de nodos
    num_nodes = len(adj_matrix)
    
    # Añadir nodos al grafo
    G.add_nodes_from(range(num_nodes))
    
    # Añadir aristas ponderadas al grafo
    for i in range(num_nodes):
        for j in range(i, num_nodes):  # Para grafos no dirigidos, solo necesitamos la mitad superior de la matriz
            if adj_matrix[i][j] != 0:
                G.add_edge(i, j, weight=adj_matrix[i][j])
    
    return G

def draw_graph(G):
    # Dibujar el grafo
    pos = nx.spring_layout(G)  # Posiciones de los nodos
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=15)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
    
def removeNegativeEdges(G):
    k=len(G)
    for i in range(k):
        for j in range(k):
            if G[i][j]==-1:
                G[i][j]=0
                
    return G