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

#Funcion para crear grafo de intervalo unitario positivo
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

def set_complete():
    #Fijar semilla 
    random.seed(123)

    # Parámetros
    vertex_counts = list(range(10, 231, 20))  # Número de vértices de 10 a 230 en pasos de 20
    p_values=[1]
    probability_negative_edges = [0.2,0.5,0.8]  # Porcentajes de aristas negativas
    total_graphs = 108  # Total de grafos
    generated_graphs = []  # Lista para almacenar los grafos generados

    c = 0  # Contador para identificar cada matriz de adyacencia
    matrices = {}  # Diccionario para almacenar las Matrices de adyacencia de cada grafo

    for idx, n in enumerate(vertex_counts):
        for p in p_values:
            for neg_prob in probability_negative_edges:
                for i in range(3):  # Generar 12 grafos para cada combinación
                    c+=1
                    G = create_signed_graph(n, p, neg_prob)
                    generated_graphs.append(G)

                    # Convertir el grafo a matriz de adyacencia con signo
                    adj_matrix = graph_to_signed_adjacency_matrix(G)
                    
                    m=[adj_matrix,neg_prob]
                    
                    matrices[f'matriz{c}'] = np.array(m, dtype=object)

    print(c)

def set_random():
    #Fijar semilla 
    random.seed(216)

    # Parámetros
    vertex_counts = list(range(10, 251, 20)) 
    p_values = [0.2, 0.5, 0.8]  # Porcentajes de aristas
    probability_negative_edges = [0.2,0.5,0.8]  # Porcentajes de aristas negativas
    total_graphs = 117  # Total de grafos
    generated_graphs = []  # Lista para almacenar los grafos generados

    c = 0  # Contador para identificar cada matriz de adyacencia
    matrices = {}  # Para almacenar las Matrices de adyacencia de cada grafo

    for idx, n in enumerate(vertex_counts):
        for p in p_values:
            for neg_prob in probability_negative_edges:
                    c+=1
                    G = create_signed_graph(n, p, neg_prob)
                    generated_graphs.append(G)

                    # Convertir el grafo a matriz de adyacencia con signo
                    adj_matrix = graph_to_signed_adjacency_matrix(G)
                    
                    m=[adj_matrix,neg_prob]
                    
                    matrices[f'matriz{c}'] = np.array(m, dtype=object)

    print(c)

def set_ui():
    #Fijar semilla 
    random.seed(217)

    # Parámetros
    vertex_counts = list(range(10, 251, 20))  # Número de vértices de 10 a 250 en pasos de 20
    epsilon_values = [1/8, 1/2,3/4]  # valores de epsilon
    probability_negative_edges = [0.2, 0.5, 0.8]  # Porcentajes de aristas negativas
    total_graphs = 117  # Total de grafos que necesitamos (117)
    generated_graphs = []  # Lista para almacenar los grafos generados

    matrices={}
    c=0

    for idx, n in enumerate(vertex_counts):
        for epsilon in epsilon_values:
            for neg_prob in probability_negative_edges:
                c+=1
                G = generar_grafo_intervalo_unitario(n,epsilon)
                g = agregar_aristas_negativas(G, neg_prob)
                generated_graphs.append(g)

                # Convertir el grafo a matriz de adyacencia con signo
                adj_matrix = graph_to_signed_adjacency_matrix(g)

                m=[adj_matrix,epsilon]

                matrices[f'matriz{c}'] = np.array(m, dtype=object)

    print(c)

