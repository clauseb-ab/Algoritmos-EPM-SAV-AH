import numpy as np
import random
import time
from itertools import chain
from typing import Dict, Set, List, Tuple

from errores import *
#Algoritmo constructivo

def calculate_errors(pi, adj_matrix):
    n = len(pi)
    errors = np.zeros(n)
    
    for x in range(n):
        sum_ = 0
        
        row = list(range(n))  #Lista para explorar las filas
        neg = 0              # contador aristas negativas
        pos = 0              # contador arista positiva
        
        #Exploracion hacia la derecha
        for y in row[x:]:
            if adj_matrix[pi[x]][pi[y]] == 1:
                if neg > 0:  # se cuenta la arista positiva solo si hay aristas negativas
                    pos += 1
            elif adj_matrix[pi[x]][pi[y]] == -1:
                if neg == 0:  # Se suma uno si aparece la primera arista negativa 
                    neg = 1
                else:  # Si hay una nueva arista negativa
                    sum_ += neg * pos
                    neg += 1
                    pos = 0
        
        # Si el ultimo vertice hacia la derecha tiene arista positiva 
        if neg > 0 and pos > 0:
            sum_ += neg * pos
            
        # Reiniciar contadores
        neg = 0
        pos = 0
        
        # Invertir sentido de exploracion, exploracion hacia la izquierda
        for z in reversed(row[:x]):
            if adj_matrix[pi[x]][pi[z]] == 1:
                if neg > 0:  # se cuenta la arista positiva solo si hay aristas negativas
                    pos += 1
            elif adj_matrix[pi[x]][pi[z]] == -1:
                if neg == 0:  # Se suma uno si aparece la primera arista negativa 
                    neg = 1
                else:  # Si hay una nueva arista negativa
                    sum_ += neg * pos
                    neg += 1
                    pos = 0
        
        # Si el ultimo vertice hacia la derecha tiene arista positiva
        if neg > 0 and pos > 0:
            sum_ += neg * pos
        
        #se le asigna el numero de error al vertice
        errors[x] = sum_
    
    return errors

def total_errors(pi, adj_matrix):
    errors = calculate_errors(pi,  adj_matrix)
    return np.sum(errors)

def remove_negative_edges(graph: np.ndarray) -> np.ndarray:
    """
    Convierte los valores negativos en la matriz de adyacencia a 0.
    """
    return np.maximum(graph, 0)

def get_adjacency_dict(graph: np.ndarray) -> Dict[int, Set[int]]:
    """
    Genera un diccionario de adyacencia a partir de una matriz.
    """
    return {i: set(np.where(row > 0)[0]) for i, row in enumerate(graph)}

def bron_kerbosch(r: Set[int], p: Set[int], x: Set[int], graph: Dict[int, Set[int]], 
                  cliques: List[Set[int]], start_time: float, time_limit: int) -> None:
    """
    Implementación del algoritmo Bron-Kerbosch con pivoting.
    """
    if time.time() - start_time > time_limit:
        return
        
    if not p and not x:
        if len(r) > 1:
            cliques.append(r)
        return
        
    pivot = max((len(graph[v] & p), v) for v in p | x)[1]
    for v in p - graph[pivot]:
        new_r = r | {v}
        new_p = p & graph[v]
        new_x = x & graph[v]
        bron_kerbosch(new_r, new_p, new_x, graph, cliques, start_time, time_limit)
        p = p - {v}
        x = x | {v}

def find_cliques(graph: np.ndarray, time_limit: int = 5) -> Tuple[List[Set[int]], Dict[int, Set[int]]]:
    """
    Encuentra todas las cliques en el grafo.
    """
    adj_dict = get_adjacency_dict(graph)
    vertices = set(adj_dict.keys())
    cliques = []
    start_time = time.time()
    
    # Primero encontrar cliques mayores
    bron_kerbosch(set(), vertices, set(), adj_dict, cliques, start_time, time_limit)
    
    # Asegurar que todos los vértices están incluidos
    vertices_in_cliques = set().union(*cliques) if cliques else set()
    remaining_vertices = set(range(len(graph)))
    
    # Añadir vértices sueltos como cliques unitarias
    for v in remaining_vertices:
        cliques.append({v})
    
    # Ordenar cliques por tamaño
    cliques.sort(key=len, reverse=True)
    #print(cliques[:10])
    return cliques, adj_dict

def create_projected_matrix(ordering: List[int], adj_dict: Dict[int, Set[int]]) -> np.ndarray:
    """
    Crea una matriz de adyacencia proyectada basada en un ordenamiento.
    """
    n = len(ordering)
    matrix = np.zeros((n, n), dtype=int)
    
    for i, v1 in enumerate(ordering):
        for j, v2 in enumerate(ordering[i+1:], i+1):
            if v2 in adj_dict[v1]:
                matrix[i, j] = matrix[j, i] = 1
                
    return matrix

def evaluate_insertion(ordering: List[int], adj_dict: Dict[int, Set[int]]) -> float:
    """
    Evalúa la calidad de un ordenamiento dado.
    """
    try:
        projected_matrix = create_projected_matrix(ordering, adj_dict)
        return total_errors(list(range(len(ordering))), projected_matrix)
    except Exception as e:
        print(f"Error en evaluate_insertion: {e}")
        return float('inf')

def insert_clique(current_ordering: List[List[int]], adj_dict: Dict[int, Set[int]], 
                clique: Set[int]) -> List[List[int]]:
    """
    Inserta una clique en la mejor posición posible.
    """
    n = len(current_ordering)
    best_position = 0
    best_score = float('inf')
    clique_list = list(clique)
    
    for i in range(n + 1):
        new_ordering_list = current_ordering[:i] + [clique_list] + current_ordering[i:]
        new_ordering = list(chain.from_iterable(new_ordering_list))
        score = evaluate_insertion(new_ordering, adj_dict)
        
        if score < best_score:
            best_score = score
            best_position = i
            
    return current_ordering[:best_position] + [clique_list] + current_ordering[best_position:]

def constructive(graph: np.ndarray) -> List[int]:
    """
    Construye una solución completa usando el algoritmo constructivo.
    """
    n = len(graph)
    # transformar el grafo a grafo positivo
    processed_graph = remove_negative_edges(graph.copy())
    
    # Encontrar cliques
    cliques, adj_dict = find_cliques(processed_graph)
    
    # Construir solución
    ordering: List[List[int]] = []
    remaining_cliques = cliques.copy()
    
    while remaining_cliques:
        current_clique = remaining_cliques.pop(0)
        ordering = insert_clique(ordering, adj_dict, current_clique)
        
        # Filtrar cliques que intersectan con la actual
        remaining_cliques = [
            c for c in remaining_cliques 
            if not current_clique.intersection(c)
        ]
    
    # Aplanar el ordenamiento final
    final_ordering = list(chain.from_iterable(ordering))
    
    return final_ordering

#Algoritmo BVNS
def insert(phi, i, j):
    """Insertar un elemento de la posición i a la posición j"""
    n = phi[i]
    phip = phi.copy()
    phip.pop(i)
    phip.insert(j, n)
    return phip

def interchange(phi, i, j):
    """Intercambiar elementos entre posiciones i y j"""
    if i < j:
        phi_new = insert(insert(phi, i, j), j-1, i)
    elif i > j:
        phi_new = insert(insert(phi, i, j), j+1, i)
    return phi_new

def shake(phi, k):
    """Perturbar la solución phi k veces"""
    for _ in range(k):
        i = random.randint(0, len(phi) - 1)
        j = random.randint(0, len(phi) - 1)
        while i == j:
            j = random.randint(0, len(phi) - 1)
        phi = interchange(phi, i, j)
    return phi

def LocalSearch(pi, adj_matrix):
    """ Búsqueda local con inserción usando primera mejora """
    n = len(pi)
    
    # Elegir dos índices aleatorios distintos
    i = random.randint(0, n - 1)
    j = random.randint(0, n - 1)
    while i == j:
        j = random.randint(0, n - 1)

    # Calcular error solo para vértices afectados (i y sus vecinos)
    affected_vertices = set([pi[i]])  # El vértice movido
    affected_vertices.update(get_neighbors(pi[i], adj_matrix))  # Sus vecinos
    
    # Calcular errores antes del cambio
    old_errors = sum(calculate_errors_local(pi, adj_matrix, v) for v in affected_vertices)

    # Aplicar el movimiento de inserción
    new_pi = insert(pi, i, j)

    # Calcular errores después del cambio
    new_errors = sum(calculate_errors_local(new_pi, adj_matrix, v) for v in affected_vertices)

    # Aceptar el cambio si mejora la solución
    if new_errors < old_errors:
        return new_pi,new_errors  # Se acepta el movimiento
    else:
        return pi,old_errors  # Se mantiene la solución original

def get_neighbors(v, adj_matrix):
    """ Obtener los vecinos de un vértice """
    return [i for i in range(len(adj_matrix)) if adj_matrix[v][i] != 0]

def calculate_errors_local(pi, adj_matrix, x):
    """ Calcular el error solo para un vértice específico """
    n = len(pi)
    sum_ = 0
    neg = 0  # contador aristas negativas
    pos = 0  # contador aristas positivas

    # Exploración hacia la derecha
    for y in range(x + 1, n):
        if adj_matrix[pi[x]][pi[y]] == 1:
            if neg > 0:
                pos += 1
        elif adj_matrix[pi[x]][pi[y]] == -1:
            if neg == 0:
                neg = 1
            else:
                sum_ += neg * pos
                neg += 1
                pos = 0

    if neg > 0 and pos > 0:
        sum_ += neg * pos

    # Exploración hacia la izquierda
    neg = 0
    pos = 0
    for z in reversed(range(x)):
        if adj_matrix[pi[x]][pi[z]] == 1:
            if neg > 0:
                pos += 1
        elif adj_matrix[pi[x]][pi[z]] == -1:
            if neg == 0:
                neg = 1
            else:
                sum_ += neg * pos
                neg += 1
                pos = 0

    if neg > 0 and pos > 0:
        sum_ += neg * pos

    return sum_


def NeighborhoodChange_t_e(phi, phi_2, k, adj_matrix,t_e,t_e2):
    """Decidir si se acepta la nueva solución"""
    error = t_e
    error2 = total_errors(phi_2, adj_matrix)
    
    if error2 < error:
        return phi_2, 1, error2
    else:
        return phi, k+1, error

def BVNS_t_e(phi, k_max, t_max, adj_matrix):
    """Algoritmo BVNS (Basic Variable Neighborhood Search)"""
    t = 0
    t_e=total_errors(phi, adj_matrix)
    while t < t_max:
        inicio = time.time()
        k = 1
        while k <= k_max:
            # Perturbar solución actual
            phi_1 = shake(phi, k)
            
            # Aplicar búsqueda local
            phi_2,t_e2 = LocalSearch(phi_1, adj_matrix)
            
            # Decidir si cambiar la solución
            phi, k, t_e = NeighborhoodChange_t_e(phi, phi_2, k, adj_matrix,t_e,t_e2)
        
        fin = time.time()
        t += (fin - inicio)
    
    
    return phi, t_e


def initial_solution(n):
    """Generar solución inicial aleatoria"""
    if n <= 0:
        raise ValueError("El tamaño de la solución debe ser positivo")
    return list(range(n))


# Ejemplo de uso
def run_BVNS(adj_matrix, k_max=5, t_max=10):
    """
    Ejecutar BVNS en una matriz de adyacencia
    
    Parámetros:
    - adj_matrix: Matriz de adyacencia del grafo
    - k_max: Máximo número de sacudidas
    - t_max: Tiempo máximo de ejecución
    """
    # Convertir a numpy array si no lo está
    adj_matrix = np.array(adj_matrix)
    
    # Solución inicial (permutación identidad)
    ini_time=time.time()
    initial_phi = constructive(adj_matrix)
    #initial_phi = initial_solution(len(adj_matrix))
    final_time=time.time()
    t_max=t_max-(final_time-ini_time)
    #Ejecutar BVNS
    best_phi, best_error = BVNS_t_e(
        initial_phi, 
        k_max=k_max, 
        t_max=t_max, 
        adj_matrix=adj_matrix
    )
    
    return best_phi, best_error