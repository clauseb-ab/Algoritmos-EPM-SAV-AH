import random
import time
import array
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from deap import base, creator, tools, algorithms

from errores import *

def TSNE_TE(adjacency_matrix,distance_matrix,perplexity):
    # Aplicar t-SNE
    tsne = TSNE(n_components=1,metric='precomputed', perplexity=perplexity,learning_rate=1000,early_exaggeration=30,init='random')
    embedding_1d = tsne.fit_transform(distance_matrix)
    
    # Reducir la dimensionalidad de 2D a 1D usando PCA
#     pca = PCA(n_components=1)
#     embedding_1d = pca.fit_transform(embedding)
    # Ordenar los nodos según su coordenada en el embedding 1D
    sorted_indices = np.argsort(embedding_1d.flatten())

    # Crear una lista con el orden de los nodos
    ordered_nodes = sorted_indices.tolist()
    
    return ordered_nodes

def possible_orders(adjacency_matrix,n,k): #Funcion usada para encontrar ordenes iniciales para luego usarse en el algoritmo genetico
    orders=[]
    t_m=len(adjacency_matrix)
    ds=[t_m/2,t_m]
    distance_matrix = np.where(adjacency_matrix == 1, 1,
                            np.where(adjacency_matrix == 0, 0, 
                            np.where(adjacency_matrix == -1, ds[k], adjacency_matrix)))
    n_samples = distance_matrix.shape[0]
    perplexity=n_samples//10*2
    for i in range(n):
        order=TSNE_TE(adjacency_matrix,distance_matrix,perplexity)
        orders.append(order)
        #print(i)
    return orders

#Función de mutación (intercambio e inserción) Funcion de creacion propia usada en el algoritmo genetico
def mutSwapAndInsert(individual, indpb_swap=0.5, indpb_insert=0.5):
    """Aplica una mutación por intercambio o por inserción, asegurando que el individuo se mantenga válido."""
    size = len(individual)

    # Realizar intercambio
    if random.random() < indpb_swap:
        pos1, pos2 = random.sample(range(size), 2)
        individual[pos1], individual[pos2] = individual[pos2], individual[pos1]

    # Realizar inserción
    if random.random() < indpb_insert:
        pos1 = random.randint(0, size - 1)
        pos2 = random.randint(0, size - 1)
        value = individual.pop(pos1)

        if pos2 > pos1:
            individual.insert(pos2 - 1, value)
        else:
            individual.insert(pos2, value)

    return individual,

# Crear tipo de aptitud y tipo de individuo
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizar errores
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)  # Individuos son permutaciones de nodos

def custom_initial_population(adj_matrix, custom_individuals):
    population = []
    for individual in custom_individuals:
        pop_individual = creator.Individual(individual)
        population.append(pop_individual)
    return population

def eaSimple_datos(population, toolbox, cxpb, mutpb, ngen, stats=None,
                halloffame=None, verbose=__debug__, time_limit=600):
    """
    Algoritmo evolutivo simple con tiempo limitado.
    
    Args:
        time_limit: Tiempo límite en segundos para la ejecución del algoritmo
    """
    start_time = time.time()
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # Evaluar individuos con aptitud no válida
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        
    if halloffame is not None:
        halloffame.update(population)
    
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Comenzar el proceso generacional
    for gen in range(1, ngen + 1):
        # Verificar tiempo límite
        if time.time() - start_time > time_limit:
            #print(f"Tiempo límite de {time_limit} segundos alcanzado en generación {gen}")
            break
            
        # Seleccionar individuos de la siguiente generación
        offspring = toolbox.select(population, len(population))
        
        # Variar el grupo de individuos
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        
        # Evaluar individuos con aptitud no válida
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Actualizar el hall of fame con los individuos generados
        if halloffame is not None:
            halloffame.update(offspring)
            
        # Reemplazar la población actual por la descendencia
        population[:] = offspring
        
        # Registrar las estadísticas de la generación actual
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        
    return population, logbook

def genetic_algorithm_TSNE_V2(adj_matrix, ngen, custom_individuals=None, n=100, time_limit=600):
    """
    Algoritmo genético principal.
    
    Args:
        adj_matrix: Matriz de adyacencia del grafo
        ngen: Número de generaciones
        custom_individuals: Lista de individuos personalizados (opcional)
        n: Tamaño de la población si no se proporcionan individuos personalizados
        time_limit: Tiempo límite en segundos para la ejecución del algoritmo
    """
    num_nodes = len(adj_matrix)
    toolbox = base.Toolbox()
    
    # Configuración de la población inicial
    if custom_individuals is None:
        toolbox.register("indices", random.sample, range(num_nodes), num_nodes)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        population = toolbox.population(n)
    else:
        population = custom_initial_population(adj_matrix, custom_individuals)
    
    # Función de evaluación
    def evaluate(individual):
        return total_errors(individual, adj_matrix), #Si se quiere calcular otros errores aqui es donde se debe cambiar la funcion
    
    # Registrar operadores genéticos
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", mutSwapAndInsert, indpb_swap=0.5, indpb_insert=0.5)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("evaluate", evaluate)
    
    # Configurar parámetros
    cxpb = 0.7
    mutpb = 0.5
    
    # Configurar estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", min)
    
    # Configurar Hall of Fame
    halloffame = tools.HallOfFame(1)
    
    # Ejecutar el algoritmo genético con el time_limit especificado
    final_population, logbook = eaSimple_datos(
        population, 
        toolbox, 
        cxpb, 
        mutpb, 
        ngen, 
        stats, 
        halloffame, 
        verbose=False,
        time_limit=time_limit
    )
    
    best_individual = halloffame[0]
    return best_individual, evaluate(best_individual)[0]

