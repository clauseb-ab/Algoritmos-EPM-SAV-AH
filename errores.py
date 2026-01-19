import numpy as np

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

def max_errors(pi, adj_matrix):
    errors = calculate_errors(pi, adj_matrix)
    return np.amax(errors)

def num_vert_errors(pi,  adj_matrix):
    errors = calculate_errors(pi, adj_matrix)
    return np.sum(errors > 0)

def errors(pi, adj_matrix):
    errors = calculate_errors(pi,  adj_matrix)
    return np.sum(errors),np.amax(errors),np.sum(errors > 0)