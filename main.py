import numpy as np

def cria_linhas(lines):
    matrices = []
    current_matrix = []
    for line in lines:
        row = list(map(int, line.strip().split()))
        if len(row) == 0:
            if current_matrix:
                # Transformar a última linha da matriz em uma coluna
                last_row = current_matrix[-1]
                last_column = np.array(last_row).reshape(len(last_row), 1)
                current_matrix = current_matrix[:-1]  # Remover a última linha da matriz
                current_matrix = [np.append(row, col) for row, col in zip(current_matrix, last_column)]
                matrices.append(current_matrix)
                current_matrix = []
        else:
            current_matrix.append(row)

    if current_matrix:
        matrices.append(current_matrix)

    return [np.array(matrix) for matrix in matrices]

def lendoarquivo(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    return cria_linhas(lines)

arquivo = 'teste.txt'
matriz = lendoarquivo(arquivo)


def gaussiana_pivoteamento(matriz):
    matriz = matriz.astype(np.float64)
    linhas, colunas = matriz.shape
    print("Matriz:")
    print(matriz)

    if colunas <= linhas:
        print("Está faltando uma coluna.")
        return

    for i in range(linhas - 1):
        pivot_row = i
        for k in range(i + 1, linhas):
            if abs(matriz[k, i]) > abs(matriz[pivot_row, i]):
                pivot_row = k

        matriz[[i, pivot_row], :] = matriz[[pivot_row, i], :]
        fator_diagonal = matriz[i, i]
        matriz[i, :] /= fator_diagonal

        for j in range(i + 1, linhas):
            fator = matriz[j, i]
            matriz[j, :] -= fator * matriz[i, :]

        print(f"Matriz após a {i + 1}ª iteração:")
        np.set_printoptions(suppress=True, precision=2)
        print(np.where(np.isclose(matriz, 0), 0, matriz))

    matriz[-1, :] /= matriz[-1, -2]

    print(f"Matriz após a {linhas}ª iteração:")
    np.set_printoptions(suppress=True, precision=2)
    print(np.where(np.isclose(matriz, 0), 0, matriz))

    # Vetor solução
    vetor_b = matriz[:, -1]
    matriz_sistema = matriz[:, :-1]
    solucao = np.linalg.solve(matriz_sistema, vetor_b)
    print("Vetor solução:")
    print(solucao)

def LU_pivoteamento(matriz):
    n = matriz.shape[0]

    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):

        max_row = i
        for k in range(i + 1, n):
            if abs(matriz[k, i]) > abs(matriz[max_row, i]):
                max_row = k


        matriz[[i, max_row]] = matriz[[max_row, i]]


        for j in range(i, n):
            soma = 0
            for k in range(i):
                soma += L[i, k] * U[k, j]
            U[i, j] = (matriz[i, j] - soma)

        for j in range(i + 1, n):
            soma = 0
            for k in range(i):
                soma += L[j, k] * U[k, i]
            if abs(U[i, i]) < 1e-8:
                U[i, i] = 1e-8
            L[j, i] = (matriz[j, i] - soma) / U[i, i]

        L[i, i] = 1

    b = matriz[:, -1]

    #Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]

    #Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]

    print("Matriz L:")
    print(L)

    print("\nMatriz U:")
    print(U)

    print("\nSolução de y:")
    print(y)

    print("\nSolução de x:")
    print(x)

def jacobi(A, max_iteracoes):
    b = A[:, -1]
    A = np.delete(A, -1, axis=1)
    n = len(A)
    x = np.zeros(n)
    gap = []

    for itr in range(max_iteracoes):
        x_new = np.zeros(n)
        for i in range(n):
            s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
            if A[i, i] != 0:
                x_new[i] = (b[i] - s) / A[i, i]
            else:
                x_new[i] = 0

        # Calcular o vetor gap (Ax - b)
        gap = np.dot(A, x_new) - b

        # Verificar se a solução é exata
        if np.allclose(np.dot(A, x_new), b):
            print("A solução é exata.")
            print(f"Iteração {itr + 1}:")
            print("x:", x_new)
            print("Gap:", gap)
            print("\n")
            break
        else:
            print("O valor é aproximado")

        x = x_new

        print(f"Iteração {itr + 1}:")
        print("x:", x_new)
        print("Gap:", gap)
        print("\n")

    return x





