import numpy as np
import math

def getTextToNumberTable():
    table = {chr(i+64): i for i in range(1,27)}
    table[' '] = 0
    return table


def getNumberToTextTable():
    table = {i: chr(i+64) for i in range(1,27)}
    table[0] = ' '
    return table


def getInverseNumber(num):
    num = int(num)
    for i in range(1,27):
        if (i*num) % 27 == 1:
            return i
    return -1


def getCofactorMatrix(matrix):
    C = np.zeros(matrix.shape)
    nrows, ncols = C.shape
    for row in range(nrows):
        for col in range(ncols):
            minor = matrix[np.array(list(range(row))+list(range(row+1, nrows)))[:, np.newaxis],
                           np.array(list(range(col))+list(range(col+1, ncols)))]
            C[row, col] = (-1)**(row+col) * np.linalg.det(minor)
    return np.round(C)


def splitMessage(message, n):
    numSplits = len(message)//n
    vector = []
    currentSplit = []
    for ind, c in enumerate(message):
        currentSplit.append(c)
        if len(currentSplit) == n or ind == len(message)-1:
            vector.append(currentSplit[:])
            currentSplit.clear()
    for split in vector:
        if len(split) != n:
            diff = n - len(split)
            for i in range(diff):
                split.append(' ')
    return vector


def convertSplitMessage(vector, table):
    for i, split in enumerate(vector):
        for j, char in enumerate(split):
            vector[i][j] = table[vector[i][j]]
    return vector


def transformSplitMessage(vector, k):
    vector = np.array(vector)
    for i, split in enumerate(vector):
        temp = split
        vector[i] = np.dot(k, temp)
    return np.array(vector).T


def convertBackToText(vector, table):
    vector = np.array(vector).T
    result = np.empty(vector.shape, dtype=str)
    for i, split in enumerate(vector):
        for j, num in enumerate(split):
            result[i][j] = table[num]
    string = ""
    for row in range(len(result)):
        for column in range(len(result[0])):
            string += result[row][column]
    return string


def main():
    ##### CONFIG #####
    # n = 2
    n = 4
    message = "PLEASE USE THE OTHER DOOR"
    textToNumber = getTextToNumberTable()
    numberToText = getNumberToTextTable()
    # k = np.matrix([
    #     [3, 5],
    #     [1, 6]
    # ])
    k = np.matrix([
        [2, 3, 5, 7],
        [11, 13, 17, 19],
        [23, 29, 31, 37],
        [41, 43, 47, 53]
    ])
    det_k = np.linalg.det(k)
    print("-" * 20)
    print(f"Key matrix:\n{k}")
    print(f"Determinant of k: {det_k}")
    print("-" * 20)

    ##### ENCRYPT #####
    split_message = splitMessage(message, n)
    converted_split_message = convertSplitMessage([row[:] for row in split_message], textToNumber)
    transformed_split_message = transformSplitMessage([row[:] for row in converted_split_message], k)
    mod_split_message = transformed_split_message % 27
    encryptedMessage = convertBackToText([row[:] for row in mod_split_message], numberToText)
    print(f"Message to encrypt: {message}")
    print(f"Split message:\n{split_message}")
    print(f"Converted split message:\n{converted_split_message}")
    print(f"Transformed split message:\n{transformed_split_message}")
    print(f"Mod split message:\n{mod_split_message}")
    print(f"Encrypted message:\n{encryptedMessage}")
    print("-" * 20)

    ##### DECRYPT #####
    inverse_det_k = getInverseNumber(det_k)
    cofactor_k = getCofactorMatrix(k)
    adjoint_k = cofactor_k.T
    inverse_k = adjoint_k * (1/det_k)
    inverse_k_in_mod_27 = (inverse_det_k * adjoint_k) % 27
    split_encrypted_message = splitMessage(encryptedMessage, n)
    converted_split_encrypted_message = convertSplitMessage([row[:] for row in split_encrypted_message], textToNumber)
    transformed_split_encrypted_message = transformSplitMessage([row[:] for row in converted_split_encrypted_message], inverse_k_in_mod_27)
    mod_split_encrypted_message = transformed_split_encrypted_message % 27
    decryptedMessage = convertBackToText([row[:] for row in mod_split_encrypted_message], numberToText)
    print(f"Inverse of determinant of k: {inverse_det_k}")
    print(f"Inverse of k in mod 27:\n{inverse_k_in_mod_27}")
    print(f"Message to decrypt:\n{encryptedMessage}")
    print(f"Split encrypted message:\n{split_encrypted_message}")
    print(f"Converted split encrypted message:\n{converted_split_encrypted_message}")
    print(f"Transformed split encrypted message:\n{transformed_split_encrypted_message}")
    print(f"Mod split encrypted message:\n{mod_split_encrypted_message}")
    print(f"Decrypted message:\n{decryptedMessage}")
    print("-" * 20)


if __name__ == "__main__":
    main()
