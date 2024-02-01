from braille_writer import toBraille, matrixPrinter, dotPlacer

s = input("What would you like to translate?\n\n")

brailleList = toBraille(s)

# create matrix of width
# len(brailleList)*2+(len(brailleList)-1)
# for the 2 columns each braill character
# requires, and the spaces between
cols = len(brailleList)*3 - 1
rows = 3
matrix = [[0 for i in range(cols)] for j in range(rows)]
filledMatrix = dotPlacer(matrix, brailleList)
print(filledMatrix)
matrixPrinter(matrix)
