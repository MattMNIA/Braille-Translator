letterToBraille = {
    "a":[1],
    "b":[1,2],
    "c":[1,4],
    "d":[1,4,5],
    "e":[1,5],
    "f":[1,2,4],
    "g":[1,2,4,5],
    "h":[1,2,5],
    "i":[2,4],
    "j":[2,4,5],
    "k":[1,3],
    "l":[1,2,3],
    "m":[1,3,4],
    "n":[1,3,4,5],
    "o":[1,3,5],
    "p":[1,2,3,4],
    "q":[1,2,3,4,5],
    "r":[1,2,3,5],
    "s":[2,3,4],
    "t":[2,3,4,5],
    "u":[1,3,6],
    "v":[1,2,3,6],
    "w":[2,4,5,6],
    "x":[1,3,4,6],
    "y":[1,3,4,5,6],
    "z":[1,3,5,6],
}
numberToBraille = {
    "1":[1],
    "2":[1,2],
    "3":[1,4],
    "4":[1,4,5],
    "5":[1,5],
    "6":[1,2,4],
    "7":[1,2,4,5],
    "8":[1,2,5],
    "9":[2,4],
    "0":[2,4,5],
}
symbolToBraille = {
    ",":[1],
    ";":[2,3],
    ":":[2,5],
    ".":[2,5,6],
    "?":[2,3,6],
    "!":[2,3,5],
    "'":[3],
    '"':[3,0,2,3,5,6],
    "(":[5,0,1,2,6],
    ")":[5,0,3,4,5],
    "/":[4,5,6,0,3,4],
    " ": []
}
arrToCoord = {
    1:(0,0),
    2:(1,0),
    3:(2,0),
    4:(0,1),
    5:(1,1),
    6:(2,1),
}

def toBraille(s):
    """
    Coverts each character into
    an array of braille dots
    param: s: string to be converted
    return: braille: a list of braille dot arrays
    """
    s = s.lower()
    braille = list()
    prevNum = False
    for c in s:
        if c == "&":
            break
        if(c.isalpha()):
            if prevNum:
                braille.append([2,3])
            braille.append(letterToBraille[c])
            prevNum = False
        elif(c.isnumeric()):
            if not prevNum:
                braille.append([3,4,5,6])
            braille.append(numberToBraille[c])
            prevNum = True
        else:
            if 0 in symbolToBraille[c]:
                lists = dotSeparator(symbolToBraille[c])
                braille.append(lists[0])
                braille.append(lists[1])
            else:
                braille.append(symbolToBraille[c])
    return braille
def dotSeparator(arr):
    """
    Separates both parts of a symbol
    param: arr: array to be split
    return: tuple(list1,list2)
    """
    list1 = list()
    list2 = list()
    zeroFound = False
    for n in arr:
        if n == 0:
            zeroFound = True
        elif zeroFound:
            list2.append(n)
        else:
            list1.append(n)
    return (list1,list2)
def dotPlacer(matrix, brailleList):
    """
    decodes braille list to put
    dots where they belong in 
    the matrix
    param: matrix: 2d array
    param: brailleList: list of letters in braille code
    return: filled in matrix
    """
    for i in range(len(brailleList)):
        matrixInc = i*3
        for n in brailleList[i]:
            coord = arrToCoord[n]
            coordTransform = [coord[0],coord[1]+matrixInc]
            matrix[coordTransform[0]][coordTransform[1]] = 1
    return matrix
def matrixPrinter(matrix):
    """
    Prints results with spaces and astrixes
    param: matrix: 2d array filled with 1s and 0s
    return: void
    """
    print()
    for arr in matrix:
        s = ""
        for i in range(len(arr)):
            if arr[i] == 0:
                s = s + " "
            elif arr[i] == 1:
                s = s + "*"
            if (i+1) % 3 == 0 and i>0:
                s = s + "|"
        print(s + "\n")
