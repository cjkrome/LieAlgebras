import math
from sympy import *
from copy import deepcopy
import numpy as np


# Object for holding the result of a bracket operation.
# Constructor parameters:
#   1. k - the X index of the result.
#   2. alpha - the alpha value for the result (may be a number or Sympy symbol).
# Example: Given the bracket [X1, X2] = X3, the bracket result object would be:
#   k = 3, alpha = 1
class BracketResult:
    def __init__(self, k, alpha):
        self.index = k
        self.alpha = alpha

    # Overloaded multiplication operator:
    # Makes a copy of the algebra, multiplies the alpha value, then returns the copy.
    def __mul__(self, other):
        bracketcopy = deepcopy(self)
        bracketcopy.alpha *= other
        return bracketcopy


# A lie algebra object.
# Constructor parameters:
#   1. dimension - the dimension of the algebra.
#   2. name (optional) - the named algebra to construct.
# Contains:
#   1. A dictionary of brackets (indexed by tuple).
#   2. A dimension
#   3. A d-value
#   4. An extension count
#   5. An array of Jacobi triples to test.
#   6. A dictionary of Jacobi test results (indexed by triple).
class LieAlgebra:
    def __init__(self, dimension, name=None):
        self.brackets = {}
        self.dimension = dimension
        self.d = 0
        self.extension = 0
        self.JacobiToTest = []
        self.JacobiTestResults = {}
        self.type = None

        # If "L" is the name provided, brackets [X1, Xi] = X(i+1) for (i=1,...,n-1) should be added.
        if name == "L":
            self.d = 2
            self.type = 'A'
            for j in range(2, dimension):
                self.AddBracket(1, j, j + 1)

        # j = 2
        # AddBracket(1, 2, 3)

    # Accepts i, j, and k as indices and then adds a new bracket to the lie algebra.
    # (Alpha is optional and defaults to 1)
    def AddBracket(self, i, j, k, alpha=1):
        res = BracketResult(k, alpha)
        self.brackets[i, j] = res

    # Accepts i, j, and k as eigenvalues, converts them to indices, and then uses the indices to add a bracket.
    def AddEigenvalueBracket(self, i, j, k, d):
        i = self.ConvertEigenvalueToIndex(i, d)
        j = self.ConvertEigenvalueToIndex(j, d)
        k = self.ConvertEigenvalueToIndex(k, d)

        # This format ensures correct Latex printing:
        alphatext = "alpha_" + str(i) + "," + str(j) + "^" + str(k) + ""
        self.AddBracket(i, j, k, Symbol(alphatext))

    # Performs the bracket operation [Xi, Xj]
    def Bracket(self, i, j):
        newAlpha = 1

        if type(i) == BracketResult:
            newAlpha *= i.alpha
            i = i.index
        if type(j) == BracketResult:
            newAlpha *= j.alpha
            j = j.index

        if (i, j) in self.brackets:
            return self.brackets[(i, j)] * newAlpha
        # If [Xi, Xj] is not found, check for [Xj, Xi] (and multiply the resulting alpha by -1 if found).
        elif (j, i) in self.brackets:
            return self.brackets[(j, i)] * (newAlpha * -1)
        else:
            return 0

    def ConvertIndexToEigenvalue(self, index, d=0):
        if d == 0:
            d = self.d
        return index + d - 2

    def ConvertEigenvalueToIndex(self, eigenval, d=0):
        if d == 0:
            d = self.d
        return eigenval - d + 2

    def PrintBrackets(self):
        for key1, key2 in sorted(self.brackets):
            res = self.brackets[(key1, key2)]

            # Doubling the bracket acts as an escape character
            # (i.e. '{{' becomes '{' in the final output),
            # and we need the keys to be inside of brackets
            # before being converted to Latex
            # so that multiple digits stick together inside of a subscript.
            bracketFormat = latex("[X_{{{}}},X_{{{}}}]".format(key1, key2))
            resultFormat = latex(res.alpha * Symbol("X" + str(res.index)))

            print("$${} = {}$$".format(bracketFormat, resultFormat))

    def CreateY(self):
        numTriples = len(self.brackets)
        I = np.identity(self.dimension)
        Y = np.zeros((numTriples, self.dimension))
        i = 0
        for key1, key2 in sorted(self.brackets):
            key3 = self.brackets[(key1, key2)].index
            Y[i] = I[key1-1] + I[key2-1] - I[key3-1]
            i = i + 1
        print(Y)
        rank = np.linalg.matrix_rank(Y)
        corank = numTriples - rank
        print('rank = {}, corank = {}'.format(rank, corank))
        """

            # Doubling the bracket acts as an escape character
            # (i.e. '{{' becomes '{' in the final output),
            # and we need the keys to be inside of brackets
            # before being converted to Latex
            # so that multiple digits stick together inside of a subscript.
            bracketFormat = latex("[X_{{{}}},X_{{{}}}]".format(key1, key2))
            resultFormat = latex(res.alpha * Symbol("X" + str(res.index)))

            print("$${} = {}$$".format(bracketFormat, resultFormat))
            """
    def PrintJacobiToTest(self):
        for triple in self.JacobiToTest:
            res = self.JacobiTestResults[triple]
            print("Jacobi results for {}: $${}$$".format(triple, latex(res)))

    def AddJacobiToTest(self, triple):
        self.JacobiToTest.append(triple)


# Find all of the non-trivial triples that need to be tested.
def TestAllJacobi(LA):
    for j in range(2, LA.dimension - 2):
        for k in range(j + 1, LA.dimension - 1):
            ej = LA.ConvertIndexToEigenvalue(index=j)
            ek = LA.ConvertIndexToEigenvalue(index=k)
            emax = LA.ConvertIndexToEigenvalue(LA.dimension)
            if ej + ek + 1 == emax:
                resultset = TestJacobi(LA, 1, j, k)
                if resultset is not False:
                    LA.JacobiToTest.append((1, j, k))
                    LA.JacobiTestResults[(1, j, k)] = resultset


# Test an individual triple to see if it is trivial.
def TestJacobi(LA, i, j, k):
    r1 = LA.Bracket(i, LA.Bracket(j, k))
    r2 = LA.Bracket(j, LA.Bracket(k, i))
    r3 = LA.Bracket(k, LA.Bracket(i, j))
    if r1 != 0 or r2 != 0 or r3 != 0:
        return Eq(GetEqTerm(r1) + GetEqTerm(r2) + GetEqTerm(r3))
        # return (r1, r2, r3)
    else:
        return False


# Convert a jacobi test result into an equation term.
def GetEqTerm(res):
    if type(res) == BracketResult:
        return res.alpha
               # * Symbol("X" + str(res.index))
    else:
        return 0


# Check if a 'd' value is valid for a given 'n'.
def IsValidD(n, d):
    if n - d % 2 == 0:
        return (n - 2) > d
    else:
        return (n - 1) > d


# Extends a lie algebra.
# If a d value is supplied, only a single lie algebra is returned.
# Otherwise, a list of lie algebras is returned (one for each valid d value).
def ExtendLieAlgebra(LA, d=0):
    NewLieList = []

    if d == 0:
        d = LA.d  # Start d value
        while IsValidD(LA.dimension, d):
            NewLieList.append(GenerateExtendedLA(LA, d))
            d += 1
        return NewLieList
    else:
        return GenerateExtendedLA(LA, d)


# Accepts a lie algebra and a d value, increments the dimension, and then adds the new brackets.
def GenerateExtendedLA(LA, d):
    n = LA.dimension
    n2 = n + 1  # The dimension of the new Lie Algebras

    LastValue = n + d - 1  # The last eigenvalue

    NewLieAlg = deepcopy(LA)
    NewLieAlg.extension += 1
    NewLieAlg.dimension += 1
    NewLieAlg.d = d
    NewLieAlg.AddBracket(1, n, n2)

    # Odd case
    if (n - d) % 2 != 0:
        CenterValue = int(LastValue / 2)

        for i in range(d, CenterValue):
            j = LastValue - i
            NewLieAlg.AddEigenvalueBracket(i, j, LastValue, d)

    # Even case
    else:
        CenterValue = int((LastValue - 1) / 2)

        for i in range(d, CenterValue + 1):
            j = LastValue - i
            NewLieAlg.AddEigenvalueBracket(i, j, LastValue, d)

    return NewLieAlg


# Accepts a lie algebra and finds the TypeB that can be made by extending it.
def FindNextDimensionTypeB(LA):
    n = LA.dimension
    d = LA.d
    NewLieAlg = deepcopy(LA)
    NewLieAlg.dimension += 1
    NewLieAlg.type = 'B'

    LastValue = 2 * d + n - 2
    CenterValue = math.floor(LastValue / 2)
    RangeEnd = CenterValue + 1 if ((n - d) % 2 != 0) else CenterValue

    for i in range(d, RangeEnd):
        j = LastValue - i
        NewLieAlg.AddEigenvalueBracket(i, j, LastValue, d)

    return NewLieAlg


def PrintFoundLieAlgebras(LAFound):
    # If only one LA was provided there is no need to loop.
    if type(LAFound) == LieAlgebra:
        PrintExtendedLA(LAFound)
        LAFound.CreateY()
    else:
        for LA in LAFound:
            PrintExtendedLA(LA)
            LA.CreateY()


def PrintExtendedLA(LA):
    TestAllJacobi(LA)
    sectionFormat = latex("m_{{{}}}({}, {})".format(str(LA.extension) + LA.type, LA.d, LA.dimension))
    print("\section{{{}}}".format(sectionFormat))
    LA.PrintBrackets()
    print("\nNon-trivial Jacobi Tests:")
    LA.PrintJacobiToTest()
    print("\n")


def RecursiveExtension(LA, depth, start=True):
    # For the first call, try all d values.
    if start is True:
        LAFound = ExtendLieAlgebra(LA)
        PrintFoundLieAlgebras(LAFound)

        for NewLA in LAFound:
            RecursiveExtension(NewLA, depth - 1, False)

    elif depth > 0:
        LAFound = ExtendLieAlgebra(LA, d=LA.d)
        PrintFoundLieAlgebras(LAFound)
        RecursiveExtension(LAFound, depth - 1, False)


L4 = LieAlgebra(name="L", dimension=4)

RecursiveExtension(LA=L4, depth=2)
