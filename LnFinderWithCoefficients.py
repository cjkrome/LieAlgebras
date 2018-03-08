import math
from sympy import *
from copy import deepcopy
import numpy as np
import scipy

#------------------------------------------------------------------------------
# Class Restrict
#------------------------------------------------------------------------------

# Utility class
class Restrict:
    def __init__(self, d, dimension, extension, type):
        self.d = d
        self.dimension = dimension
        self.extension = extension
        self.type = type

#------------------------------------------------------------------------------
# Class BracketResult
#------------------------------------------------------------------------------

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


#------------------------------------------------------------------------------
# Class LieAlgebra
#------------------------------------------------------------------------------

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
        self.JacobiTestResults2 = {}
        self.type = None

        # If "L" is the name provided, brackets [X1, Xi] = X(i+1) for (i=1,...,n-1) should be added.
        if name == "L":
            self.d = 2
            self.type = 'A'
            for j in range(2, dimension):
                self.AddBracket(1, j, j + 1)

        # j = 2
        # AddBracket(1, 2, 3)

    # r = restrictions
    def matches(self, r):
        if r == None:
            return True
        return (self.d == r.d and
                self.dimension == r.dimension and
                (r.extension == None or self.extension == r.extension) and
                self.type == r.type)

    # Accepts i, j, and k as indices and then adds a new bracket to the lie algebra.
    # (Alpha is optional and defaults to 1)
    def AddBracket(self, i, j, k, alpha=1):
        res = BracketResult(k, alpha)
        self.brackets[i, j] = res

    # Accepts i, j, and k as eigenvalues, converts them to indices, and then uses the indices to add a bracket.
    def AddEigenvalueBracket(self, i, j, k, d, n, extType):
        i = self.ConvertEigenvalueToIndex(i, d, n, extType)
        j = self.ConvertEigenvalueToIndex(j, d, n, extType)
        k = self.ConvertEigenvalueToIndex(k, d, n, extType)

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

    def ConvertEigenvalueToIndex(self, eigenval, d, n, extType):
        if d == 0:
            # if d is zero, then set it to self.d
            raise "d == 0"
        if eigenval == 1:
            raise "eigenval == 1"
        if extType == 'A':
            return eigenval - d + 2
        else:
            if eigenval < n + 2 * d - 3:
                return eigenval - d + 2
            return n+1
        
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


        # Testing out groebner bases to solve Jacobi equations
    def TestJacobiGroebner(self):
        name2sym = { 'alpha_2,5^8' : 'x5' }

        #print("Testing groebner")
        for triple in self.JacobiToTest:
            res = self.JacobiTestResults2[triple]
            print("triple")
            for r in res:
                if type(r) == Mul:
                    first, second = r.as_two_terms()
                    print(name2sym.get(first, first), name2sym.get(second.name, second))
                    print(type(second))
            #print("Jacobi results for {}: $${}$$".format(triple, latex(res)))

    def CreateY(self):
        numTriples = len(self.brackets)
        I = np.identity(self.dimension)
        Y = np.zeros((numTriples, self.dimension))
        i = 0
        for key1, key2 in sorted(self.brackets):
            # key1 and key2 are indices
            key3 = self.brackets[(key1, key2)].index
            #print("{}, {}, {}, {}, {}".format(key1, key2, key3, self.d, self.dimension))
            Y[i] = I[key1-1] + I[key2-1] - I[key3-1]
            i = i + 1
        #print(Y)
        rank = np.linalg.matrix_rank(Y)
        corank = numTriples - rank
        #print('rank = {}, corank = {}'.format(rank, corank))
        return Y
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


#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------

# Find all of the non-trivial triples that need to be tested.
def TestAllJacobi(LA):
    d = LA.d
    n = LA.dimension
    #if d < (n-4)/2:
#    if False:
#        for j in range(2, LA.dimension - 2):
#            for k in range(j + 1, LA.dimension - 1):
#                ej = LA.ConvertIndexToEigenvalue(index=j)
#                ek = LA.ConvertIndexToEigenvalue(index=k)
#                emax = LA.ConvertIndexToEigenvalue(LA.dimension)
#                if ej + ek + 1 == emax:
#                    resultset = TestJacobi(LA, 1, j, k)
#                    if resultset is not False:
#                        LA.JacobiToTest.append((1, j, k))
#                        LA.JacobiTestResults[(1, j, k)] = resultset
    #else:
    #    if LA.type == "A":
    #        JacobiTestsFromY(LA)
                
    #    msg = "TestAllJacobi condition not met: d={} n={}".format(d, n)
    #    print(msg)

    if LA.type == "A":
        JacobiTestsFromY(LA)
                    

def JacobiTestsFromY(LA):
    Y = LA.CreateY()
    #print("Y = \n{}".format(Y))
    U = np.dot(Y, Y.transpose())
    #print("U = \n{}".format(U))
    negOnes = np.where(U == -1)
    #print(negOnes[0])
    allIndices = []
    for i in range(len(negOnes[0])):
        idx0 = negOnes[0][i]
        idx1 = negOnes[1][i]
        #print("Y[{}] = {}".format(idx0, Y[idx0]))
        sum = np.add(Y[idx0], Y[idx1])
        #print(sum)
        indices = np.where(sum == 1)
        #print(indices)
        isIn = False
        for i in allIndices:
            isIn = isIn or np.array_equal(indices, i)
        if not isIn:
            allIndices.append(indices)
    #print(allIndices)
    LA.JacobiToTest = []
    for indices in allIndices:
        i = indices[0][0]+1
        j = indices[0][1]+1
        k = indices[0][2]+1
        resultset = TestJacobi(LA, i, j, k)
        resultset2 = TestJacobi2(LA, i, j, k)
        #print(resultset)
        if resultset is not False:
            LA.JacobiToTest.append((i, j, k))
            #print("appending {} len = {}".format((i,j,k), len(LA.JacobiToTest)))
            LA.JacobiTestResults[(i, j, k)] = resultset
            LA.JacobiTestResults2[(i, j, k)] = resultset2


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
# Test an individual triple to see if it is trivial.
def TestJacobi2(LA, i, j, k):
    r1 = LA.Bracket(i, LA.Bracket(j, k))
    r2 = LA.Bracket(j, LA.Bracket(k, i))
    r3 = LA.Bracket(k, LA.Bracket(i, j))
    if r1 != 0 or r2 != 0 or r3 != 0:
        return (GetEqTerm(r1), GetEqTerm(r2), GetEqTerm(r3))
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
    #print("***** valid " + str(n) + " " + str(d))
    if n - d % 2 == 0:
        #return (n - 2) > d
        valid = (n - 2) > d
    else:
        #return (n - 1) > d
        valid = (n - 1) > d
    #print("isValidD returning {}".format(valid))
    return valid


# Accepts a lie algebra and a d value, increments the dimension, and then adds the new brackets.
def GenerateExtendedLA(LA, d, extType):
    n = LA.dimension
    n2 = n + 1  # The dimension of the new Lie Algebras

    LastValue = n + d - 1  # The last eigenvalue
    if extType == 'B':
        LastValue = n + 2*d - 2

    NewLieAlg = deepcopy(LA)
    NewLieAlg.extension += 1
    NewLieAlg.dimension += 1
    NewLieAlg.d = d
    NewLieAlg.type = extType
    if extType == 'A':
        NewLieAlg.AddBracket(1, n, n2)
    else:
        NewLieAlg.AddBracket(2, n, n2)

    # Odd case
    startValue = d
    if extType == 'B':
        startValue = d+1
    if (n - d) % 2 != 0:
        CenterValue = int(LastValue / 2)

        for i in range(startValue, CenterValue):
            j = LastValue - i
            NewLieAlg.AddEigenvalueBracket(i, j, LastValue, d, n, extType)

    # Even case
    else:
        CenterValue = int((LastValue - 1) / 2)

        for i in range(startValue, CenterValue + 1):
            j = LastValue - i
            NewLieAlg.AddEigenvalueBracket(i, j, LastValue, d, n, extType)

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
        NewLieAlg.AddEigenvalueBracket(i, j, LastValue, d, n, extType)

    return NewLieAlg


def PrintFoundLieAlgebras(LAFound, restrict=None):
    # If only one LA was provided there is no need to loop.
    if type(LAFound) == LieAlgebra:
        PrintExtendedLA(LAFound, restrict)
        #Y = LAFound.CreateY()
        #print("Y = {}".format(Y))
    else:
        for LA in LAFound:
            PrintExtendedLA(LA, restrict)
            #LA.CreateY()

def null(A, eps=1e-15):
    u, s, vh = np.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return np.transpose(null_space)

def PrintExtendedLA(LA, restrict=None):
    if restrict == None or (LA.matches(restrict)):
        print("Testing Jacobi: {}, {}".format(LA.d, LA.dimension));
        TestAllJacobi(LA)
        print("Printing Jacobi: {}, {}".format(LA.d, LA.dimension));
        sectionFormat = latex("m_{{{}}}({}, {})".format(str(LA.extension) + LA.type, LA.d, LA.dimension))
        print("\section{{{}}}".format(sectionFormat))
        LA.PrintBrackets()
        print("\nNon-trivial Jacobi Tests:")
        #TestAllJacobi(LA)
        """
        Y = LA.CreateY()
        print("Y = \n{}".format(Y))
        nullY = null(Y)
        print("null(Y) = \n{}".format(nullY))
        U = np.dot(Y, Y.transpose())
        print("U = \n{}".format(U))
        m = Y.shape[0]
        print("Rank = {}, m = {}".format(np.linalg.matrix_rank(U), m))
        print("Solving Uv = [1 1 ... 1]'")
        ones = np.ones((m,1))
        #print("Ones = {}".format(ones))
        v = np.linalg.solve(U, ones)
        print("v = \n{}".format(v))
        """
        LA.PrintJacobiToTest()
        print("\n")

        #LA.TestJacobiGroebner()
    

# Extends a lie algebra, checking for all possible d values.
def FirstExtendLieAlgebra(LA):
    NewLieList = []
    if LA.type == 'B':
        return NewLieList

    if LA.dimension % 2 == 0:
        d = 2
    else:
        d = 3
    while IsValidD(LA.dimension, d):
        newLA = GenerateExtendedLA(LA, d, extType='A')
        NewLieList.append(newLA)
        # Generate extended type B LA from newLA
        newLA_B = GenerateExtendedLA(newLA, newLA.d, extType='B')
        NewLieList.append(newLA_B)
        d += 2
    return NewLieList

# Extends a lie algebra using the d value of LA.
def ExtendLieAlgebra(LA):
    NewLieList = []
    if LA.type == 'B':
        return NewLieList

    d = LA.d # start d value
    newLA = GenerateExtendedLA(LA, d, extType='A')
    NewLieList.append(newLA)
    # Generate extended type B LA from newLA
    newLA_B = GenerateExtendedLA(newLA, newLA.d, extType='B')
    NewLieList.append(newLA_B)
    return NewLieList

def RecursiveExtension(LA, depth, output=True, restrict_output=None):
    ret = []
    if depth > 0:
        LAFound = ExtendLieAlgebra(LA)
        ret.extend(LAFound)
        if output:
            PrintFoundLieAlgebras(LAFound, restrict_output)
        for NewLA in LAFound:
            ret.extend(RecursiveExtension(NewLA, depth - 1, output, restrict_output))
    return ret

def ExtendL(LA, depth, output=True, restrict_output=None):
    ret = []
    LAFound = FirstExtendLieAlgebra(LA)
    if output:
        PrintFoundLieAlgebras(LAFound, restrict_output)
    ret.extend(LAFound)
    for NewLA in LAFound:
        ret.extend(RecursiveExtension(NewLA, depth - 1, output, restrict_output))
    return ret
    
