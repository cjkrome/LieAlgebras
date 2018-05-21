import math
from sympy import *
from copy import deepcopy
import numpy as np
import scipy
import sys

import networkx as nx
import matplotlib.pyplot as plt
import string

#------------------------------------------------------------------------------
# Graph comparison
#------------------------------------------------------------------------------

def test_degree(G1, G2):
    G1_degrees = sorted([G1.degree(n) for n in G1.nodes()])
    G2_degrees = sorted([G2.degree(n) for n in G2.nodes()])
    return G1_degrees == G2_degrees

def draw(G):
    edge_colors = ['blue' if w == 1 else 'green' for w in (nx.get_edge_attributes(G, 'weight').values())]
    #nx.draw_shell(G, with_labels=True, edge_color=edge_colors, node_size=2000)
    nx.draw_circular(G, with_labels=True, edge_color=edge_colors, node_size=2000)
    plt.show(block=True)
    
# Get all nodes of degree degree
def degree_nodes(G, degree):
    return list(filter(lambda n: G.degree(n) == degree, G.nodes()))

# Pre-condition: |G1| == |G2|
# Run test_degree() before calling this function
# G1 will take on the values of G2, so G2 will not be modified
def graphs_equal_impl(G1, G2, G1_unmapped, G2_unmapped):
    if len(G1_unmapped) == 0:
        return True
    n1 = G1_unmapped.pop()
    deg = G1.degree(n1)
    G2_nodes = degree_nodes(G2, deg)
    # For each unvisited node that has the same degree as n1,
    # check to see if it is compatible with n1 by comparing 
    # edges with visited (mapped) nodes.
    for n2 in set(G2_nodes).intersection(G2_unmapped):
        mapped = n2
        G1.nodes[n1]['mapped'] = mapped
        consistent = True
        # n1 is consistent with n2 if all mapped neighbors to n1
        # have edges in G2 and those edges have identical labels.
        for nbr in G1.adj[n1]:
            nmapped = G1.nodes[nbr]['mapped']
            if nmapped != '' and (not G2.has_edge(mapped,nmapped)
                                  or G2[mapped][nmapped]['weight'] != G1[n1][nbr]['weight']):
                consistent = False
        if consistent:
            G1_unmapped_copy = G1_unmapped.copy()
            G2_unmapped_copy = G2_unmapped.copy()
            G2_unmapped_copy.remove(n2)
            if graphs_equal_impl(G1, G2, G1_unmapped_copy, G2_unmapped_copy):
                return True
        G1.nodes[n1]['mapped'] = ''
    return False

def graphs_equal(G1, G2):
    if not test_degree(G1, G2):
        return False
    return graphs_equal_impl(G1, G2, set(G1.nodes()), set(G2.nodes))

#------------------------------------------------------------------------------
# Class Filter
#------------------------------------------------------------------------------

# Utility class
class Filter:
    def __init__(self, d=None, dimension=None, extension=None, type=None, U_matrix=False):
        self.d = d
        self.dimension = dimension
        self.extension = extension
        self.type = type
        self.U_matrix = U_matrix

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
# Graph
#------------------------------------------------------------------------------
graph = nx.DiGraph()
def draw_graph():
    #nx.draw_shell(graph, with_labels=True)
    nx.draw_networkx(graph, with_labels=True)
    plt.show(block=True)
    

#------------------------------------------------------------------------------
# JacobiIdentities
#------------------------------------------------------------------------------
#class JacobiIdentities:

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
    def __init__(self, dimension):#, name=None):
        self.brackets = {}
        self.dimension = dimension
        self.d = 0
        self.extension = 0
        self.jacobi_tests = {}
        self.type = None
        self.parent = None

    def __repr__(self):
        return 'm{}({},{})'.format(
            str(self.extension)+self.type, self.d, self.dimension)

    def simple_repr(self):
        return latex("m{}{}{}{}".format(
            str(self.extension), self.type, self.d, self.dimension))

    def latex_repr(self):
        return latex("$m_{{{}}}({}, {})$".format(
            str(self.extension) + self.type, self.d, self.dimension))

    # f = filter
    def matches(self, f):
        if f == None:
            return True
        return ((f.d == None or self.d in f.d) and
                (f.dimension == None or self.dimension in f.dimension) and
                (f.extension == None or self.extension in f.extension) and
                (f.type == None or self.type == f.type))

    # Accepts i, j, and k as indices and then adds a new bracket to the lie
    # algebra. (Alpha is optional and defaults to 1)
    def add_bracket(self, i, j, k, alpha=1):
        res = BracketResult(k, alpha)
        #if self.type == 'B':
        #    print('bracket for {},{} = {},{}'.format(i, j, k, alpha));
        self.brackets[i, j] = res

    # Accepts i, j, and k as eigenvalues, converts them to indices, and then
    # uses the indices to add a bracket.
    def add_eigenvalue_bracket(self, i, j, k, d, n, extType, ext):
        i = self.convert_eigenvalue_to_index(i, d, n, extType)
        j = self.convert_eigenvalue_to_index(j, d, n, extType)
        k = self.convert_eigenvalue_to_index(k, d, n, extType)

        # Hard-code structure constant of 1 for first B extensions
        if extType == 'B' and i == 2:
            self.add_bracket(i, j, k)
        elif extType == 'A' and ext == 1:
            self.add_bracket(i, j, k, (-1)**i)
        else:
            # This format ensures correct Latex printing:
            alphatext = "alpha_{},{}^{}".format(i, j, k)
            self.add_bracket(i, j, k, Symbol(alphatext))

    # Performs the bracket operation [Xi, Xj]
    def bracket(self, i, j):
        newAlpha = 1

        if type(i) == BracketResult:
            newAlpha *= i.alpha
            i = i.index
        if type(j) == BracketResult:
            newAlpha *= j.alpha
            j = j.index

        if (i, j) in self.brackets:
            return self.brackets[(i, j)] * newAlpha
        # If [Xi, Xj] is not found, check for [Xj, Xi] (and multiply the
        # resulting alpha by -1 if found).
        elif (j, i) in self.brackets:
            return self.brackets[(j, i)] * (newAlpha * -1)
        else:
            return 0

    def convert_index_to_eigenvalue(self, index, la_type, d=0):
        if la_type == 'A':
            if d == 0:
                d = self.d
            return index + d - 2
        return index + 2 * d - 3

    def convert_eigenvalue_to_index(self, eigenval, d, n, extType):
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
        
    def print_brackets(self, lines):
        lines.append('\\begin{align*}')
        for key1, key2 in sorted(self.brackets):
            res = self.brackets[(key1, key2)]

            # Doubling the bracket acts as an escape character
            # (i.e. '{{' becomes '{' in the final output),
            # and we need the keys to be inside of brackets
            # before being converted to Latex
            # so that multiple digits stick together inside of a subscript.
            bracketFormat = latex("[X_{{{}}},X_{{{}}}]".format(key1, key2))
            resultFormat = latex(res.alpha * Symbol("X" + str(res.index)))

            #lines.append("$${} = {}$$".format(bracketFormat, resultFormat))
            lines.append("{} &= {} \\\\".format(bracketFormat, resultFormat))
        lines.append('\\end{align*}')


    # Find all of the non-trivial triples that need to be tested.
    def create_jacobi_tests(self):
        Y = self.create_Y()
        U = np.dot(Y, Y.transpose())
        self.Y = Y
        self.U = U
    
        # Build U graph
        self.G = nx.Graph()
    
        labels = range(U.shape[0])#['a', 'b', 'c']
        self.G.add_nodes_from(labels)
    
        for x,y in np.ndindex(U.shape):
            w = U[x,y]
            if w != 0:
                self.G.add_edge(x, y, weight=w)
            
        # Do the rest
        negOnes = np.where(U == -1)
        allIndices = []
        for i in range(len(negOnes[0])):
            idx0 = negOnes[0][i]
            idx1 = negOnes[1][i]
            sum = np.add(Y[idx0], Y[idx1])
            indices = np.where(sum == 1)
            isIn = False
            for i in allIndices:
                isIn = isIn or np.array_equal(indices, i)
            if not isIn:
                allIndices.append(indices)
        #self.JacobiToTest = []
        for indices in allIndices:
            i = indices[0][0]+1
            j = indices[0][1]+1
            k = indices[0][2]+1
            resultset = test_jacobi(self, i, j, k)
            if resultset != None:
                self.jacobi_tests[(i, j, k)] = resultset

        # Groebner basis and solution
        eqns = self.jacobi_tests.values()
        if len(eqns) == 0:
            self.gsolutions = []
            return
        for eqn in eqns:
            if eqn == False:
                self.gsolutions = None
                return

        # Get the set of all symbols and come up with new
        # symbols. Since the existing symbols have superscripts
        # they'll confuse the Groebner routine, so we have
        # to use replacement variables.
        syms = []
        for eqn in eqns:
            syms.extend(eqn.free_symbols)
        syms = set(syms)
        n = len(syms)
        new_syms = ['x_{{{}}}'.format(i) for i in range(1,n+1)]
        self.old2new = list(zip(syms, new_syms))
        self.new2old = dict(zip(new_syms, syms))

        # do substitution
        eqns = [eqn.subs(self.old2new) for eqn in eqns]

        self.geqns = groebner(eqns)
        try:
            self.gsolutions = solve_poly_system(self.geqns)
        except:
            self.gsolutions = 'Unable to determine solution'

    def create_Y(self):
        numTriples = len(self.brackets)
        I = np.identity(self.dimension)
        Y = np.zeros((numTriples, self.dimension))
        i = 0
        for key1, key2 in sorted(self.brackets):
            # key1 and key2 are indices
            key3 = self.brackets[(key1, key2)].index
            Y[i] = I[key1-1] + I[key2-1] - I[key3-1]
            i = i + 1
        rank = np.linalg.matrix_rank(Y)
        corank = numTriples - rank
        return Y

    def print_jacobi_tests_impl(self, lines, tests):
        lines.append('\\begin{align*}')
        for triple,eqn in tests.items():
            triple_str = '(e_{{{}}}, e_{{{}}}, e_{{{}}})'.format(
                triple[0], triple[1], triple[2]);
            if eqn == False:
                lines.append("{}: & \\quad INVALID\\\\".format(triple_str))
            else:
                lines.append("{}: & \\quad \\displaystyle {} &&= 0\\\\".format(
                    triple_str, latex(eqn.lhs)))
        lines.append('\\end{align*}')

    def print_jacobi_tests(self, lines):
        self.print_jacobi_tests_impl(lines, self.jacobi_tests)

        if self.gsolutions == None:
            lines.append('There are no solutions.\\\\')
        elif type(self.gsolutions) == str:
            lines.append('{}.\\\\'.format(self.gsolutions))
        else:
            for solution in self.gsolutions:
                lines.append('Solution:\\\\')
                lines.append('\\begin{align*}')
                for i,v in enumerate(self.geqns.gens):
                    lines.append('{} &= {} \\\\'.format(
                        latex(self.new2old[str(v)]), solution[i]))
                
                lines.append('\\end{align*}')

    def print_groebner(self, lines):
        eqns = self.jacobi_tests.values()
        if len(eqns) == 0 or self.gsolutions == None:
            return

        # Show the symbol substitution
        lines.append('\\textit{How the solution was or was not found:}\\\\')
        lines.append('Change variables\\\\')
        for o,n in self.old2new:
            lines.append('$${} \\rightarrow {}$$'.format(latex(o), latex(n)))

        # Print the Jacobi tests
        lines.append('Jacobi Tests\\\\')
        tests = {key:eqn.subs(self.old2new)
                 for key,eqn in self.jacobi_tests.items()}
        self.print_jacobi_tests_impl(lines, tests)
        #for triple,eqn in self.jacobi_tests.items():
        #    lines.append("Jacobi identity for {}: $${}$$".format(
        #        triple, latex(eqn.subs(self.old2new))))

        # Print the equations in the Groebner basis
        lines.append('Groebner basis\\\\')
        for geqn in self.geqns:
            lines.append("$${}=0$$".format(latex(geqn)))

        if type(self.gsolutions) != str:
            for solution in self.gsolutions:
                lines.append('Solution:')
                for i,v in enumerate(self.geqns.gens):
                    lines.append('$${} = {}$$'.format(v, solution[i]))

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------

def create_L(dimension):
    LA = LieAlgebra(dimension)
    LA.d = 2
    LA.type = 'A'
    for j in range(2, dimension):
        LA.add_bracket(1, j, j + 1)
    return LA

# Test an individual triple to see if it is trivial.
def test_jacobi(LA, i, j, k):
    r1 = LA.bracket(i, LA.bracket(j, k))
    r2 = LA.bracket(j, LA.bracket(k, i))
    r3 = LA.bracket(k, LA.bracket(i, j))
    if r1 != 0 or r2 != 0 or r3 != 0:
        ret = Eq(GetEqTerm(r1) + GetEqTerm(r2) + GetEqTerm(r3))
        if ret == True:
            #print('equation for {} is true: {} + {} + {}'.format(
            #    LA, GetEqTerm(r1), GetEqTerm(r2), GetEqTerm(r3)));
            ret = None
        return ret
    else:
        return None

# Convert a jacobi test into an equation term.
def GetEqTerm(eqn):
    if type(eqn) == BracketResult:
        return eqn.alpha
    else:
        return 0


# Check if a 'd' value is valid for a given 'n'.
def IsValidD(n, d):
    if n - d % 2 == 0:
        valid = (n - 2) > d
    else:
        valid = (n - 1) > d
    return valid


# Accepts a lie algebra and a d value, increments the dimension, and then
# adds the new brackets.
def extend_LA(LA, d, extType):
    n = LA.dimension
    n2 = n + 1  # The dimension of the new Lie Algebras

    LastValue = n + d - 1  # The last eigenvalue
    if extType == 'B':
        LastValue = n2 + 2*d - 3

    try:
        new_LA = deepcopy(LA)
    except:
        print('Failed to extend {}: failure in deep copy'.format(LA))
        raise

    new_LA.extension += 1
    new_LA.dimension += 1
    new_LA.d = d
    new_LA.type = extType
    new_LA.parent = LA

    graph.add_node(new_LA)
    graph.add_edge(LA, new_LA);

    if extType == 'A':
        new_LA.add_bracket(1, n, n2)
    else:
        new_LA.add_bracket(2, n, n2)

    extension = new_LA.extension
    startValue = d

    #print('from {} to {}'.format(LA, new_LA))

    # Odd case
    if (n - d) % 2 != 0:
        CenterValue = int(LastValue / 2)
        for i in range(startValue, CenterValue + 1):
            j = LastValue - i
            if i != j:
                new_LA.add_eigenvalue_bracket(
                    i, j, LastValue, d, n, extType, extension)
    # Even case
    else:
        CenterValue = int((LastValue - 1) / 2.0)
        for i in range(startValue, CenterValue + 1):
            j = LastValue - i
            new_LA.add_eigenvalue_bracket(
                i, j, LastValue, d, n, extType, extension)

    new_LA.create_jacobi_tests()
    return new_LA


# Accepts a lie algebra and finds the TypeB that can be made by extending it.
def FindNextDimensionTypeB(LA):
    n = LA.dimension
    d = LA.d
    new_LA = deepcopy(LA)
    new_LA.dimension += 1
    new_LA.type = 'B'

    LastValue = 2 * d + n - 2
    CenterValue = math.floor(LastValue / 2)
    RangeEnd = CenterValue + 1 if ((n - d) % 2 != 0) else CenterValue

    for i in range(d, RangeEnd):
        j = LastValue - i
        new_LA.add_eigenvalue_bracket(i, j, LastValue, d, n, extType)

    return new_LA

def null(A, eps=1e-15):
    u, s, vh = np.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return np.transpose(null_space)

def Y_test(LA):
    print('Y_test')
    """
    Y = LA.create_Y()
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

def print_U_matrix(LA, lines):
    lines.append('$Y=');
    lines.append('\\begin{bmatrix}')
    lines.append(" \\\\\n".join([" & ".join(map(str,line)) for line in LA.Y]))
    lines.append('\\end{bmatrix}')
    lines.append('$')
    lines.append()
    
    lines.append('$U=');
    lines.append('\\begin{bmatrix}')
    lines.append(" \\\\\n".join([" & ".join(map(str,line)) for line in LA.U]))
    lines.append('\\end{bmatrix}')
    lines.append('$')
    lines.append()

    nullU = null(LA.U)
    lines.append('$null(U)=');
    lines.append('\\begin{bmatrix}')
    lines.append(" \\\\\n".join([" & ".join(map(str,line)) for line in nullU]))
    lines.append('\\end{bmatrix}')
    lines.append('$')
    lines.append()


def print_LA(LA, verbose=False):    
    lines = []
    lines.append("\section*{{{}}}".format(LA.latex_repr()))
    lines.append('{} (this line included for string searching purposes)'.format(
        LA.simple_repr()))
    LA.print_brackets(lines)
    lines.append("\nNon-trivial Jacobi Tests:\n")
    #Y_test(LA)
    LA.print_jacobi_tests(lines)
    LA.print_groebner(lines)

    if verbose:
        print_U_matrix(LA, lines)
        
    # Draw the graph
    #draw(LA.G)

    return '\n'.join(lines)
            

# Extends a lie algebra, checking for all possible d values.
# This function is iterative (no recursion).
def FirstExtendLieAlgebra(LA):
    NewLieList = []
    if LA.type == 'B':
        return NewLieList

    if LA.dimension % 2 == 0:
        d = 2
    else:
        d = 3
    while IsValidD(LA.dimension, d):
        newLA = extend_LA(LA, d, extType='A')
        NewLieList.append(newLA)
        # Generate extended type B LA from newLA
        if newLA.dimension % 2 == 1:
            newLA_B = extend_LA(newLA, newLA.d, extType='B')
            NewLieList.append(newLA_B)
        d += 2
    return NewLieList

# Extends a lie algebra using the d value of LA.
def ExtendLieAlgebra(LA):
    NewLieList = []
    if LA.type == 'B':
        return NewLieList

    d = LA.d # start d value
    try:
        newLA = extend_LA(LA, d, extType='A')
        NewLieList.append(newLA)

        # Generate extended type B LA from newLA
        if newLA.dimension % 2 == 1:
            try:
                newLA_B = extend_LA(newLA, newLA.d, extType='B')
                NewLieList.append(newLA_B)
            except:
                print('Failed B extension of {}'.format(newLA))
    except:
        print('Failed A extension of {}'.format(LA))


    return NewLieList

def RecursiveExtension(LA, depth):
    ret = []
    if depth > 0:
        LAFound = ExtendLieAlgebra(LA)
        ret.extend(LAFound)
        for NewLA in LAFound:
            ret.extend(RecursiveExtension(NewLA, depth - 1))
    return ret

def ExtendL(LA, depth):
    ret = []
    LAFound = FirstExtendLieAlgebra(LA)
    ret.extend(LAFound)
    for NewLA in LAFound:
        ret.extend(RecursiveExtension(NewLA, depth - 1))

    return ret
    
def print_latex(LAs):
    lines = []
    lines.append(
        '\\documentclass{article}\n\\setlength{\\parindent}{0cm} ' +
        '% Default is 15pt.')
    lines.append('\\usepackage{amsmath}')
    lines.append('\\setcounter{MaxMatrixCols}{30}')
    lines.append('\\usepackage{multicol}')
    lines.append('\\begin{document}')
    #lines.append('\\begin{multicols}{2}')


    lines.append('\\begin{multicols}{2}')
    
    n = len(LAs)
    num_per_col = 42
    for i in range((n//num_per_col)+1):
        #lines.append('\\newpage')
        lines.append('\\begin{tabular}{|l|l|c|}')
        lines.append('\hline')
        lines.append('{} & {} & {}\\\\'.format('search', '', 'has'))
        lines.append('{} & {} & {}\\\\'.format('string', 'algebra', 'soln'))
        lines.append('\hline')
        stop = min(n, (i+1)*num_per_col)
        for LA in LAs[i*num_per_col:stop]:
            s = LA.gsolutions
            sol = '' if s == None else '?' if type(s) == str else '$\\surd$'
            lines.append('\\tiny{{{}}} & {} & {} \\\\'.format(
                LA.simple_repr(), LA.latex_repr(), sol))
            lines.append('\hline')
        lines.append('\\end{tabular}')
        lines.append('\\vfill')
        #lines.append('\\vfill\\null')
        #lines.append('\\vspace*{\\fill}')
        #lines.append('\\columnbreak')

    lines.append('\\end{multicols}')

    for LA in LAs:
        lines.append(print_LA(LA))

    #lines.append('\\end{multicols}')
    lines.append('\n\\end{document}\n')
    return '\n'.join(lines)
