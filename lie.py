import math
from sympy import *
from copy import deepcopy
import numpy as np
import scipy
import sys
import sympy.polys.polyerrors
from functools import reduce

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
#   1. k - the e index of the result.
#   2. alpha - the alpha value for the result (may be a number or Sympy symbol).
# Example: Given the bracket [e_1, e_2] = e_3, the bracket result object would be:
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
        return latex("$\\frakm_{{{}}}({}, {})$".format(
            str(self.extension) + self.type, self.d, self.dimension))

    # f = filter
    def matches(self, f):
        if f == None:
            return True
        return ((f.d == None or self.d in f.d) and
                (f.dimension == None or self.dimension in f.dimension) and
                (f.extension == None or self.extension in f.extension) and
                (f.type == None or self.type == f.type))

    #------------------------------------------------------------
    # Functions to create brackets
    #------------------------------------------------------------

    # Accepts i, j, and k as indices and then adds a new bracket to the lie
    # algebra. (Alpha is optional and defaults to 1)
    def add_bracket(self, i, j, k, alpha):
        self.brackets[i, j] = BracketResult(k, alpha)

    # Accepts i, j, and k as eigenvalues, converts them to indices, and then
    # uses the indices to add a bracket.
    def add_bracket_smart(self, i, j, k, d, n, extType, ext):
        # Hard-code structure constant of 1 for first B extensions
        if extType == 'B' and i == 2:
            alpha = 1
        elif extType == 'A' and ext == 1:
            # 1st extension variable reduction hack (proposition 5.1)
            alpha = (-1)**i
        elif extType == 'A' and ext == 2:
            # 2nd extension variable reduction hack (proposition 5.2)
            alpha = ((-1)**i)*(((n-d+3)//2)-i)
#        elif extType == 'A' and ext == 3 and (n-d)%2==0:
#            # 3rd extension variable reduction hack (proposition 5.x)
#            l = int((n-d+2)/2)
#            x = ((-1)**i)*((l-i+1)*(l-i))/2
#            y = int(((-1)**i)*(-1)**l)
#            alpha = x+y*Symbol('s')
        else:
            # This format ensures correct Latex printing:
            alphatext = "alpha_{},{}^{}".format(i, j, k)
            alpha = Symbol(alphatext)

        self.add_bracket(i, j, k, alpha)

    # Performs the bracket operation [e_i, e_j]
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
        # If [e_i, e_j] is not found, check for [e_j, e_i] (and multiply the
        # resulting alpha by -1 if found).
        elif (j, i) in self.brackets:
            return self.brackets[(j, i)] * (newAlpha * -1)
        else:
            return 0
        
    def print_soln_brackets(self, lines):
        lines.append("\\\\")
        for i,a2s in enumerate(self.alpha2soln):
            lines.append("Solution {}".format(i+1))
            lines.append('\\begin{align*}')
            i = 0
            for key1, key2 in sorted(self.brackets):
                res = self.brackets[(key1, key2)]

                bracket = latex("[e_{{{}}},e_{{{}}}]".format(key1, key2))
                if res.alpha in a2s:
                    result = latex(a2s[res.alpha] * Symbol("e" + str(res.index)))
                else:
                    result = latex(res.alpha * Symbol("e" + str(res.index)))

                if i % 2 == 0:
                    lines.append("{} &= {} &".format(bracket, result))
                else:
                    lines.append("{} &= {} \\\\".format(bracket, result))
                i += 1
            lines.append('\\end{align*}')

    def print_orig_brackets(self, lines):
        lines.append('\\begin{align*}')
        i = 0
        for key1, key2 in sorted(self.brackets):
            res = self.brackets[(key1, key2)]

            # Doubling the bracket acts as an escape character
            # (i.e. '{{' becomes '{' in the final output),
            # and we need the keys to be inside of brackets
            # before being converted to Latex
            # so that multiple digits stick together inside of a subscript.
            bracket = latex("[e_{{{}}},e_{{{}}}]".format(key1, key2))
            result = latex(res.alpha * Symbol("e" + str(res.index)))
            if i % 2 == 0:
                lines.append("{} &= {} &".format(bracket, result))
            else:
                lines.append("{} &= {} \\\\".format(bracket, result))
            i += 1
        lines.append('\\end{align*}')


    def get_jacobi_indices_old(self):
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

        # First array of negOnes are the rows? and the second array are the columns?
        negOnes = np.where(U == -1)
        allIndices = []
        # len(negOnes[0]) is the number of negative ones in U
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
        isets = []
        for indices in allIndices:
            i = indices[0][0]+1
            j = indices[0][1]+1
            k = indices[0][2]+1
            isets.append((i,j,k))
#            resultset = test_jacobi(self, i, j, k)
#            if resultset != None:
#                self.jacobi_tests[(i, j, k)] = resultset
        return isets

    # Find all of the non-trivial triples that need to be tested.
    def create_jacobi_tests(self):
        self.alpha2soln = []
        if self.type == 'A':
            isets = get_jacobi_indices_typeA(self.dimension-1, self.d)
        else:
            isets = get_jacobi_indices_typeB(self.dimension-1, self.d)
        for iset in isets:
            resultset = test_jacobi(self, iset[0], iset[1], iset[2])
            if resultset != None:
                self.jacobi_tests[iset] = resultset
#            else:
#                print('trivial: {} {}'.format(self, iset))

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
        #new_syms = ['x_{{{}}}'.format(i) for i in range(1,n+1)]
        new_sym_strings = ['x_{{{}}}'.format(i) for i in range(1,n+1)]
        new_syms = [Symbol('x_{}'.format(i)) for i in range(1,n+1)]
        self.old2new = list(zip(syms, new_syms))
        self.new2old = dict(zip(new_syms, syms))

        # do substitution
        eqns = [eqn.subs(self.old2new) for eqn in eqns]

        self.geqns = groebner(eqns)
        try:
            self.gsolutions = solve_poly_system(self.geqns)
            for solution in self.gsolutions:
                a2s = {}
                for i,v in enumerate(self.geqns.gens):
                    a2s[self.new2old[v]] = solution[i]
                self.alpha2soln.append(a2s)

        #except Exception as e:
        except NotImplementedError as e:
            self.gsolutions = 'Infinite number of solutions: {}'.format(e)
            #print('Infinite number of solutions: {}: {} {}'.format(self, type(e), e))

            #print(self.geqns)
            #for e in self.geqns:
            #    print(type(e))
                #print(e.args)
                #try:
                #    if not e.is_number():
                #        print(degree(e.as_poly()))
                #    else:
                #        print('is number: {}'.format(''))
                #except Exception as e:
                #    print('exception: {}'.format(e))
        except sympy.polys.polyerrors.ComputationFailed as e:
            self.gsolutions = 'No solutions'
            #print('No solutions: {} {}'.format(type(e), e))
            #print('No solutions: {}'.format(e))
            

    def jacobi_tests_consistent(self):
        return self.gsolutions != None

    def has_solution(self):
        return self.gsolutions != None and type(self.gsolutions) != str

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

    #------------------------------------------------------------
    # Output functions
    #------------------------------------------------------------

    def print_jacobi_tests_impl(self, lines, tests):
        lines.append('\\begin{align*}')
        for triple,eqn in tests.items():
            triple_str = '(e_{{{}}}, e_{{{}}}, e_{{{}}})'.format(
                triple[0], triple[1], triple[2]);
            if eqn == False:
                lines.append("{}: & \\quad \\text{{no solutions}}\\\\".format(triple_str))
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
            for i,solution in enumerate(self.gsolutions):
                lines.append('Solution {}:\\\\'.format(i+1))
                lines.append('\\begin{align*}')
                for i,v in enumerate(self.geqns.gens):
#                    lines.append('{} &= {} \\\\'.format(
#                        latex(self.new2old[str(v)]), solution[i]))
                    lines.append('{} &= {} \\\\'.format(
                        latex(self.new2old[v]), solution[i]))
                
                lines.append('\\end{align*}')

    def print_groebner(self, lines):
        eqns = self.jacobi_tests.values()
        if len(eqns) == 0 or self.gsolutions == None:
            return

        # Show the symbol substitution
        lines.append('\\textit{How the solution(s) were or were not found:}\\\\')
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
        polys = self.geqns.polys
        num_linear = reduce((lambda x, y: x + y), [1 if poly.is_linear else 0 for poly in polys])
        num_nonlinear = len(polys) - num_linear
        lines.append('Groebner basis ({} variables, {} linear, {} nonlinear)\\\\'.
                     format(len(self.geqns.gens), num_linear, num_nonlinear))
        for geqn in self.geqns:
            lines.append("$${}=0$$".format(latex(geqn)))

        if type(self.gsolutions) != str:
            for i,solution in enumerate(self.gsolutions):
                lines.append('Solution {}:'.format(i+1))
                for i,v in enumerate(self.geqns.gens):
                    lines.append('$${} = {}$$'.format(v, solution[i]))

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------

# Computes lambda_i
def lam(i, n, d, ext_type):
    if i == 1:
        return 1
    if ext_type == 'A':
        return i + (d-2)
    else:
        if i == n:
            return n+2*d-3
        return i+(d-2)

def get_jacobi_indices_impl(n, d, ext_type):
    isets = []
    imax = max(1, int((lam(n+1, n+1, d, ext_type)-3*d+3)/3))
    for i in range(1, imax+1):
        jmax = math.floor((lam(n+1, n+1, d, ext_type)-lam(i, n+1, d, ext_type)-2*d+3)/2)
        for j in range(i+1, jmax+1):
            k = lam(n+1,n+1,d,ext_type) - lam(i,n+1,d,ext_type) - j - 2*d + 4
            isets.append((i,j,k))
            j = j + 1
        i = i + 1
    return isets

def get_jacobi_indices_typeA(n, d):
    return get_jacobi_indices_impl(n,d,'A')

# TODO: change this code to match the paper
def get_jacobi_indices_typeB(n, d):
    ext_type='B'
    isets = []
    imax = max(1, int((lam(n+1, n+1, d, 'B')-3*d+3)/3))
    for i in range(1, imax+1):
        jmax = math.floor((lam(n+1, n+1, d, ext_type)-lam(i, n+1, d, ext_type)-2*d+3)/2)
        for j in range(i+1, jmax+1):
            k = lam(n+1,n+1,d,ext_type) - lam(i,n+1,d,ext_type) - j - 2*d + 4
            isets.append((i,j,k))
            j = j + 1
        i = i + 1
    return isets

#print(get_jacobi_indices(5, 2))
#print(get_jacobi_indices(6, 2))
#print(get_jacobi_indices(7, 2))
#print(get_jacobi_indices(8, 2))
#print(get_jacobi_indices(10, 4))
#print(get_jacobi_indices_typeB(7, 2))
#print(get_jacobi_indices(10, 2))

# Test an individual triple to see if it is trivial.
def test_jacobi(LA, i, j, k):
    r1 = LA.bracket(i, LA.bracket(j, k))
    r2 = LA.bracket(j, LA.bracket(k, i))
    r3 = LA.bracket(k, LA.bracket(i, j))
    if r1 != 0 or r2 != 0 or r3 != 0:
        ret = Eq(GetEqTerm(r1) + GetEqTerm(r2) + GetEqTerm(r3), 0)
        # Eq returns True if the equality is trivially equal.
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

#------------------------------------------------------------
# Output functions
#------------------------------------------------------------

def mystr(x):
    return '{:.0f}'.format(x)

def print_U_matrix(LA, lines):
    lines.append('$Y=');
    lines.append('\\begin{bmatrix}')
    #lines.append(" \\\\\n".join([" & ".join(map(str,line)) for line in LA.Y]))
    lines.append(" \\\\\n".join([" & ".join(map(mystr,line)) for line in LA.Y]))
    lines.append('\\end{bmatrix}')
    lines.append('$')
    lines.append('')
    
    lines.append('$U=');
    lines.append('\\begin{bmatrix}')
#    lines.append(" \\\\\n".join([" & ".join(map(str,line)) for line in LA.U]))
    lines.append(" \\\\\n".join([" & ".join(map(mystr,line)) for line in LA.U]))
    lines.append('\\end{bmatrix}')
    lines.append('$')
    lines.append('')

#    nullU = null(LA.U)
#    lines.append('$null(U)=');
#    lines.append('\\begin{bmatrix}')
#    lines.append(" \\\\\n".join([" & ".join(map(str,line)) for line in nullU]))
#    lines.append('\\end{bmatrix}')
#    lines.append('$')
#    lines.append('')


def print_LA(LA, verbose=False):    
    lines = []
    lines.append("\section*{{{}}}".format(LA.latex_repr()))
    lines.append('{} (this line included for string searching purposes)'.format(
        LA.simple_repr()))
    LA.print_soln_brackets(lines)
    lines.append("\nOriginal brackets:\n")
    LA.print_orig_brackets(lines)
    lines.append("\nNon-trivial Jacobi Tests:\n")
    #Y_test(LA)
    LA.print_jacobi_tests(lines)
    LA.print_groebner(lines)

    if verbose:
        print_U_matrix(LA, lines)
        
    # Draw the graph
    #draw(LA.G)

    return '\n'.join(lines)
            
def print_latex(LAs):
    lines = []
    lines.append(
        '\\documentclass{article}\n\\setlength{\\parindent}{0cm} ' +
        '% Default is 15pt.')
    lines.append('\\usepackage{amsmath}')
    lines.append('\\usepackage{amsfonts}')

    lines.append('\\newcommand{\\frakm}{\\ensuremath{\\mathfrak{m}} }')

    lines.append('\\setcounter{MaxMatrixCols}{30}')
    lines.append('\\usepackage{multicol}')
    lines.append('\\begin{document}')
    #lines.append('\\begin{multicols}{2}')

    lines.append('\\begin{multicols}{2}')
    
    n = len(LAs)
    num_per_col = 42
    for i in range((n//num_per_col)+1):
        #lines.append('\\newpage')
        lines.append('\\begin{tabular}{|l|l|c|c|}')
        lines.append('\hline')
        #lines.append('{} & {} & {} & {} & {}\\\\'.format(
        #    'search', 'algebra', 'Jac', 'lin', 'sol'))
        lines.append('{} & {} & {} & {}\\\\'.format(
            'search', 'algebra', 'Jac', 'sol'))
        lines.append('\hline')
        stop = min(n, (i+1)*num_per_col)
        for LA in LAs[i*num_per_col:stop]:
            s = LA.gsolutions
            if not LA.jacobi_tests_consistent():
                sol = '0'
                jac = ''
            elif s == None:
                sol = ''
                jac = ''
            elif s == 'No solutions':
                sol = '0'
                jac = ''
            elif type(s) == str and s.startswith('Infinite number of solutions'):
                sol = '$\\infty$'
                jac = '$\\surd$'
            else:
                sol = str(max([1, len(LA.gsolutions)]))
                jac = '$\\surd$'
#            jac = '$\\surd$' if LA.jacobi_tests_consistent() else '-'
#            if not LA.jacobi_tests_consistent():
#                lines.append('\\tiny{{{}}} & {}  & {}  & {} \\\\'.format(
#                    LA.simple_repr(), LA.latex_repr(), jac, '-'))
#            else:
#                lines.append('\\tiny{{{}}} & {} & {}  & {} \\\\'.format(
#                    LA.simple_repr(), LA.latex_repr(), jac, sol))
            lines.append('\\tiny{{{}}} & {} & {}  & {} \\\\'.format(
                LA.simple_repr(), LA.latex_repr(), jac, sol))
            lines.append('\hline')
        lines.append('\\end{tabular}')
        lines.append('\\vfill')
        #lines.append('\\vfill\\null')
        #lines.append('\\vspace*{\\fill}')
        #lines.append('\\columnbreak')

    lines.append('\\end{multicols}')

    lines.append('Jac = Jacobi tests are consistent\\\\')
    lines.append('lin = Equations in Groebner basis are linear\\\\')
    lines.append('sol = Found solution\\\\')

    for LA in LAs:
        verbose = False
        lines.append(print_LA(LA, verbose))

    #lines.append('\\end{multicols}')
    lines.append('\n\\end{document}\n')
    return '\n'.join(lines)

def print_csv(LAs):
    lines = []
    lines.append('m, d, n, type, solns')
    
    for LA in LAs:
        s = LA.gsolutions
        if s == None:
            sol = ''
        elif s == 'No solutions':
            sol = '0'
        elif type(s) == str and s.startswith('Infinite number of solutions'):
            sol = 'inf'
        elif not LA.jacobi_tests_consistent():
            sol = '-'
        else:
            sol = str(max([1, len(LA.gsolutions)]))

        line = '{}, {}, {}, {}, {}'.format(LA.extension, LA.d, LA.dimension, LA.type, sol);
        lines.append(line);
    return '\n'.join(lines)

#------------------------------------------------------------
# Functions to create extensions
#------------------------------------------------------------

# Accepts a lie algebra and a d value, increments the dimension, and then
# adds the new brackets.
def extendLA(LA, d, extType):
    n = LA.dimension

    try:
        ext = deepcopy(LA)
    except:
        print('Failed to extend {}: failure in deep copy: {}'.format(LA, sys.exc_info()[0]))
        raise

    ext.extension += 1
    ext.dimension += 1
    ext.d = d
    ext.type = extType
    ext.parent = LA

    graph.add_node(ext)
    graph.add_edge(LA, ext);

    if extType == 'A':
        ext.add_bracket(1, n, n+1, 1)
    else:
        ext.add_bracket(2, n, n+1, 1)

    extNum = ext.extension

    # This code finds all brackets such that the eigenvalues sum
    # to the newly-added eigenvalue. For n=11 and d=3:
    #  i | 1  2  3  4  5  6  7  8  9  10 | 11
    #  e | 1  3  4  5  6  7  8  9  10 11 | 12
    #               |_____|
    #            |___________|
    #         |_________________|
    #         i                 j

    i = 2
    if extType == 'A':
        j = n-d+3-i
        imax = math.floor((n-d+2)/2)
    else:
        j = n+2-i
        imax = (n+1)/2

    while (i <= imax):
        ext.add_bracket_smart(
            i, j, n+1, d, n, extType, extNum)
        i = i+1
        j = j-1

    ext.create_jacobi_tests()
    return ext

def extendRecursively(LA, extensions, maxDimension):
    # Reached our maximum depth
    if LA.dimension == maxDimension:
        return []

    # Take type A extension
    try:
        extension = extendLA(LA, LA.d, extType='A')
        extensions.append(extension)
        # Make recursive call
        extendRecursively(extension, extensions, maxDimension)
    except Exception as e:
        print('Failed A extension of {}: {}'.format(LA, e))

    # Take type B extension if it exists
    if LA.dimension % 2 == 1:
        try:
            extension = extendLA(LA, LA.d, extType='B')
            extensions.append(extension)
            # No recursive call for type B -- they can't be extended
        except:
            print('Failed B extension of {}'.format(LA))

# This extends a standard (L) algebra. Find all the one-fold
# extensions for different d values then recursively find remaining
# extensions up to depth.
def extendStandard(LA, maxDimension):
    extensions = []

    # Check for every d from 2/3 (for n is even/odd, resp.) to n-2. For every d,
    # take an A extension and then a B extension if it exists.
    n = LA.dimension
    for d in range(2+(n%2), n-1, 2):
        ext = extendLA(LA, d, extType='A')
        extensions.append(ext)
        extendRecursively(ext, extensions, maxDimension)
        d += 2

    return extensions
    
def create_L(dimension):
    LA = LieAlgebra(dimension)
    LA.d = 2
    LA.type = 'A'
    for j in range(2, dimension):
        LA.add_bracket(1, j, j + 1, 1)
    return LA


def __main__():
    max_dim = 8
    Ls = [create_L(n) for n in range(4, max_dim)]
#    Ls = [create_L(n) for n in [8]]

    found = []

    for L in Ls:
        print('Extending {}'.format(L.simple_repr()))
        found.extend(extendStandard(LA=L, maxDimension=max_dim))

    # sort algebras in order of dimension and output
    #found.sort(key=lambda la: (la.dimension, la.d, la.extension, la.type))
    found.sort(key=lambda la: (la.type, la.dimension, la.d, la.extension))

    #la_filter = Filter(dimension=[6, 8, 10, 12], type='B')
    #la_filter = Filter(dimension=range(6, 10), type='B', U_matrix=False)
    la_filter = Filter() # Everything

    found = list(filter(lambda la : la.matches(la_filter), found))

    f = open('output/output.tex', 'w')
    f.write(print_latex(found))
    f.close()

    f = open('output/output.csv', 'w')
    f.write(print_csv(found))
    f.close()

    f = open('output/grid.dot', 'w') 
    f.write('digraph G {\n')
    for la in found:
        if la.parent:
            f.write('  "{}" -> "{}"\n'.format(la.parent, la))
    f.write('}\n')
    f.close()

__main__()
