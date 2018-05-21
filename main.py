from sympy import *

from LnFinderWithCoefficients import LieAlgebra
from LnFinderWithCoefficients import RecursiveExtension
from LnFinderWithCoefficients import ExtendL
from LnFinderWithCoefficients import *

L4 = create_L(4)
L5 = create_L(5)
L6 = create_L(6)
L7 = create_L(7)
L8 = create_L(8)
L9 = create_L(9)
L10 = create_L(10)
L11 = create_L(11)
L12 = create_L(12)
L13 = create_L(13)
L14 = create_L(14)
L15 = create_L(15)

Ls = [ L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14, L15 ]

found = []
max_dim = 14
for L in Ls:
    if L.dimension < max_dim:
        found.extend(ExtendL(LA=L, depth=max_dim-L.dimension))


# sort algebras in order of dimension and output
#found.sort(key=lambda la: (la.dimension, la.d, la.extension, la.type))
found.sort(key=lambda la: (la.type, la.dimension, la.d, la.extension))

#la_filter = Filter(dimension=[6, 8, 10, 12], type='B')
#la_filter = Filter(dimension=range(6, 10), type='B', U_matrix=False)
la_filter = Filter()

found = list(filter(lambda la : la.matches(la_filter), found))

f = open('test.tex', 'w')
f.write(print_latex(found))
f.close()

f = open('grid.dot', 'w') 
f.write('digraph G {\n')
for la in found:
    if la.parent:
        f.write('  "{}" -> "{}"\n'.format(la.parent, la))
f.write('}\n')
f.close()
