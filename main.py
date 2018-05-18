from sympy import *

from LnFinderWithCoefficients import LieAlgebra
from LnFinderWithCoefficients import RecursiveExtension
from LnFinderWithCoefficients import ExtendL
from LnFinderWithCoefficients import *

#print("***********************************************************************")
#print("***********************************************************************")
#print("***********************************************************************")

"""
L4 = LieAlgebra(name="L", dimension=4)
L5 = LieAlgebra(name="L", dimension=5)
L6 = LieAlgebra(name="L", dimension=6)
L7 = LieAlgebra(name="L", dimension=7)
L8 = LieAlgebra(name="L", dimension=8)
L9 = LieAlgebra(name="L", dimension=9)
L10 = LieAlgebra(name="L", dimension=10)
L11 = LieAlgebra(name="L", dimension=11)
L12 = LieAlgebra(name="L", dimension=12)
L13 = LieAlgebra(name="L", dimension=13)
L14 = LieAlgebra(name="L", dimension=14)
L15 = LieAlgebra(name="L", dimension=15)
"""
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
#Ls = [ L4 ]

found = []
for L in Ls:
    #ExtendL(LA=L, depth=10, output=True,
    #        output_filter=Filter([2,4], [10], [4,5,6], 'A'))

    # Document for Beau and Sara
    #found.extend(ExtendL(LA=L, depth=11, output=False,
    #                     output_filter=None))

    #found.extend(ExtendL(LA=L, depth=11))
    found.extend(ExtendL(LA=L, depth=11))

print('\\documentclass{article}\n\\setlength{\\parindent}{0cm} ' +
      '% Default is 15pt.')
print('\\usepackage{amsmath}')
print('\\setcounter{MaxMatrixCols}{30}')
print('\\begin{document}')

# sort algebras in order of dimension and output
found.sort(key=lambda la: (la.dimension, la.d, la.extension))

## filter for beau and sara
filter = Filter(dimension=[6, 8, 10, 12], type='B')
#filter = Filter(dimension=[6], type='B')
PrintFoundLieAlgebras(found, filter)

#filter = Filter(dimension=[8], type='A')
#PrintFoundLieAlgebras(found, filter)

#filter = Filter(dimension=[6, 7, 8, 9], type='A')
#PrintFoundLieAlgebras(found, filter)

print('\n\\end{document}\n')
