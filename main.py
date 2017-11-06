from sympy import *

from LnFinderWithCoefficients import LieAlgebra
from LnFinderWithCoefficients import RecursiveExtension
from LnFinderWithCoefficients import ExtendL

print("***********************************************************************")
print("***********************************************************************")
print("***********************************************************************")


L4 = LieAlgebra(name="L", dimension=4)
L5 = LieAlgebra(name="L", dimension=5)
L6 = LieAlgebra(name="L", dimension=6)
L7 = LieAlgebra(name="L", dimension=7)
L8 = LieAlgebra(name="L", dimension=8)
L9 = LieAlgebra(name="L", dimension=9)
L10 = LieAlgebra(name="L", dimension=10)
L11 = LieAlgebra(name="L", dimension=11)

Ls = [ L4, L5 ]

#algebras = RecursiveExtension(LA=L4, depth=3, output=True)
#algebras = RecursiveExtension(LA=L5, depth=3, output=True)

for L in Ls:
    #RecursiveExtension(LA=L, depth=1, output=True)
    ExtendL(LA=L, depth=3, output=True)
