from sympy import *

from LnFinderWithCoefficients import LieAlgebra
from LnFinderWithCoefficients import RecursiveExtension

print("***********************************************************************")
print("***********************************************************************")
print("***********************************************************************")

L4 = LieAlgebra(name="L", dimension=4)
algebras = RecursiveExtension(LA=L4, depth=5, output=True)
