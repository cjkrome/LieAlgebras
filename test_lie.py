from LnFinderWithCoefficients import LieAlgebra
from LnFinderWithCoefficients import RecursiveExtension

def test_answer():
    L4 = LieAlgebra(name="L", dimension=4)
    algebras = RecursiveExtension(LA=L4, depth=5, output=False)
    assert len(algebras) == 10
    #assert 3 == 5
