#------------------------------------------------------------
# To run the unit tests, type "py.test" at the command line.
#------------------------------------------------------------

from sympy import Symbol

from LnFinderWithCoefficients import LieAlgebra
from LnFinderWithCoefficients import RecursiveExtension
from LnFinderWithCoefficients import CreateL

def check_1_A_2_5(la):
    assert la.JacobiToTest == []
    assert la.brackets[1, 2].index == 3
    assert la.brackets[1, 2].alpha == 1
    assert la.brackets[1, 3].index == 4
    assert la.brackets[1, 3].alpha == 1
    assert la.brackets[1, 4].index == 5
    assert la.brackets[1, 4].alpha == 1
    assert la.brackets[2, 3].index == 5
    assert la.brackets[2, 3].alpha == Symbol("alpha_2,3^5")

def check_2_A_2_6(la):
    assert la.brackets[1, 2].index == 3
    assert la.brackets[1, 2].alpha == 1
    assert la.brackets[1, 3].index == 4
    assert la.brackets[1, 3].alpha == 1
    assert la.brackets[1, 4].index == 5
    assert la.brackets[1, 4].alpha == 1
    assert la.brackets[1, 5].index == 6
    assert la.brackets[1, 5].alpha == 1
    assert la.brackets[2, 3].index == 5
    assert la.brackets[2, 3].alpha == Symbol("alpha_2,3^5")
    assert la.brackets[2, 4].index == 6
    assert la.brackets[2, 4].alpha == Symbol("alpha_2,4^6")

def check_4_A_2_8(la):
    assert la.brackets[1, 2].index == 3
    assert la.brackets[1, 2].alpha == 1
    assert la.brackets[1, 3].index == 4
    assert la.brackets[1, 3].alpha == 1
    assert la.brackets[1, 4].index == 5
    assert la.brackets[1, 4].alpha == 1
    assert la.brackets[1, 5].index == 6
    assert la.brackets[1, 5].alpha == 1
    assert la.brackets[1, 6].index == 7
    assert la.brackets[1, 6].alpha == 1
    assert la.brackets[1, 7].index == 8
    assert la.brackets[1, 7].alpha == 1
    assert la.brackets[2, 3].index == 5
    assert la.brackets[2, 3].alpha == Symbol("alpha_2,3^5")
    assert la.brackets[2, 4].index == 6
    assert la.brackets[2, 4].alpha == Symbol("alpha_2,4^6")
    assert la.brackets[2, 5].index == 7
    assert la.brackets[2, 5].alpha == Symbol("alpha_2,5^7")
    assert la.brackets[2, 6].index == 8
    assert la.brackets[2, 6].alpha == Symbol("alpha_2,6^8")
    assert la.brackets[3, 4].index == 7
    assert la.brackets[3, 4].alpha == Symbol("alpha_3,4^7")
    assert la.brackets[3, 5].index == 8
    assert la.brackets[3, 5].alpha == Symbol("alpha_3,5^8")

def test_answer():
    #L4 = LieAlgebra(name="L", dimension=4)
    L4 = CreateL(4)
    algebras = RecursiveExtension(LA=L4, depth=5, output=False)
    assert len(algebras) == 10

    found = 0
    for la in algebras:
        if (la.extension == 1 and la.type == 'A' and
            la.d == 2 and la.dimension == 5):
            check_1_A_2_5(la)
            found += 1
        if (la.extension == 2 and la.type == 'A' and
            la.d == 2 and la.dimension == 6):
            check_2_A_2_6(la)
            found += 1
        if (la.extension == 4 and la.type == 'A' and
            la.d == 2 and la.dimension == 8):
            check_4_A_2_8(la)
            found += 1

    # did all checks
    assert found == 3
