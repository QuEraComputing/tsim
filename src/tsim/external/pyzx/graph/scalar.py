# PyZX - Python library for quantum circuit rewriting 
#        and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains the Scalar class used to represent a global scalar in a Graph."""

import math
import cmath
import copy
from fractions import Fraction
from typing import List, Set, Dict
import json

from ..utils import FloatInt, FractionLike

__all__ = ['Scalar']

def cexp(val) -> complex:
    return cmath.exp(1j*math.pi*val)

unicode_superscript = {
    '0': '⁰',
    '1': '¹',
    '2': '²',
    '3': '³',
    '4': '⁴',
    '5': '⁵',
    '6': '⁶',
    '7': '⁷',
    '8': '⁸',
    '9': '⁹'
}

unicode_fractions = {
    Fraction(1,4): '¼',
    Fraction(1,2): '½',
    Fraction(3,4): '¾',
}

class DyadicNumber:
    k: int
    a: int
    b: int
    c: int
    d: int

    def __init__(self, k: int = 0, a: int = 0, b: int = 0, c: int = 0, d: int = 0):

        while a % 2 == 0 and b % 2 == 0 and c % 2 == 0 and d % 2 == 0:
            a //= 2
            b //= 2
            c //= 2
            d //= 2
            k -= 1

        self.k = k
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def to_complex(self) -> complex:
        return (
            self.a
            + self.b * cmath.exp(1j * math.pi / 4)
            + self.c * 1j
            + self.d * cmath.exp(-1j * math.pi / 4)
        ) / (2**self.k)

    def __mul__(self, other: "DyadicNumber") -> "DyadicNumber":
        return DyadicNumber(
            self.k + other.k,
            self.a * other.a + self.b * other.d - self.c * other.c + self.d * other.b,
            self.a * other.b + self.b * other.a + self.c * other.d + self.d * other.c,
            self.a * other.c + self.b * other.b + self.c * other.a - self.d * other.d,
            self.a * other.d - self.b * other.c - self.c * other.b + self.d * other.a,
        )

    @staticmethod
    def sqrt2() -> "DyadicNumber":
        return DyadicNumber(0, 0, 1, 0, 1)

    @staticmethod
    def one() -> "DyadicNumber":
        return DyadicNumber(0, 1, 0, 0, 0)

    def conjugate(self) -> "DyadicNumber":
        return DyadicNumber(self.k, self.a, self.d, -self.c, self.b)

    def copy(self) -> "DyadicNumber":
        return DyadicNumber(self.k, self.a, self.b, self.c, self.d)

class SpiderPair:
    def __init__(self, alpha, beta, paramsA, paramsB):
        self.alpha:   int      = alpha   # phase n of n*pi/4 (i.e. = 0,1,2,3,4,5,6,7) #TODO: Can change this to fraction if we want to support any alpha
        self.beta:    int      = beta    # phase n of n*pi/4 (i.e. = 0,1,2,3,4,5,6,7)
        self.paramsA: Set[str] = paramsA # the set of XOR'd variables added to alpha
        self.paramsB: Set[str] = paramsB # the set of XOR'd variables added to beta
        # gamma   = (alpha + beta) % 2
        # paramsC = XOR(paramsA, paramsB)

class Scalar(object):
    """Represents a global scalar for a Graph instance."""
    def __init__(self) -> None:
        self.power2: int = 0 # Stores power of square root of two
        self.phase: Fraction = Fraction(0) # Stores complex phase of the number
        #self.phasevars: Dict[str,int] = dict() # Stores the phase variables. e.g. variable p: stores px/2, where x=0,1,2,3;  Dict[p,x]
        self.phasevars_pi: Set[str] = set() # Stores the basic phase variable terms with pi coefficients (c=pi)
        self.phasevars_pi_pair: List[List[Set[str]]] = [] # Stores the AND-pair phase variable term pairs with pi coefficients (c=pi)
        self.phasevars_halfpi: Dict[List[Set[str]]] = dict() # Stores the phase variable terms with +-pi/2 coeffs: c[terms[vars]], where c=1 or 3 (pi/2 or 3pi/2); These arise from lcomp's
        self.phasepairs: List[SpiderPair] = [] # Stores list of spider-pairs
        self.phasenodes: List[FractionLike] = [] # Stores list of legless spiders, by their phases.
        self.phasenodevars: List[Set[str]] = [] # Stores the added parameters of the legless spider phases
        self.floatfactor: DyadicNumber = DyadicNumber.one()
        self.approximate_floatfactor: complex = 1.0 + 0.0j
        self.is_unknown: bool = False # Whether this represents an unknown scalar value
        self.is_zero: bool = False

    def __repr__(self) -> str:
        return "Scalar({})".format(str(self))

    def __str__(self) -> str:
        if self.is_unknown:
            return "UNKNOWN"
        s = "{0.real:.2f}{0.imag:+.2f}i = ".format(self.to_number())
        if self.floatfactor.to_complex() != 1.0:
            s += "{0.real:.2f}{0.imag:+.2f}i".format(self.floatfactor.to_complex())
        if self.phase:
            s += "exp({}ipi)".format(str(self.phase))
        s += "sqrt(2)^{:d}".format(self.power2)
        for node in self.phasenodes:
            s += "(1+exp({}ipi))".format(str(node))
        return s

    def print_attrs(self):
        s = f"phasenode: {self.phasenodes}, {self.phasenodevars}"
        s += f"\nphasevars_pi_pair: {self.phasevars_pi_pair}"
        s += f"\nphasepairs: {[(pp.alpha, pp.beta, pp.paramsA, pp.paramsB) for pp in self.phasepairs]}"
        s += f"\nphasevars_halfpi: {self.phasevars_halfpi}"
        s += f"\nphase: {self.phase}"
        s += f"\npower2: {self.power2}"
        s += f"\nfloatfactor: {self.floatfactor.to_complex()}"
        s += f"\napproximate_floatfactor: {self.approximate_floatfactor}"
        s += f"\nis_zero: {self.is_zero}"
        print(s)
        return s

    
    def print_expr(self) -> str:
        scalar = self
        scalar_str = ""

        def format_phase_str(alpha, params):
            a_str = str(alpha) if alpha != 0 else ""
            for vars in params:
                a_str += f"+{vars}"
            if len(a_str) > 0 and a_str[0] == "+":
                a_str = a_str[1:]
            return a_str

        for const, vars in zip(scalar.phasenodes, scalar.phasenodevars):
            scalar_str += f"(1 + exp(iπ {format_phase_str(const, vars)}))"

        for pp in scalar.phasepairs:
            a_str = format_phase_str(pp.alpha / 4, pp.paramsA)
            b_str = format_phase_str(pp.beta / 4, pp.paramsB)

            scalar_str += (
                f"(1 + exp(iπ {a_str}) + exp(iπ {b_str}) - exp(iπ ({a_str} + {b_str})))"
            )

        for c in [1, 3]:
            if c not in scalar.phasevars_halfpi:
                continue
            for vars in scalar.phasevars_halfpi[c]:
                a_str = " + ".join(vars)
                scalar_str += f"exp(iπ {a_str} * {c}/4)"

        for pp in scalar.phasevars_pi_pair:
            if len(pp[0]) == 0 or len(pp[1]) == 0:
                continue
            a_str = " + ".join(pp[0])
            b_str = " + ".join(pp[1])

            scalar_str += f"exp(iπ {a_str} * {b_str})"

        if scalar.power2 % 2 == 0:
            if scalar.power2 > 0:
                scalar_str += f" * {2 ** (scalar.power2 // 2)}"
            elif scalar.power2 < 0:
                scalar_str += f" / {2 ** ((-scalar.power2) // 2)}"
        else:
            scalar_str += f" * sqrt(2) ** {scalar.power2}"

        print(scalar_str)
        return scalar_str

    def evaluate_scalar(self, vals: dict[str, Fraction]) -> complex:
        scalar = self
        number = 1

        vals["1"] = Fraction(1)

        # phase nodes
        for const, vars in zip(scalar.phasenodes, scalar.phasenodevars):
            number *= 1 + cexp(const + sum(vals[var] for var in vars))

        # phase pairs
        for pp in scalar.phasepairs:
            psi = pp.alpha / 4 + sum(vals[var] for var in pp.paramsA)
            phi = pp.beta / 4 + sum(vals[var] for var in pp.paramsB)
            number *= 1 + cexp(psi) + cexp(phi) - cexp(psi + phi)

        # half-pi
        for c in [1, 3]:
            if c not in scalar.phasevars_halfpi:
                continue
            for vars in scalar.phasevars_halfpi[c]:
                number *= cexp(sum(vals[var] for var in vars) % 2 * c / 2)

        # pi-pair
        for pp in scalar.phasevars_pi_pair:
            psi = sum(vals[var] for var in pp[0])
            phi = sum(vals[var] for var in pp[1])
            number *= cexp(psi * phi)

        if scalar.is_zero:
            return 0

        number *= cexp(scalar.phase)

        number *= math.sqrt(2) ** scalar.power2
        number *= scalar.floatfactor.to_complex()
        number *= scalar.approximate_floatfactor

        return number

    def __complex__(self) -> complex:
        return self.to_number()

    def copy(self, conjugate: bool = False) -> 'Scalar':
        s = Scalar()
        s.power2 = self.power2
        s.phase = self.phase if not conjugate else -self.phase
        s.phasevars_pi = copy.copy(self.phasevars_pi)
        
        #TEMP:
        s.phasevars_pi_pair: List[List[Set[str]]] = []
        for i in self.phasevars_pi_pair:
            psA = copy.copy(i[0])
            psB = copy.copy(i[1])
            s.phasevars_pi_pair.append([psA,psB])
        
        #TEMP:
        s.phasevars_halfpi: Dict[List[Set[str]]] = dict()
        if 1 in self.phasevars_halfpi:
            for i in self.phasevars_halfpi[1]:
                if 1 not in s.phasevars_halfpi: s.phasevars_halfpi[1] = list()
                s.phasevars_halfpi[1].append(i)
        if 3 in self.phasevars_halfpi:
            for i in self.phasevars_halfpi[3]:
                if 3 not in s.phasevars_halfpi: s.phasevars_halfpi[3] = list()
                s.phasevars_halfpi[3].append(i)
        
        s.phasepairs = copy.copy(self.phasepairs)
        s.phasenodes = copy.copy(self.phasenodes) if not conjugate else [-p for p in self.phasenodes]
        s.phasenodevars = copy.copy(self.phasenodevars)
        s.floatfactor = self.floatfactor.copy() if not conjugate else self.floatfactor.conjugate()
        s.approximate_floatfactor = self.approximate_floatfactor if not conjugate else self.approximate_floatfactor.conjugate()
        s.is_unknown = self.is_unknown
        s.is_zero = self.is_zero
        return s

    def to_number(self) -> complex:
        if self.is_zero: return 0
        val = cexp(self.phase)
        for node in self.phasenodes: # Node should be a Fraction
            val *= 1+cexp(node)
        for sp in self.phasepairs:
            # alpha, beta are stored as integer multiples of pi/4
            a = Fraction(sp.alpha, 4)
            b = Fraction(sp.beta, 4)
            val *= 1 + cexp(a) + cexp(b) - cexp(a + b)
        val *= math.sqrt(2)**self.power2
        return val*self.floatfactor.to_complex() * self.approximate_floatfactor

    def to_latex(self) -> str:
        """Converts the Scalar into a string that is compatible with LaTeX."""
        if self.is_zero: return "0"
        elif self.is_unknown: return "Unknown"
        f = self.floatfactor.to_complex()
        for node in self.phasenodes:
            f *= 1+cexp(node)
        if self.phase == 1:
            f *= -1

        s = "$"
        if abs(f+1) < 0.001: #f \approx -1
            s += "-"
        elif abs(f-1) > 0.0001: #f \neq 1
            s += str(self.floatfactor.to_complex())
        if self.power2 != 0:
            s += r"\sqrt{{2}}^{{{:d}}}".format(self.power2)
        if self.phase not in (0,1):
            s += r"\exp(i~\frac{{{:d}\pi}}{{{:d}}})".format(self.phase.numerator,self.phase.denominator)
        s += "$"
        if s == "$$": return ""
        return s

    def to_unicode(self) -> str:
        """Returns a representation of the scalar that uses unicode
        to represent pi's and sqrt's."""
        if self.is_zero: return "0"
        elif self.is_unknown: return "Unknown"
        f = self.floatfactor.to_complex()
        for node in self.phasenodes:
            f *= 1+cexp(node)
        phase = Fraction(self.phase)
        if self.phase >= 1:
            f *= -1
            phase -= 1

        if abs(f+1) > 0.001 and abs(f-1) > 0.001:
            return str(f)

        s = ""
        if abs(f+1) < 0.001: #f \approx -1
            s += "-"
        if self.power2 != 0:
            s += r"√2"
            if self.power2 < 0:
                s += "⁻"
            val = str(abs(self.power2))
            s += "".join([unicode_superscript[i] for i in val])
        if phase != 0:
            s += "exp(i"
            if phase in unicode_fractions:
                s += unicode_fractions[phase] + "π)"
            else:
                s += "{:d}/{:d}π)".format(phase.numerator,phase.denominator)
        return s

    def to_json(self) -> str:
        d = {"power2": self.power2, "phase": str(self.phase)}
        if abs(self.floatfactor.to_complex() - 1) > 0.00001:
            d["floatfactor"] =  self.floatfactor.to_complex()
        if self.phasenodes:
            d["phasenodes"] = [str(p) for p in self.phasenodes]
        if self.is_zero:
            d["is_zero"] = self.is_zero
        if self.is_unknown:
            d["is_unknown"] = self.is_unknown,
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> 'Scalar':
        d = json.loads(s)
        d["phase"] = Fraction(d["phase"])
        if "phasenodes" in d:
            d["phasenodes"] = [Fraction(p) for p in d["phasenodes"]]
        scalar = Scalar()
        scalar.__dict__.update(d)
        return scalar

    def set_unknown(self) -> None:
        self.is_unknown = True
        self.phasenodes = []

    def add_power(self, n) -> None:
        """Adds a factor of sqrt(2)^n to the scalar."""
        self.power2 += n
    def add_phase(self, phase: FractionLike) -> None:
        """Multiplies the scalar by a complex phase."""
        self.phase = (self.phase + phase) % 2
    def add_phase_vars_halfpi(self, ps:Set[str], c:int) -> None: # These terms arise from lcomp's
        """Adds a term of XOR'd phase variables to the multiplier, for a +-pi/2 coefficient"""
        if (len(ps)>0): # Don't bother adding empty sets (i.e. those with no parameters)
            if c not in self.phasevars_halfpi: self.phasevars_halfpi[c] = list()
            self.phasevars_halfpi[c].append(ps)
    def add_phase_vars_pi(self, psA:Set[str]) -> None:
        """Adds XOR'd phase variables to the multiplier, for a pi coefficient"""
        self.phasevars_pi = self.phasevars_pi.symmetric_difference(psA)
    def add_phase_vars_pi_pair(self, psA:Set[str], psB:Set[str]) -> None:
        """Adds a term of XOR'd phase variable set pairs to the multiplier, for a pi coefficient"""
        self.phasevars_pi_pair.append([psA, psB])
    def add_phase_pair(self, alpha: FractionLike, beta: FractionLike, paramsA: Set[str], paramsB: Set[str]) -> None:
        """Add a new spider-pair scalar term"""
        a = int(alpha*4) # Convert via alpha=a*pi/4
        b = int(beta*4)  # Convert via  beta=b*pi/4
        assert a == alpha * 4, f"alpha: {alpha}, a: {a}"
        assert b == beta * 4, f"beta: {beta}, b: {b}"
        sp = SpiderPair(a, b, paramsA, paramsB)
        self.phasepairs.append(sp)
        
    def add_node(self, node: FractionLike, node_params: Set[str]) -> None:
        """A solitary spider with a phase ``node`` is converted into the
        scalar 1+e^(i*pi*node)."""
        if (node == 0 and len(node_params) == 0):
            self.power2 += 2
        else:
            assert node.denominator in [1,2,4]
            self.phasenodes.append(node)
            self.phasenodevars.append(node_params) # XOR
        if (node == 1 and len(node_params) == 0): self.is_zero = True
    def add_float(self,f: DyadicNumber) -> None:
        self.floatfactor *= f

    def mult_with_scalar(self, other: 'Scalar') -> None:
        """Multiplies two instances of Scalar together."""
        self.power2 += other.power2
        self.phase = (self.phase +other.phase)%2
        self.phasenodes.extend(other.phasenodes)
        self.floatfactor *= other.floatfactor
        self.approximate_floatfactor *= other.approximate_floatfactor
        if other.is_zero: self.is_zero = True
        if other.is_unknown: self.is_unknown = True

    def add_spider_pair(self, p1: FractionLike,p2: FractionLike, params1:Set[str], params2:Set[str]) -> None:
        """Add the scalar corresponding to a connected pair of spiders (p1)-H-(p2)."""
        # These if statements look quite arbitrary, but they are just calculations of the scalar
        # of a pair of connected single wire spiders of opposite colors.
        # We make special cases for Clifford phases and pi/4 phases.
        
        #if (p2 in (0,1) and len(params2) == 0):
        #    p1, p2 = p2, p1
        #    params1, params2 = params2, params1
        #if (p1 == 0 and len(params1) == 0):
        #    self.add_power(1)
        #    return
        #elif (p1 == 1 and len(params1) == 0):
        #    self.add_power(1)
        #    self.add_phase(p2)
        #    return
        
        #if p2.denominator == 2:
        #    p1, p2 = p2, p1
        #    params1, params2 = params2, params1
        #if p1 == Fraction(1,2):
        #    self.add_phase(Fraction(1,4))
        #    self.add_node((p2-Fraction(1,2))%2)
        #    return
        #elif p1 == Fraction(3,2):
        #    self.add_phase(Fraction(7,4))
        #    self.add_node((p2-Fraction(3,2))%2)
        #    return
        #if (p1 + p2) % 2 == 0:
        #    if p1.denominator == 4:
        #        if p1.numerator in (3,5):
        #            self.add_phase(Fraction(1))
        #        return
        #    self.add_power(1)
        #    self.add_float(math.cos(p1))
        #    return
        
        # Generic case:
        self.add_power(-1)
        self.add_phase_pair(p1,p2,params1,params2)
        #self.add_float(1+cexp(p1)+cexp(p2) - cexp(p1+p2))
        return
