import cmath
import math

try:
    import mpmath
except ImportError:
    mpmath = None


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
    def from_complex(
        z: complex, max_k: int = 20, precision: int = 15
    ) -> "DyadicNumber":
        if mpmath is None:
            raise ImportError("mpmath is required for from_complex")

        # Use the provided precision.
        # Note: If z is a standard float/complex, it only has ~15-16 digits of precision.
        # Using a much higher precision for PSLQ (e.g. 100) will fail because the
        # float input contains noise at the 16th digit relative to the true algebraic value.
        with mpmath.workdps(precision):
            if isinstance(z, complex):
                re_z = mpmath.mpf(z.real)
                im_z = mpmath.mpf(z.imag)
            else:
                re_z = mpmath.mpf(z)
                im_z = mpmath.mpf(0)

            # Relation for Real part: q1*Re(z) + q2*1 + q3*sqrt(2) = 0
            # We expect Re(z) = (2a + (b+d)sqrt(2)) / 2^{k+1}
            basis_re = [re_z, mpmath.mpf(1), mpmath.sqrt(2)]
            # PSLQ returns integers [q1, q2, q3]

            if abs(re_z) < 1e-12:
                rel_re = [1, 0, 0]
            else:
                rel_re = mpmath.pslq(basis_re)

            if rel_re is None:
                raise ValueError("PSLQ failed to find relation for real part", z)
            q1, q2, q3 = rel_re

            # Relation for Imag part: p1*Im(z) + p2*1 + p3*sqrt(2) = 0
            # We expect Im(z) = (2c + (b-d)sqrt(2)) / 2^{k+1}
            basis_im = [im_z, mpmath.mpf(1), mpmath.sqrt(2)]

            if abs(im_z) < 1e-12:
                rel_im = [1, 0, 0]
            else:
                rel_im = mpmath.pslq(basis_im)

            if rel_im is None:
                raise ValueError(
                    f"PSLQ failed to find relation for imaginary part for {z.imag}"
                )
            p1, p2, p3 = rel_im

            # Ensure positive denominators (q1 and p1 correspond to the z coefficient)
            if q1 < 0:
                q1, q2, q3 = -q1, -q2, -q3
            if p1 < 0:
                p1, p2, p3 = -p1, -p2, -p3

            # Handle cases where PSLQ returns 0 for the coefficient of interest
            # This happens if the value is 0 (since 1*0 + 0*1 + 0*sqrt(2) = 0 is found as 1,0,0 usually?
            # Actually if val=0, PSLQ on [0, 1, sqrt(2)] should return [1, 0, 0].
            # So q1=1, q2=0, q3=0.
            # If q1=0, it means it found a relation between 1 and sqrt(2) independent of z,
            # which is impossible as 1 and sqrt(2) are independent over rationals.
            # Unless precision is too low and it thinks 1 and sqrt(2) are related?
            # Or if z is huge.
            if q1 == 0:
                raise ValueError(
                    "Found zero denominator coefficient in PSLQ relation (real part)", z
                )
            if p1 == 0:
                raise ValueError(
                    "Found zero denominator coefficient in PSLQ relation (imag part)", z
                )

            # Find a common denominator D = 2^{k+1}
            # Start with lcm(q1, p1)
            common_denom = (q1 * p1) // math.gcd(q1, p1)

            if common_denom <= 0:
                raise ValueError("Invalid denominator")

            # Find smallest power of 2 >= common_denom
            # k_exp such that 2^k_exp >= common_denom
            k_exp = common_denom.bit_length()
            if (1 << (k_exp - 1)) == common_denom:
                k_exp -= 1

            # The algorithm requires D = 2^{k+1}. So let K_candidate = k_exp.
            # We iterate starting from K_candidate, increasing if necessary to satisfy parity constraints.

            # D corresponds to 2^{real_k + 1}.
            # We need to find real_k.
            # Let D_exp run from k_exp upwards.
            # Then k = D_exp - 1.

            for D_exp in range(k_exp, max_k + 2):
                D = 1 << D_exp
                k = D_exp - 1

                # Check divisibility of D by q1 and p1
                if D % q1 != 0 or D % p1 != 0:
                    continue

                # Re(z) = -(q2 + q3*sqrt(2))/q1 = (2a + (b+d)sqrt(2))/D
                # => 2a = -q2 * (D/q1)
                # => b+d = -q3 * (D/q1)

                fac_re = D // q1
                A_prime = -q2 * fac_re
                B_prime = -q3 * fac_re

                # Im(z) = -(p2 + p3*sqrt(2))/p1 = (2c + (b-d)sqrt(2))/D
                # => 2c = -p2 * (D/p1)
                # => b-d = -p3 * (D/p1)

                fac_im = D // p1
                C_prime = -p2 * fac_im
                D_prime = -p3 * fac_im

                # Constraints:
                # 1. A_prime is even (gives a)
                # 2. C_prime is even (gives c)
                # 3. B_prime + D_prime is even (gives b)
                # 4. B_prime - D_prime is even (gives d) - implied by 3

                if A_prime % 2 != 0:
                    continue
                if C_prime % 2 != 0:
                    continue
                if (B_prime + D_prime) % 2 != 0:
                    continue

                a = A_prime // 2
                c = C_prime // 2
                b = (B_prime + D_prime) // 2
                d = (B_prime - D_prime) // 2

                # Found a solution!
                return DyadicNumber(k, int(a), int(b), int(c), int(d))

        raise ValueError(f"Could not reconstruct dyadic number within max_k={max_k}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DyadicNumber):
            return False
        return (
            self.a == other.a
            and self.b == other.b
            and self.c == other.c
            and self.d == other.d
            and self.k == other.k
        )

    def __repr__(self):
        return (
            f"DyadicNumber(k={self.k}, a={self.a}, b={self.b}, c={self.c}, d={self.d})"
        )


def test_dyadic_number():
    d = DyadicNumber(1, 7, 3, 4, 5)
    n = d.to_complex() + 1e-14 * 1j
    print(f"Original: {d}")
    print(f"Complex: {n}")
    d2 = DyadicNumber.from_complex(n)
    print(f"Reconstructed: {d2}")

    assert abs(n - d2.to_complex()) < 1e-10, f"Value mismatch: {n} vs {d2.to_complex()}"
    assert d == d2, "Structure mismatch"
    print("Test passed for from_complex!")

    # Test real
    # z = (a + b*sqrt(2)) / 2^k
    # Try a=3, b=1, k=2 => (3 + sqrt(2))/4
    val_real = (3 + math.sqrt(2)) / 4
    d_real = DyadicNumber.from_complex(val_real)
    print(f"Real input: {val_real}")
    print(f"Reconstructed real: {d_real}")

    expected_real = DyadicNumber(2, 3, 1, 0, 1)
    assert abs(val_real - d_real.to_complex().real) < 1e-10
    assert d_real == expected_real
    print("Test passed for real!")

    # Test imag
    # z = i * (c + b*sqrt(2)) / 2^k
    # Try c=5, b=2, k=3 => i * (5 + 2*sqrt(2)) / 8
    val_imag_part = (5 + 2 * math.sqrt(2)) / 8
    val_imag = 1j * val_imag_part
    d_imag = DyadicNumber.from_complex(val_imag)
    print(f"Imag input: {val_imag}")
    print(f"Reconstructed imag: {d_imag}")

    expected_imag = DyadicNumber(3, 0, 2, 5, -2)
    assert abs(val_imag - d_imag.to_complex()) < 1e-10
    assert d_imag == expected_imag
    print("Test passed for imag!")


if __name__ == "__main__":
    # z = 4.22e-14 + 1j*1.27e+04
    z = 1j * 1.27e04
    z = 1500
    z = 12671.676759431628 / 8
    d = DyadicNumber.from_complex(z, max_k=1500)
    print(d)
    print(d.to_complex() - z)
