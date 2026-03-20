import numpy as np
from src.rheology.utils import poly_gcd, clean_poly

class RheoModel:
    def __init__(self, N, D):
        self.N = clean_poly(N)
        self.D = clean_poly(D)

    def simplify(self):
        """Kiejti a számláló és a nevező legnagyobb közös polinom osztóját."""
        gcd = poly_gcd(self.N, self.D)
        if len(gcd) > 1 or (len(gcd) == 1 and gcd[0] != 1.0):
            # Maradék nélküli osztás a közös tényezővel
            self.N, _ = np.polydiv(self.N, gcd)
            self.D, _ = np.polydiv(self.D, gcd)
            self.N = clean_poly(self.N)
            self.D = clean_poly(self.D)
        
        # Normalizáljuk az egyenletet úgy, hogy a szigma legkisebb rendű tagja 1 legyen
        if len(self.D) > 0 and self.D[-1] != 0:
            norm_factor = self.D[-1]
            self.N = self.N / norm_factor
            self.D = self.D / norm_factor

    def get_ode(self):
        """Visszaadja a formázott differenciálegyenlet stringet."""
        self.simplify()
        
        def format_side(poly, var):
            terms = []
            degree = len(poly) - 1
            for i, coef in enumerate(poly):
                if abs(coef) < 1e-8:
                    continue
                deriv = degree - i
                
                coef_str = f"{coef:g}" if abs(coef - 1.0) > 1e-8 or deriv == 0 else ""
                if deriv == 0:
                    var_str = var
                elif deriv == 1:
                    var_str = f"d{var}/dt"
                else:
                    var_str = f"d^{deriv}{var}/dt^{deriv}"
                
                term = f"{coef_str}{var_str}" if coef_str else var_str
                terms.append(term)
            return " + ".join(terms) if terms else "0"

        sigma_side = format_side(self.D, "σ")
        epsilon_side = format_side(self.N, "ε")
        return f"{sigma_side} = {epsilon_side}"

class Spring(RheoModel):
    def __init__(self, E):
        # Q(s) = E / 1
        super().__init__(N=[E], D=[1.0])

class Dashpot(RheoModel):
    def __init__(self, eta):
        # Q(s) = (eta * s) / 1
        super().__init__(N=[eta, 0.0], D=[1.0])

class Parallel(RheoModel):
    def __init__(self, m1: RheoModel, m2: RheoModel):
        # N_p = N1*D2 + N2*D1
        # D_p = D1*D2
        n1d2 = np.convolve(m1.N, m2.D)
        n2d1 = np.convolve(m2.N, m1.D)
        
        # Összeadásnál figyelni kell a tömbök hosszára, a numpy polyadd ezt megteszi
        N_new = np.polyadd(n1d2, n2d1)
        D_new = np.convolve(m1.D, m2.D)
        
        super().__init__(N_new, D_new)

class Series(RheoModel):
    def __init__(self, m1: RheoModel, m2: RheoModel):
        # N_s = N1*N2
        # D_s = N1*D2 + N2*D1
        N_new = np.convolve(m1.N, m2.N)
        
        n1d2 = np.convolve(m1.N, m2.D)
        n2d1 = np.convolve(m2.N, m1.D)
        D_new = np.polyadd(n1d2, n2d1)
        
        super().__init__(N_new, D_new)