import numpy as np

# src/math/polynomials.py

def clean_poly(p, tol=1e-8):
    """Eltávolítja a numerikus zajból adódó vezető nullákat."""
    p = np.array(p, dtype=float)
    p[np.abs(p) < tol] = 0.0
    trimmed = np.trim_zeros(p, 'f')
    return trimmed if len(trimmed) > 0 else np.array([0.0])

def poly_gcd(p1, p2, tol=1e-8):
    """Euklideszi algoritmus polinomokra a közös gyökök kiejtéséhez."""
    a = clean_poly(p1)
    b = clean_poly(p2)
    
    while np.any(np.abs(b) > tol):
        _, r = np.polydiv(a, b)
        r = clean_poly(r, tol)
        a = b
        b = r
        
    # Főegyüttható normalizálása 1-re
    if len(a) > 0 and a[0] != 0:
        a = a / a[0]
    return clean_poly(a)