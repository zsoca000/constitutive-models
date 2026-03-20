# bin/main.py

from src.rheology.elements import Spring, Dashpot, Series, Parallel

def main():
    print("--- 1. Maxwell Modell (Soros rugó és csillapító) ---")
    E1, eta1 = 100.0, 50.0
    maxwell = Series(Spring(E1), Dashpot(eta1))
    print(f"E = {E1}, η = {eta1}")
    print(maxwell.get_ode())
    print()

    print("--- 2. Standard Lineáris Szilárd test (Zener Modell) ---")
    # Egy Maxwell elem párhuzamosan egy másik rugóval
    E2 = 200.0
    zener = Parallel(maxwell, Spring(E2))
    print(f"Párhuzamos ág: E = {E2}")
    print(zener.get_ode())
    print()

    print("--- 3. Burgers Modell ---")
    # Egy Maxwell és egy Kelvin-Voigt (párhuzamos) elem sorba kötve
    kelvin = Parallel(Spring(300.0), Dashpot(150.0))
    burgers = Series(maxwell, kelvin)
    print("Maxwell + Kelvin-Voigt sorba kötve")
    print(burgers.get_ode())

if __name__ == "__main__":
    main()