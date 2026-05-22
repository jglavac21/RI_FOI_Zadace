import csv
import os
import time
import numpy as np


PROBLEMI = [
    "MDL36CI.dat",
    "MDL49BL.dat",
    "MDL100CI.dat",
    "MDL144BL.dat",
]

ALPHA = 1.0
RHO = 0.02
BROJ_ITERACIJA = 500
SEED = 42

BROJ_POCETNIH_RJESENJA = 20

def ucitaj_problem(putanja):

    with open(putanja, "r", encoding="utf-8") as file:
        podaci = np.fromstring(file.read(), sep=" ", dtype=np.int64)

    n = int(podaci[0])

    ocekivani_broj = 1 + 2 * n * n
    if len(podaci) != ocekivani_broj:
        raise ValueError(
            f"Datoteka {putanja} nema očekivani broj vrijednosti. "
            f"Očekivano: {ocekivani_broj}, pronađeno: {len(podaci)}"
        )

    pocetak_d = 1
    kraj_d = pocetak_d + n * n

    pocetak_f = kraj_d
    kraj_f = pocetak_f + n * n

    D = podaci[pocetak_d:kraj_d].reshape((n, n))
    F = podaci[pocetak_f:kraj_f].reshape((n, n))

    return n, D, F


def izracunaj_cijenu(D, F, rjesenje):

    return int(np.sum(D * F[np.ix_(rjesenje, rjesenje)]))


def generiraj_pocetno_rjesenje(D, F, n, rng):

    najbolje_rjesenje = None
    najbolja_cijena = None

    for _ in range(BROJ_POCETNIH_RJESENJA):
        rjesenje = rng.permutation(n)
        cijena = izracunaj_cijenu(D, F, rjesenje)

        if najbolja_cijena is None or cijena < najbolja_cijena:
            najbolje_rjesenje = rjesenje.copy()
            najbolja_cijena = cijena

    return najbolje_rjesenje, najbolja_cijena


def izracunaj_tau_min(tau_max, n, alpha):

    p_best = 0.05
    korijen = p_best ** (1.0 / n)

    faktor = (1.0 - korijen) / ((n / 2.0 - 1.0) * korijen)

    tau_min = tau_max * (faktor ** (1.0 / alpha))

    return tau_min


def odaberi_sljedecu_sondu(feromoni, lokacija, dostupne_sonde, rng):

    tezine = feromoni[lokacija, dostupne_sonde] ** ALPHA
    suma = float(np.sum(tezine))

    if suma <= 0 or not np.isfinite(suma):
        indeks = int(rng.integers(len(dostupne_sonde)))
        return indeks

    kumulativno = np.cumsum(tezine)
    slucajna_vrijednost = rng.random() * suma

    indeks = int(np.searchsorted(kumulativno, slucajna_vrijednost, side="right"))

    if indeks >= len(dostupne_sonde):
        indeks = len(dostupne_sonde) - 1

    return indeks


def konstruiraj_rjesenje_mrava(feromoni, n, rng):

    dostupne_sonde = np.arange(n)
    rjesenje = np.empty(n, dtype=np.int64)

    for lokacija in range(n):
        indeks = odaberi_sljedecu_sondu(feromoni, lokacija, dostupne_sonde, rng)

        odabrana_sonda = dostupne_sonde[indeks]
        rjesenje[lokacija] = odabrana_sonda

        dostupne_sonde = np.delete(dostupne_sonde, indeks)

    return rjesenje


def mmas(problem_datoteka, broj_iteracija=BROJ_ITERACIJA):

    naziv_problema = os.path.splitext(os.path.basename(problem_datoteka))[0]
    csv_datoteka = f"rezultati_{naziv_problema}.csv"

    rng = np.random.default_rng(SEED)

    n, D, F = ucitaj_problem(problem_datoteka)

    broj_mrava = n

    najbolje_rjesenje, najbolja_cijena = generiraj_pocetno_rjesenje(D, F, n, rng)

    tau_max = 1.0 / (RHO * najbolja_cijena)
    tau_min = izracunaj_tau_min(tau_max, n, ALPHA)

    feromoni = np.full((n, n), tau_max, dtype=np.float64)

    print("=" * 70)
    print(f"Problem: {problem_datoteka}")
    print(f"n = {n}")
    print(f"Broj mrava = {broj_mrava}")
    print(f"Broj iteracija = {broj_iteracija}")
    print(f"Početna najbolja cijena = {najbolja_cijena}")
    print(f"CSV izlaz = {csv_datoteka}")
    print("=" * 70)

    pocetak_vrijeme = time.time()

    with open(csv_datoteka, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "iteracija",
            "najbolja_vrijednost_iteracije",
            "najbolja_vrijednost_globalno",
            "najbolje_rjesenje_globalno"
        ])

        for iteracija in range(1, broj_iteracija + 1):
            najbolje_rjesenje_iteracije = None
            najbolja_cijena_iteracije = None

            for _ in range(broj_mrava):
                rjesenje = konstruiraj_rjesenje_mrava(feromoni, n, rng)
                cijena = izracunaj_cijenu(D, F, rjesenje)

                if najbolja_cijena_iteracije is None or cijena < najbolja_cijena_iteracije:
                    najbolje_rjesenje_iteracije = rjesenje.copy()
                    najbolja_cijena_iteracije = cijena

            if najbolja_cijena_iteracije < najbolja_cijena:
                najbolje_rjesenje = najbolje_rjesenje_iteracije.copy()
                najbolja_cijena = najbolja_cijena_iteracije

                tau_max = 1.0 / (RHO * najbolja_cijena)
                tau_min = izracunaj_tau_min(tau_max, n, ALPHA)

            feromoni *= (1.0 - RHO)

            delta_tau = 1.0 / najbolja_cijena
            feromoni[np.arange(n), najbolje_rjesenje] += delta_tau

            np.clip(feromoni, tau_min, tau_max, out=feromoni)

            writer.writerow([
                iteracija,
                najbolja_cijena_iteracije,
                najbolja_cijena,
                " ".join(map(str, najbolje_rjesenje.tolist()))
            ])

            if iteracija == 1 or iteracija % 10 == 0 or iteracija == broj_iteracija:
                print(
                    f"Iteracija {iteracija:4d} | "
                    f"najbolje u iteraciji = {najbolja_cijena_iteracije} | "
                    f"globalno najbolje = {najbolja_cijena}"
                )

    kraj_vrijeme = time.time()
    trajanje = kraj_vrijeme - pocetak_vrijeme

    print()
    print(f"Završeno: {problem_datoteka}")
    print(f"Najbolja pronađena vrijednost: {najbolja_cijena}")
    print(f"Najbolje rješenje: {najbolje_rjesenje.tolist()}")
    print(f"Vrijeme izvođenja: {trajanje:.2f} sekundi")
    print(f"Rezultati zapisani u: {csv_datoteka}")
    print()

    return najbolja_cijena, najbolje_rjesenje


def main():

    for problem in PROBLEMI:
        if not os.path.exists(problem):
            print(f"Preskačem {problem} jer datoteka nije pronađena.")
            continue

        mmas(problem)


if __name__ == "__main__":
    main()