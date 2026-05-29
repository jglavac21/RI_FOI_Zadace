import math


W1 = [
    [0.2, -0.4, 0.1],
    [0.7, 0.3, -0.5],
    [-0.6, 0.8, 0.2],
    [0.1, -0.2, 0.9]
]


B1 = [0.1, -0.2, 0.05, 0.3]


W2 = [
    [0.5, -0.3, 0.8, 0.2],
    [-0.4, 0.6, 0.1, -0.7]
]


B2 = [0.2, -0.1]


def sigmoid(net):
    
    return 1 / (1 + math.exp(-net))


def linearna(net):
    
    return net


def ponderirana_suma(ulazi, tezine, bias):
    
    net = bias

    for x, w in zip(ulazi, tezine):
        net += x * w

    return net


def izracunaj_sloj(ulazi, matrica_tezina, biasi, aktivacijska_funkcija):
    
    izlazi_sloja = []

    for tezine_neurona, bias_neurona in zip(matrica_tezina, biasi):
        net = ponderirana_suma(ulazi, tezine_neurona, bias_neurona)
        izlaz = aktivacijska_funkcija(net)
        izlazi_sloja.append(izlaz)

    return izlazi_sloja


def unaprijedni_prolaz(ulazni_vektor):
    
    skriveni_izlazi = izracunaj_sloj(
        ulazni_vektor,
        W1,
        B1,
        sigmoid
    )

    izlazni_izlazi = izracunaj_sloj(
        skriveni_izlazi,
        W2,
        B2,
        linearna
    )

    return skriveni_izlazi, izlazni_izlazi


def srednja_kvadratna_pogreska(ocekivani_izlaz, dobiveni_izlaz):
    
    suma = 0

    for ocekivano, dobiveno in zip(ocekivani_izlaz, dobiveni_izlaz):
        suma += (ocekivano - dobiveno) ** 2

    return suma / len(ocekivani_izlaz)


def ispisi_rezultate(ulazni_vektor, skriveni_izlazi, izlazni_izlazi):
    
    print("Ulazni vektor:")
    print(ulazni_vektor)

    print("\nIzlazi skrivenog sloja:")
    for i, vrijednost in enumerate(skriveni_izlazi, start=1):
        print(f"h{i} = {vrijednost:.6f}")

    print("\nIzlazi neuronske mreže:")
    for i, vrijednost in enumerate(izlazni_izlazi, start=1):
        print(f"y{i} = {vrijednost:.6f}")


def dio_a():
    
    print("=" * 60)
    print("DIO A - Računanje izlaza neuronske mreže")
    print("=" * 60)

    ulazni_vektor = [0.5, -0.2, 0.8]

    skriveni_izlazi, izlazni_izlazi = unaprijedni_prolaz(ulazni_vektor)

    ispisi_rezultate(ulazni_vektor, skriveni_izlazi, izlazni_izlazi)


def dio_b():
    
    print("\n" + "=" * 60)
    print("DIO B - Usporedba s očekivanim izlazom")
    print("=" * 60)

    ulazni_vektor = [0.5, -0.2, 0.8]

    ocekivani_izlaz = [1.0, 0.0]

    skriveni_izlazi, dobiveni_izlaz = unaprijedni_prolaz(ulazni_vektor)

    ispisi_rezultate(ulazni_vektor, skriveni_izlazi, dobiveni_izlaz)

    print("\nOčekivani izlaz:")
    for i, vrijednost in enumerate(ocekivani_izlaz, start=1):
        print(f"o{i} = {vrijednost:.6f}")

    print("\nRazlika između očekivanog i dobivenog izlaza:")
    for i, (ocekivano, dobiveno) in enumerate(zip(ocekivani_izlaz, dobiveni_izlaz), start=1):
        razlika = ocekivano - dobiveno
        print(f"o{i} - y{i} = {razlika:.6f}")

    pogreska = srednja_kvadratna_pogreska(ocekivani_izlaz, dobiveni_izlaz)

    print(f"\nSrednja kvadratna pogreška, MSE = {pogreska:.6f}")


def main():
    
    dio_a()
    dio_b()


if __name__ == "__main__":
    main()