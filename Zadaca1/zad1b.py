import math
import random
import csv

U1_MIN = 0
U1_MAX = 512
U2_MIN = -100
U2_MAX = 1000

BROJ_CESTICA = 30
BROJ_ITERACIJA = 10000
C = 1.49445
W_START = 0.9
W_STEP = 0.00007
REFRESHING_GAP_START = 7


def izlaz(u1, u2):
    try:
        if u1 > u2:
            return math.sin(u1 + u2**2) * math.exp(abs(1 - math.sqrt(u1**2 + 2.5 * u2**2)))
        elif u1 - 2 * u2 <= 0.5:
            return u2 * math.cos(u1 * u2 + 2.3 * u1) - 0.6
        else:
            return 2 / (3 + math.sin(u2))
    except OverflowError:
        return float("inf")


def random_position():
    u1 = random.uniform(U1_MIN, U1_MAX)
    u2 = random.uniform(U2_MIN, U2_MAX)
    return [u1, u2]


def random_velocity():
    v1 = random.uniform(-(U1_MAX - U1_MIN), U1_MAX - U1_MIN)
    v2 = random.uniform(-(U2_MAX - U2_MIN), U2_MAX - U2_MIN)
    return [v1, v2]


def calculate_pci(i, ps):
    numerator = math.exp(10 * (i - 1) / (ps - 1)) - 1
    denominator = math.exp(10) - 1
    return 0.05 + 0.45 * (numerator / denominator)


def generate_pci_list():
    pci_list = []

    for i in range(1, BROJ_CESTICA + 1):
        pci = calculate_pci(i, BROJ_CESTICA)
        pci_list.append(pci)

    return pci_list


def compute_xb(i, particles, pci_list):
    xb = [0.0, 0.0]
    pci = pci_list[i]

    for d in range(2):
        p = random.random()

        if p < pci:
            xb[d] = particles[i]["xpb"][d]
        else:
            k = random.randrange(len(particles))
            while k == i:
                k = random.randrange(len(particles))

            if particles[i]["best_value"] < particles[k]["best_value"]:
                xb[d] = particles[i]["xpb"][d]
            else:
                xb[d] = particles[k]["xpb"][d]

    return xb


def create_particle():
    position = random_position()
    velocity = random_velocity()
    value = izlaz(position[0], position[1])

    particle = {
        "x": position,
        "v": velocity,
        "xpb": position.copy(),
        "best_value": value,
        "refreshing_gap": REFRESHING_GAP_START,
        "xb": [0.0, 0.0]
    }

    return particle


def create_swarm():
    particles = []

    for _ in range(BROJ_CESTICA):
        particle = create_particle()
        particles.append(particle)

    return particles


def initialize_xb_for_swarm(particles, pci_list):
    for i in range(len(particles)):
        particles[i]["xb"] = compute_xb(i, particles, pci_list)


def update_particle(particle, w, c):
    for d in range(2):
        r = random.random()
        particle["v"][d] = w * particle["v"][d] + c * r * (particle["xb"][d] - particle["x"][d])
        particle["x"][d] = particle["x"][d] + particle["v"][d]


def enforce_bounds(particle):
    u1 = particle["x"][0]
    u2 = particle["x"][1]

    if u1 < U1_MIN or u1 > U1_MAX or u2 < U2_MIN or u2 > U2_MAX:
        particle["x"] = random_position()
        particle["v"] = random_velocity()


def update_personal_best_and_refresh(i, particles, pci_list):
    particle = particles[i]
    current_value = izlaz(particle["x"][0], particle["x"][1])

    if current_value < particle["best_value"]:
        particle["xpb"] = particle["x"].copy()
        particle["best_value"] = current_value
        particle["refreshing_gap"] = REFRESHING_GAP_START
    else:
        particle["refreshing_gap"] -= 1

        if particle["refreshing_gap"] <= 0:
            particle["xb"] = compute_xb(i, particles, pci_list)
            particle["refreshing_gap"] = REFRESHING_GAP_START

    return current_value


def write_iteration_result(writer, iteration, best_u1, best_u2, best_value):
    writer.writerow([iteration, best_u1, best_u2, best_value])


def get_global_best(particles):
    best_particle = particles[0]

    for particle in particles:
        if particle["best_value"] < best_particle["best_value"]:
            best_particle = particle

    return best_particle


def pso():
    particles = create_swarm()
    pci_list = generate_pci_list()
    initialize_xb_for_swarm(particles, pci_list)

    w = W_START

    with open("rezultat.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["iteracija", "u1", "u2", "vrijednost"])

        for iteration in range(1, BROJ_ITERACIJA + 1):
            iteration_best_position = None
            iteration_best_value = float("inf")

            for i in range(len(particles)):
                update_particle(particles[i], w, C)
                enforce_bounds(particles[i])
                current_value = update_personal_best_and_refresh(i, particles, pci_list)

                if current_value < iteration_best_value:
                    iteration_best_value = current_value
                    iteration_best_position = particles[i]["x"].copy()

            write_iteration_result(
                writer,
                iteration,
                iteration_best_position[0],
                iteration_best_position[1],
                iteration_best_value
            )

            w = w - W_STEP

    global_best = get_global_best(particles)

    print("Najbolje pronađeno rješenje:")
    print(f"u1 = {global_best['xpb'][0]:.6f}")
    print(f"u2 = {global_best['xpb'][1]:.6f}")
    print(f"minimalna vrijednost funkcije = {global_best['best_value']}")


if __name__ == "__main__":
    pso()