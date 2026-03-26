import math
import random
import csv
import subprocess
import re

X1_MIN = 0
X1_MAX = 3
X2_MIN = 0.0001
X2_MAX = 0.99
X3_MIN = 5
X3_MAX = 100000

DIMENZIJA = 3

BROJ_CESTICA = 30
BROJ_ITERACIJA = 10000
C = 1.49445
W_START = 0.9
W_STEP = 0.00007
REFRESHING_GAP_START = 7
MAX_FLOAT = 1.7976931348623157e308


def evaluate_solution(x1, x2, x3):
    command = f'Simulacija.exe {x1} {x2} {x3}'

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        shell=True
    )

    output = result.stdout.strip()

    matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', output)

    if not matches:
        raise ValueError(f"Nije pronađena numerička vrijednost u izlazu programa: {output}")

    return float(matches[-1])


def random_position():
    x1 = random.uniform(X1_MIN, X1_MAX)
    x2 = random.uniform(X2_MIN, X2_MAX)
    x3 = random.uniform(X3_MIN, X3_MAX)
    return [x1, x2, x3]


def random_velocity():
    v1 = random.uniform(-(X1_MAX - X1_MIN), X1_MAX - X1_MIN)
    v2 = random.uniform(-(X2_MAX - X2_MIN), X2_MAX - X2_MIN)
    v3 = random.uniform(-(X3_MAX - X3_MIN), X3_MAX - X3_MIN)
    return [v1, v2, v3]


def create_particle():
    position = random_position()
    velocity = random_velocity()
    value = evaluate_solution(position[0], position[1], position[2])

    particle = {
        "x": position,
        "v": velocity,
        "xpb": position.copy(),
        "best_value": value,
        "refreshing_gap": REFRESHING_GAP_START,
        "xb": [0.0, 0.0, 0.0]
    }

    return particle


def create_swarm():
    particles = []

    for _ in range(BROJ_CESTICA):
        particle = create_particle()
        particles.append(particle)

    return particles


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
    xb = [0.0, 0.0, 0.0]
    pci = pci_list[i]

    for d in range(DIMENZIJA):
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


def initialize_xb_for_swarm(particles, pci_list):
    for i in range(len(particles)):
        particles[i]["xb"] = compute_xb(i, particles, pci_list)


def update_particle(particle, w, c):
    for d in range(DIMENZIJA):
        r = random.random()
        particle["v"][d] = w * particle["v"][d] + c * r * (particle["xb"][d] - particle["x"][d])
        particle["x"][d] = particle["x"][d] + particle["v"][d]


def enforce_bounds(particle):
    x1 = particle["x"][0]
    x2 = particle["x"][1]
    x3 = particle["x"][2]

    if (
        x1 < X1_MIN or x1 > X1_MAX or
        x2 < X2_MIN or x2 > X2_MAX or
        x3 < X3_MIN or x3 > X3_MAX
    ):
        particle["x"] = random_position()
        particle["v"] = random_velocity()


def update_personal_best_and_refresh(i, particles, pci_list):
    particle = particles[i]
    current_value = evaluate_solution(
        particle["x"][0],
        particle["x"][1],
        particle["x"][2]
    )

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


def format_value_for_csv(value):
    if math.isnan(value):
        value = 0.0
    elif value == float("inf"):
        value = MAX_FLOAT
    elif value == float("-inf"):
        value = -MAX_FLOAT

    return str(value).replace(".", ",")


def write_iteration_result(writer, iteration, best_x1, best_x2, best_x3, best_value):
    writer.writerow([
        iteration,
        format_value_for_csv(best_x1),
        format_value_for_csv(best_x2),
        format_value_for_csv(best_x3),
        format_value_for_csv(best_value)
    ])


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
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["iteracija", "x1", "x2", "x3", "vrijednost"])

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
                iteration_best_position[2],
                iteration_best_value
            )

            w = w - W_STEP

    global_best = get_global_best(particles)

    print("Najbolje pronađeno rješenje:")
    print(f"x1 = {global_best['xpb'][0]:.6f}")
    print(f"x2 = {global_best['xpb'][1]:.6f}")
    print(f"x3 = {global_best['xpb'][2]:.6f}")
    print(f"minimalna vrijednost funkcije = {global_best['best_value']}")


if __name__ == "__main__":
    pso()