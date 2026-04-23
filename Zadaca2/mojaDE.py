import sys
import csv
import random
import math


DIMENSIONS = 4  # a, b, c, d
POP_SIZE = 50
MAX_ITER = 1000
F = 0.7
CR = 0.9

PARAM_MIN = -20.0
PARAM_MAX = 20.0

EPSILON = 1e-8
PENALTY = 1e12

OUTPUT_CSV = "de_rezultati.csv"


def parse_points(args):
    if len(args) != 10:
        print("Greška: potrebno je unijeti točno 10 brojeva za 5 točaka.")
        print("Primjer: python mojaDE.py 2 3.4 0.2 0.6 8 17 -3 -0.5 12 25")
        sys.exit(1)

    try:
        values = [float(arg) for arg in args]
    except ValueError:
        print("Greška: svi argumenti moraju biti brojevi.")
        sys.exit(1)

    points = []
    for i in range(0, 10, 2):
        x = values[i]
        y = values[i + 1]
        points.append((x, y))

    return points


def model_function(x, a, b, c, d):
    denominator = b * x * x + c * x + d
    if abs(denominator) < EPSILON:
        return None
    return (a * x) / denominator


def objective_function(candidate, points):
    a, b, c, d = candidate
    error = 0.0

    for x, y_hat in points:
        fx = model_function(x, a, b, c, d)
        if fx is None or math.isnan(fx) or math.isinf(fx):
            return PENALTY

        diff = fx - y_hat
        error += diff * diff

    return error


def random_candidate():
    return [random.uniform(PARAM_MIN, PARAM_MAX) for _ in range(DIMENSIONS)]


def clip_value(value):
    if value < PARAM_MIN:
        return PARAM_MIN
    if value > PARAM_MAX:
        return PARAM_MAX
    return value


def initialize_population(points):
    population = []
    fitness = []

    for _ in range(POP_SIZE):
        candidate = random_candidate()
        population.append(candidate)
        fitness.append(objective_function(candidate, points))

    return population, fitness


def find_best_index(fitness):
    best_index = 0
    best_value = fitness[0]

    for i in range(1, len(fitness)):
        if fitness[i] < best_value:
            best_value = fitness[i]
            best_index = i

    return best_index


def mutate_best_1(population, best_index, current_index):
    available_indices = [i for i in range(len(population)) if i != current_index and i != best_index]
    r1, r2 = random.sample(available_indices, 2)

    best = population[best_index]
    x_r1 = population[r1]
    x_r2 = population[r2]

    mutant = []
    for j in range(DIMENSIONS):
        value = best[j] + F * (x_r1[j] - x_r2[j])
        mutant.append(clip_value(value))

    return mutant


def crossover_bin(target, mutant):
    trial = []
    j_rand = random.randint(0, DIMENSIONS - 1)

    for j in range(DIMENSIONS):
        if random.random() < CR or j == j_rand:
            trial.append(mutant[j])
        else:
            trial.append(target[j])

    return trial


def save_history_to_csv(history):
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["iteracija", "a", "b", "c", "d", "error"])

        for row in history:
            writer.writerow(row)


def differential_evolution(points):
    population, fitness = initialize_population(points)
    history = []

    best_index = find_best_index(fitness)
    best_candidate = population[best_index][:]
    best_error = fitness[best_index]

    history.append([0, best_candidate[0], best_candidate[1], best_candidate[2], best_candidate[3], best_error])

    for iteration in range(1, MAX_ITER + 1):
        for i in range(POP_SIZE):
            best_index = find_best_index(fitness)

            target = population[i]
            mutant = mutate_best_1(population, best_index, i)
            trial = crossover_bin(target, mutant)

            trial_error = objective_function(trial, points)

            if trial_error <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_error

        best_index = find_best_index(fitness)
        current_best_candidate = population[best_index][:]
        current_best_error = fitness[best_index]

        if current_best_error < best_error:
            best_candidate = current_best_candidate[:]
            best_error = current_best_error

        history.append([
            iteration,
            best_candidate[0],
            best_candidate[1],
            best_candidate[2],
            best_candidate[3],
            best_error
        ])

    return best_candidate, best_error, history


def main():

    random.seed()

    points = parse_points(sys.argv[1:])

    best_candidate, best_error, history = differential_evolution(points)

    save_history_to_csv(history)

    print("Najbolje pronađeno rješenje:")
    print(f"a = {best_candidate[0]:.6f}")
    print(f"b = {best_candidate[1]:.6f}")
    print(f"c = {best_candidate[2]:.6f}")
    print(f"d = {best_candidate[3]:.6f}")
    print(f"Greška = {best_error:.6f}")
    print(f"Rezultati su spremljeni u datoteku: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()