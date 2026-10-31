import math


def decode_chromosome(chromosome, n_sat):
    if n_sat <= 0:
        raise ValueError("Number of satellites must be positive.")

    if len(chromosome) != 6 * n_sat:
        raise ValueError("Invalid chromosome length.")

    if not all(math.isfinite(value) for value in chromosome):
        raise ValueError("Chromosome contains invalid numeric values.")

    satellites = []

    for i in range(n_sat):
        satellite = (
            chromosome[i],
            chromosome[n_sat + i],
            chromosome[2 * n_sat + i],
            chromosome[3 * n_sat + i],
            chromosome[4 * n_sat + i],
            chromosome[5 * n_sat + i]
        )

        satellites.append(satellite)

    return satellites