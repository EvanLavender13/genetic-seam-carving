def construct_seam(individual):
    # get size of individual
    size = len(individual)

    # get start
    val = individual.start

    # create seam
    seam = []
    for i in range(size):
        val = val + individual[i]
        point = (i, val)
        seam.append(point)

    return seam
