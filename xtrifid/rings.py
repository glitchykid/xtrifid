"""Faces between circular sections, including sections reduced to a single pole."""


def bridge(lower, upper):
    n = max(len(lower), len(upper))
    for i in range(n):
        j = (i + 1) % n
        if len(lower) == 1:
            yield (lower[0], upper[j], upper[i])
        elif len(upper) == 1:
            yield (lower[i], lower[j], upper[0])
        else:
            yield (lower[i], lower[j], upper[j], upper[i])
