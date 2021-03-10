# Unique Combination Pairs for list of elements
def uniqueCombinations(numeric_col):
    l = list(itertools.combinations(numeric_col, 2))
    s = set(l)
    # print('actual', len(l), l)
    return list(s)




from __future__ import annotations
import itertools

def unique_combinations(elements: list[str]) -> list[tuple[str, str]]:
    """
    Precondition: `elements` does not contain duplicates.
    Postcondition: Returns unique combinations of length 2 from `elements`.
    """
    return list(itertools.combinations(elements, 2))
