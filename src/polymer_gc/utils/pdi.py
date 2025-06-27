import numpy as np
from typing import Union, Optional


def generate_pdi_distribution(
    n: int,
    min_pdi: float = 1.0,
    max_pdi: float = 3.0,
    r: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a distribution of Polydispersity Index (PDI) values.

    This function generates a distribution of PDI values that is:
    - Skewed towards lower values (near min_pdi)
    - Has a long tail towards higher values (near max_pdi)
    - Follows a 1/x relationship after the inversion step

    The distribution is created through a series of transformations:
    1. Generate uniform random numbers
    2. Shift and invert to create a non-uniform distribution
    3. Normalize to [0,1]
    4. Scale to the desired range [min_pdi, max_pdi]

    This distribution better reflects real-world polymer systems where:
    - Most samples have relatively low PDI values
    - Fewer samples have very high PDI values
    - The distribution is naturally skewed towards lower values

    Args:
        n (int): Number of PDI values to generate
        min_pdi (float, optional): Minimum PDI value. Defaults to 1.0.
        max_pdi (float, optional): Maximum PDI value. Defaults to 3.0.
        r (float, optional): Distribution control parameter. Defaults to 2.0.
        rng (Optional[np.random.Generator], optional): Random number generator.
            If None, a new one is created. Defaults to None.

    Returns:
        np.ndarray: Array of n PDI values distributed between min_pdi and max_pdi
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate initial random numbers
    pdi = rng.rand(n) * r  # generate random numbers between 0 and r [0, 2]
    pdi = pdi + 1  # shift range up by 1 [1, 3]
    pdi = 1 / pdi  # invert the numbers [1/3, 1]
    pdi -= 1 / (r + 1)  # subtract 1/(r+1) to shift range [0, 2/3]
    pdi = pdi / (1 - 1 / (r + 1))  # normalize to [0, 1]
    pdi *= max_pdi - min_pdi  # scale to desired range width
    pdi += min_pdi  # shift to final range [min_pdi, max_pdi]

    return pdi
