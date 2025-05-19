class Monomer:
    def __init__(
        self,
        name: str,
        mass: float,
    ):
        """
        Initialize a monomer with a name and mass.

        Args:
            name (str): Name of the monomer.
            mass (float): Mass of the monomer.
        """
        if mass <= 0:
            raise ValueError("Mass must be a positive number.")
        self._name = name
        self._mass = mass

    @property
    def name(self) -> str:
        """Get the name of the monomer."""
        return self._name

    @property
    def mass(self) -> float:
        """Get the mass of the monomer."""
        return self._mass
