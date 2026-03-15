from pathlib import Path

import numpy as np

from sorrel.entities import Entity
from sorrel.examples.allelopathicharvest.world import AllelopathicHarvestWorld
from sorrel.worlds import Gridworld

# --------------------------------- #
# endregion: Imports                #
# --------------------------------- #


# --------------------------------- #
# region: Environment Entities      #
# --------------------------------- #


class EmptyEntity(Entity[Gridworld]):
    """Empty Entity class for the Allelopathic Harvest Game."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = Path(__file__).parent / "./assets/empty.png"


class Wall(Entity[Gridworld]):
    """Wall class for the Allelopathic Harvest Game."""

    def __init__(self):
        super().__init__()
        self.passable = False
        self.sprite = Path(__file__).parent / "./assets/wall.png"


# --------------------------------- #
# endregion: Environment Entities   #
# --------------------------------- #


# --------------------------------- #
# region: Berry Entity              #
# --------------------------------- #


class Berry(Entity[AllelopathicHarvestWorld]):
    """Berry entity that can be ripe or unripe.

    Uses kind for observation (color + ripe state).
    """

    def __init__(self, color: str, ripe: bool = False):
        """
        Args:
            color: 'red', 'green', or 'blue'
            ripe: True if ripe (consumable), False if unripe (plantable)
        """
        super().__init__()
        self.color = color
        self.ripe = ripe
        self.consumable = ripe
        self.plantable = not ripe
        self.passable = True  # Agents can ALWAYS walk over berries
        self.has_transitions = True  # Berries ripen over time
        self.kind = self._berry_kind()

        # Sprite
        self.sprite = Path(__file__).parent / f"./assets/{color}_berry.png"

    def _berry_kind(self) -> str:
        """Kind string for one-hot observation (must match ENTITY_LIST in env)."""
        return f"Berry_{self.color}_{'ripe' if self.ripe else 'unripe'}"

    def transition(self, world: AllelopathicHarvestWorld):
        """Called each step - handles berry ripening via allelopathic mechanics."""
        if not self.ripe:
            ripening_prob = world.ripening_probability(self.color)
            if np.random.random() < ripening_prob:
                self.ripen()

    def ripen(self):
        """Change berry from unripe to ripe."""
        self.ripe = True
        self.consumable = True
        self.plantable = False
        self.kind = self._berry_kind()

    def change_color(self, new_color: str) -> bool:
        """Change berry color (only works for unripe berries).

        Args:
            new_color: New berry color ('red', 'green', or 'blue')

        Returns:
            True if successful, False if berry is ripe
        """
        if not self.plantable:
            return False

        self.color = new_color
        self.sprite = Path(__file__).parent / f"./assets/{new_color}_berry.png"
        self.kind = self._berry_kind()
        return True


# --------------------------------- #
# endregion: Berry Entity           #
# --------------------------------- #
