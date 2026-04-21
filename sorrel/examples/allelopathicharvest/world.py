# Import packages
from omegaconf import DictConfig, OmegaConf

from sorrel.worlds import Gridworld

# End packages


class AllelopathicHarvestWorld(Gridworld):
    """Allelopathic Harvest world."""

    def __init__(self, config: dict | DictConfig, default_entity):
        if type(config) != DictConfig:  # if it's not a dictionary
            config = OmegaConf.create(config)
        self.config = config  # figure this out
        self.object_layer = 0  # base layer w/ the berries
        self.agent_layer = 1  # agents
        self.beam_layer = 2  # beaming
        layers = 3
        super().__init__(
            config.world.height, config.world.width, layers, default_entity
        )

        self.mode = config.world.mode  # default
        self.max_turns = config.experiment.max_turns  # total number of turns
        self.turn = 0  # turn we're on

        self.channels = config.agent.agent.obs.channels
        self.full_mdp = config.world.full_mdp

        # Berry configuration
        self.total_berries = config.world.total_berries  # e.g. 348
        self.berries_per_color = self.total_berries // 3  # nominal even split
        self.allelopathic_constant = config.world.allelopathic_constant  # 5e-6
        # player preferences
        self.red_preference_count = config.world.red_preference_count  # 8
        self.green_preference_count = config.world.green_preference_count  # 8
        self.blue_preference_count = config.world.blue_preference_count  # 0

        self.preferred_berry_reward = (
            config.world.preferred_berry_reward
        )  # 2 if preferred colour
        self.other_berry_reward = config.world.other_berry_reward  # 1 otherwise

        # Track the ACTUAL number of berries of each colour present.
        # These are initialised by the environment when berries are placed.
        self.berry_counts = {
            "red": 0,
            "green": 0,
            "blue": 0,
        }

    def create_world(self) -> None:
        """Reset map and turn counter (for new episode)."""
        super().create_world()
        self.turn = 0

    def ripening_probability(self, color: str) -> float:
        """Ripening probability for each berry."""
        berry = self.berry_counts.get(color, 0)
        probability = self.allelopathic_constant * berry
        return min(probability, 1.0)  # can't be more than 1

    def curr_fractions(self) -> dict:
        """Current fractions of each berry colour."""
        total = sum(self.berry_counts.values())
        if total == 0:
            return {"red": 0.0, "green": 0.0, "blue": 0.0}
        return {color: count / total for color, count in self.berry_counts.items()}
