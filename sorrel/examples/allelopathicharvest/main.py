from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

from sorrel.examples.allelopathicharvest.entities import EmptyEntity
from sorrel.examples.allelopathicharvest.env import AllelopathicHarvestEnv
from sorrel.examples.allelopathicharvest.world import AllelopathicHarvestWorld


def main():
    """Main function to run Allelopathic Harvest experiment."""
    config_path = Path(__file__).parent / "configs" / "config.yaml"
    config = cast(DictConfig, OmegaConf.load(config_path))

    default_entity = EmptyEntity()
    world = AllelopathicHarvestWorld(config, default_entity)

    env = AllelopathicHarvestEnv(world, config)

    output_dir = Path(__file__).parent / "data"
    env.run_experiment(
        animate=True,
        logging=True,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
