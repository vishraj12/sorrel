import random
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

from sorrel.examples.allelopathicharvest.entities import EmptyEntity
from sorrel.examples.allelopathicharvest.env import AllelopathicHarvestEnv
from sorrel.examples.allelopathicharvest.world import AllelopathicHarvestWorld

# Generate unique ID for this execution
EXECUTION_ID = random.randint(1000, 9999)


def main():
    """Main function to run Allelopathic Harvest experiment."""

    print(f"[EXEC-{EXECUTION_ID}] main() called - START")

    # Load configuration
    config_path = Path(__file__).parent / "configs" / "config.yaml"
    config = cast(DictConfig, OmegaConf.load(config_path))

    print(f"[EXEC-{EXECUTION_ID}] Configuration loaded")
    print(f"[EXEC-{EXECUTION_ID}] Map size: {config.world.height}x{config.world.width}")

    # Create world
    default_entity = EmptyEntity()
    world = AllelopathicHarvestWorld(config, default_entity)

    # Create environment
    print(f"[EXEC-{EXECUTION_ID}] Creating environment and agents...")
    env = AllelopathicHarvestEnv(world, config)

    print(f"[EXEC-{EXECUTION_ID}] Environment setup complete!")
    print(f"[EXEC-{EXECUTION_ID}]   - {len(env.agents)} agents created")

    # Run training
    print(f"[EXEC-{EXECUTION_ID}] Starting training...")
    print("=" * 60)

    output_dir = Path(__file__).parent / "data"
    env.run_experiment(
        animate=True,
        logging=True,
        output_dir=output_dir,
    )

    print(f"[EXEC-{EXECUTION_ID}] Training complete!")


if __name__ == "__main__":
    print(f"[EXEC-{EXECUTION_ID}] __main__ block executing")
    main()
