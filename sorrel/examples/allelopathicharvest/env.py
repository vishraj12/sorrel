import os
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy import ndenumerate
from omegaconf import DictConfig

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.entities import Entity
from sorrel.environment import Environment
from sorrel.examples.allelopathicharvest.agents import (
    AllelopathicHarvestAgent,
    HarvestObservation,
)
from sorrel.examples.allelopathicharvest.entities import Berry, EmptyEntity, Wall
from sorrel.examples.allelopathicharvest.world import AllelopathicHarvestWorld
from sorrel.models.pytorch import PyTorchIQN
from sorrel.utils.logging import ConsoleLogger, Logger
from sorrel.utils.visualization import ImageRenderer

# --------------------------------- #
# endregion: Imports                #
# --------------------------------- #

ENTITY_LIST = [
    "EmptyEntity",
    "Wall",
    "Berry_red_unripe",
    "Berry_red_ripe",
    "Berry_green_unripe",
    "Berry_green_ripe",
    "Berry_blue_unripe",
    "Berry_blue_ripe",
    "AllelopathicHarvestAgent",
    "PlantingBeam",
    "ZapBeam",
]


class AllelopathicHarvestEnv(Environment[AllelopathicHarvestWorld]):
    """Allelopathic Harvest environment."""

    def setup_agents(self) -> None:
        """Create agents with their preferences."""
        agents = []

        # generate agent preferences
        preferences = (
            ["red"] * self.config.world.red_preference_count
            + ["green"] * self.config.world.green_preference_count
            + ["blue"] * self.config.world.blue_preference_count
        )
        if not preferences:
            preferences = ["red"] * self.config.agent.agent.num
        elif len(preferences) < self.config.agent.agent.num:
            reps = (self.config.agent.agent.num // len(preferences)) + 1
            preferences = (preferences * reps)[: self.config.agent.agent.num]
        elif len(preferences) > self.config.agent.agent.num:
            preferences = preferences[: self.config.agent.agent.num]

        # create action spec
        action_spec = ActionSpec(
            [
                "forward",
                "backward",
                "strafe_left",
                "strafe_right",
                "turn_left",
                "turn_right",
                "plant_red",
                "plant_green",
                "plant_blue",
                "zap",
            ]
        )

        # create each agent
        for agent_id in range(self.config.agent.agent.num):
            observation_spec = HarvestObservation(
                entity_list=ENTITY_LIST,
                vision_radius=self.config.agent.agent.obs.vision,
                embedding_size=self.config.agent.agent.obs.embeddings,
            )

            model = PyTorchIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                seed=torch.random.seed(),
                n_frames=self.config.agent.agent.obs.n_frames,
                **self.config.model.iqn.parameters,
            )

            agent = AllelopathicHarvestAgent(
                observation_spec=observation_spec,
                action_spec=action_spec,
                model=model,
                preference_color=preferences[agent_id],
                agent_id=agent_id,
            )

            agents.append(agent)

        self.agents = agents

    def take_turn(self) -> None:
        """Performs one step; syncs world.turn so agent.is_done() and done signal are
        correct."""
        self.turn += 1
        self.world.turn = self.turn
        for _, x in ndenumerate(self.world.map):
            if x.has_transitions and not isinstance(x, Agent):
                x.transition(self.world)
        for agent in self.agents:
            agent.transition(self.world)

    def populate_environment(self) -> None:
        """Populate the world with walls, berries, and agents."""
        for y in range(self.world.height):
            for x in range(self.world.width):
                agent_loc = (y, x, self.world.agent_layer)
                beam_loc = (y, x, self.world.beam_layer)
                self.world.add(agent_loc, EmptyEntity())
                self.world.add(beam_loc, EmptyEntity())

        for index in np.ndindex(self.world.map.shape):
            y, x, layer = index

            # walls on edges
            if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                if layer == self.world.object_layer:
                    self.world.add(index, Wall())

            elif layer == self.world.object_layer:
                pass

        # place berries randomly
        self.place_berries()

        # spawn agents
        self.spawn_agents()

    def place_berries(self):
        """Place berries evenly distributed."""
        # reset berry counts before (re)populating.
        self.world.berry_counts = {"red": 0, "green": 0, "blue": 0}

        valid_positions = []
        for y in range(1, self.world.height - 1):
            for x in range(1, self.world.width - 1):
                location = (y, x, self.world.object_layer)
                if self.world.map[location].passable:
                    valid_positions.append(location)

        np.random.shuffle(valid_positions)
        max_berries = min(self.world.total_berries, len(valid_positions))
        berry_positions = valid_positions[:max_berries]

        berry_colors = ["red", "green", "blue"]
        for i, pos in enumerate(berry_positions):
            color = berry_colors[i % 3]
            self.world.add(pos, Berry(color, ripe=False))
            self.world.berry_counts[color] += 1

    def spawn_agents(self):
        """Spawn agents in random positions."""

        # clear all agents from the agent layer
        from sorrel.examples.allelopathicharvest.entities import EmptyEntity

        for y in range(self.world.height):
            for x in range(self.world.width):
                agent_loc = (y, x, self.world.agent_layer)
                # replace with empty entity
                self.world.add(agent_loc, EmptyEntity())

        spawn_points = []
        for y in range(1, self.world.height - 1):
            for x in range(1, self.world.width - 1):
                obj_loc = (y, x, self.world.object_layer)
                if self.world.map[obj_loc].passable:
                    spawn_points.append([y, x, self.world.agent_layer])

        loc_index = np.random.choice(
            len(spawn_points), size=len(self.agents), replace=False
        )
        locs = [spawn_points[i] for i in loc_index]

        for loc, agent in zip(locs, self.agents):
            loc = tuple(loc)
            harvest_agent = cast(AllelopathicHarvestAgent, agent)
            harvest_agent.reset_agent_state()
            self.world.add(loc, harvest_agent)

    def run_experiment(
        self,
        animate: bool = True,
        logging: bool = True,
        logger: Logger | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Run experiment for this environment."""
        if output_dir is None:
            if hasattr(self.config.experiment, "output_dir"):
                output_dir = Path(self.config.experiment.output_dir)
            else:
                output_dir = Path(__file__).parent / "./data/"
            assert isinstance(output_dir, Path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if animate:
            renderer = ImageRenderer(
                experiment_name=self.__class__.__name__,
                record_period=self.config.experiment.record_period,
                num_turns=self.config.experiment.max_turns,
            )
        else:
            renderer = None

        for epoch in range(self.config.experiment.epochs + 1):
            self.reset()
            animate_this_turn = animate and (
                epoch % self.config.experiment.record_period == 0
            )
            for agent in self.agents:
                agent.model.start_epoch_action(epoch=epoch)

            while not self.turn >= self.config.experiment.max_turns:
                if animate_this_turn and renderer is not None:
                    renderer.add_image(self.world)
                self.take_turn()
                if self.world.is_done and self.stop_if_done:
                    break

            self.world.is_done = True
            if animate_this_turn and renderer is not None:
                renderer.save_gif(epoch, output_dir / "./gifs/")
            for agent in self.agents:
                agent.model.end_epoch_action(epoch=epoch)

            total_loss = 0
            for agent in self.agents:
                total_loss = agent.model.train_step()

            if logging:
                if not logger:
                    logger = ConsoleLogger(self.config.experiment.epochs)
                logger.record_turn(
                    epoch,
                    total_loss,
                    self.world.total_reward,
                    self.agents[0].model.epsilon,
                )
            for i, agent in enumerate(self.agents):
                if hasattr(self.config.model, "epsilon_decay"):
                    agent.model.epsilon_decay(self.config.model.epsilon_decay)
                if epoch % self.config.experiment.record_period == 0:
                    if hasattr(self.config.model, "save_weights"):
                        if self.config.model.save_weights:
                            agent.model.save(
                                output_dir
                                / f"./checkpoints/{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}-agent-{i}.pkl"
                            )
