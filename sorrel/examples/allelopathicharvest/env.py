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

# --------------------------------- #
# endregion: Imports                #
# --------------------------------- #

# Entity list for observations (must match Berry._berry_kind() and other entity kinds)
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
]


class AllelopathicHarvestEnv(Environment[AllelopathicHarvestWorld]):
    """Allelopathic Harvest environment."""

    def setup_agents(self) -> None:
        """Create agents with their preferences."""
        agents = []

        # Generate agent preferences
        preferences = (
            ["red"] * self.config.world.red_preference_count
            + ["green"] * self.config.world.green_preference_count
            + ["blue"] * self.config.world.blue_preference_count
        )

        # Create action spec
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
            ]
        )

        # Create each agent
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
        # Note: create_world() was already called by reset(),
        # so map is filled with the world's default entity.

        # Ensure agent and beam layers start empty (no leftover visuals if re-used).
        for y in range(self.world.height):
            for x in range(self.world.width):
                agent_loc = (y, x, self.world.agent_layer)
                beam_loc = (y, x, self.world.beam_layer)
                self.world.add(agent_loc, EmptyEntity())
                self.world.add(beam_loc, EmptyEntity())

        # Create walls (object layer); berries are placed separately below.
        for index in np.ndindex(self.world.map.shape):
            y, x, layer = index

            # Walls on edges
            if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                if layer == self.world.object_layer:
                    self.world.add(index, Wall())

            # Berries in playable area (object layer only) –
            elif layer == self.world.object_layer:
                # We'll place berries randomly below
                pass

        # Place berries randomly
        self.place_berries()

        # Spawn agents
        self.spawn_agents()

    def place_berries(self):
        """Place berries evenly distributed."""
        # Reset berry counts before (re)populating.
        self.world.berry_counts = {"red": 0, "green": 0, "blue": 0}

        valid_positions = []
        for y in range(1, self.world.height - 1):
            for x in range(1, self.world.width - 1):
                location = (y, x, self.world.object_layer)
                if self.world.map[location].passable:
                    valid_positions.append(location)

        np.random.shuffle(valid_positions)
        # Don't request more berries than there are valid positions.
        max_berries = min(self.world.total_berries, len(valid_positions))
        berry_positions = valid_positions[:max_berries]

        berry_colors = ["red", "green", "blue"]
        for i, pos in enumerate(berry_positions):
            color = berry_colors[i % 3]
            self.world.add(pos, Berry(color, ripe=False))
            # Keep world.berry_counts aligned with actual map contents.
            self.world.berry_counts[color] += 1

    def spawn_agents(self):
        """Spawn agents in random positions."""

        # FIRST: Clear all agents from the agent layer
        from sorrel.examples.allelopathicharvest.entities import EmptyEntity

        for y in range(self.world.height):
            for x in range(self.world.width):
                agent_loc = (y, x, self.world.agent_layer)
                # Replace with empty entity
                self.world.add(agent_loc, EmptyEntity())

        # NOW place agents fresh
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
