# --------------------------------- #
# region: Imports                   #
# --------------------------------- #

from pathlib import Path

import numpy as np

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.entities import Entity
from sorrel.examples.allelopathicharvest.entities import Berry, EmptyEntity
from sorrel.examples.allelopathicharvest.world import AllelopathicHarvestWorld
from sorrel.location import Location
from sorrel.models import BaseModel
from sorrel.observation import embedding, observation_spec
from sorrel.worlds import Gridworld

# --------------------------------- #
# endregion: Imports                #
# --------------------------------- #


# --------------------------------- #
# region: Observation               #
# --------------------------------- #


class HarvestObservation(observation_spec.OneHotObservationSpec):
    """Custom observation function for the Allelopathic Harvest agent class."""

    def __init__(
        self,
        entity_list: list[str],
        full_view: bool = False,
        vision_radius: int | None = None,
        embedding_size: int = 3,
    ):
        super().__init__(entity_list, full_view, vision_radius)
        self.embedding_size = embedding_size

        if self.full_view:
            self.input_size = (
                1,
                (len(entity_list) * 31 * 32)  # Default 31x32; override if different
                + (4 * self.embedding_size),
            )
        else:
            # partial observation: (2*vision_radius+1)^2 window
            self.input_size = (
                1,
                (
                    len(entity_list)
                    * (2 * self.vision_radius + 1)
                    * (2 * self.vision_radius + 1)
                )
                + (4 * self.embedding_size),
            )

    def observe(self, world: Gridworld, location: tuple | Location | None = None):
        """Location must be provided for this observation."""
        if location is None:
            raise ValueError("Location must not be None for HarvestObservation.")

        visual_field = super().observe(world, location).flatten()
        pos_code = embedding.positional_embedding(
            location, world, (self.embedding_size, self.embedding_size)
        )

        return np.concatenate((visual_field, pos_code))


# --------------------------------- #
# endregion: Observation            #
# --------------------------------- #


# --------------------------------- #
# region: Agent                     #
# --------------------------------- #


class AllelopathicHarvestAgent(Agent[AllelopathicHarvestWorld]):
    """An Allelopathic Harvest agent."""

    def __init__(
        self,
        observation_spec: HarvestObservation,
        action_spec: ActionSpec,
        model: BaseModel,
        preference_color: str,
        agent_id: int,
    ):
        super().__init__(observation_spec, action_spec=action_spec, model=model)

        self.preference_color = preference_color
        self.agent_id = agent_id
        self.current_color = "white"

        # orientation: 0=North, 1=East, 2=South, 3=West
        self.orientation = 2  # Start facing down
        self.sprite = Path(__file__).parent / "./assets/hero.png"

        # stats tracking
        self.berries_consumed = {"red": 0, "green": 0, "blue": 0}
        self.berries_planted = {"red": 0, "green": 0, "blue": 0}
        self.zap_cooldown = 0
        self.frozen_steps = 0
        self.marked_steps = 0

    def pov(self, world: AllelopathicHarvestWorld) -> np.ndarray:
        """Get agent's point of view (observation)."""
        image = self.observation_spec.observe(world, self.location)
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Get action from model given current state."""
        prev_states = self.model.memory.current_state()
        stacked_states = np.vstack((prev_states, state))

        model_input = stacked_states.reshape(1, -1)
        # get model output
        model_output = self.model.take_action(model_input)

        return model_output

    def set_color(self, color: str):
        """Update agent's visual appearance based on orientation.

        Note: For now we ignore the color parameter and just use directional sprites.
        """
        self.current_color = color
        # update sprite based on orientation
        _base = Path(__file__).parent
        if self.orientation == 0:  # North
            self.sprite = _base / "./assets/hero-back.png"
        elif self.orientation == 1:  # East
            self.sprite = _base / "./assets/hero-right.png"
        elif self.orientation == 2:  # South
            self.sprite = _base / "./assets/hero.png"
        else:  # West
            self.sprite = _base / "./assets/hero-left.png"

    def get_forward_location(self) -> tuple:
        """Get the location in front of the agent based on orientation."""
        y, x, layer = self.location

        if self.orientation == 0:  # North
            return (y - 1, x, layer)
        elif self.orientation == 1:  # East
            return (y, x + 1, layer)
        elif self.orientation == 2:  # South
            return (y + 1, x, layer)
        else:  # West
            return (y, x - 1, layer)

    def spawn_planting_beam(
        self, world: AllelopathicHarvestWorld, plant_color: str
    ) -> None:
        """Generate a planting beam in front of the agent."""
        forward_loc = self.get_forward_location()
        forward_obj_loc = (forward_loc[0], forward_loc[1], world.object_layer)

        if not world.valid_location(forward_obj_loc):
            return

        # get entity at that location
        target_entity = world.observe(forward_obj_loc)

        # try to plant if it's an unripe berry
        if isinstance(target_entity, Berry) and not target_entity.ripe:
            berry = target_entity
            old_color = berry.color

            # change berry color
            if berry.change_color(plant_color):
                # Update world berry counts
                world.berry_counts[old_color] -= 1
                world.berry_counts[plant_color] += 1

                self.berries_planted[plant_color] += 1

                # change agent color to match planted berry
                self.set_color(plant_color)

        # spawn beam visual on beam layer
        beam_loc = (forward_loc[0], forward_loc[1], world.beam_layer)
        if world.valid_location(beam_loc):
            world.add(beam_loc, PlantingBeam(plant_color))

    def apply_zap(self) -> float:
        """Apply zap effects to this agent and return immediate reward delta."""
        reward_delta = 0.0
        if self.marked_steps > 0:
            reward_delta = -10.0
        self.frozen_steps = 25
        self.marked_steps = 50
        return reward_delta

    def spawn_zap_beam(self, world: AllelopathicHarvestWorld) -> float:
        """Spawn a zap beam and apply zap effect to the first target hit."""
        if self.zap_cooldown > 0:
            return 0.0

        reward = 0.0
        self.zap_cooldown = 4

        # Simple forward ray on the agent layer, with visual beams on beam layer.
        y, x, _ = self.location
        for step in range(1, world.config.agent.agent.beam_radius + 1):
            if self.orientation == 0:  # North
                target_y, target_x = y - step, x
            elif self.orientation == 1:  # East
                target_y, target_x = y, x + step
            elif self.orientation == 2:  # South
                target_y, target_x = y + step, x
            else:  # West
                target_y, target_x = y, x - step

            agent_loc = (target_y, target_x, world.agent_layer)
            beam_loc = (target_y, target_x, world.beam_layer)
            obj_loc = (target_y, target_x, world.object_layer)

            if not world.valid_location(agent_loc):
                break

            if world.valid_location(beam_loc):
                world.add(beam_loc, ZapBeam())
            if not world.observe(obj_loc).passable:
                break

            target = world.observe(agent_loc)
            if isinstance(target, AllelopathicHarvestAgent) and target is not self:
                reward += target.apply_zap()
                break

        return reward

    def consume_berry(
        self, world: AllelopathicHarvestWorld, berry_location: tuple
    ) -> float:
        """Consume a ripe berry and get reward."""
        entity = world.observe(berry_location)

        # can only consume ripe berries
        if not isinstance(entity, Berry) or not entity.ripe:
            return 0.0

        berry = entity
        berry_color = berry.color

        # remove berry from map
        world.remove(berry_location)
        world.berry_counts[berry_color] -= 1

        self.berries_consumed[berry_color] += 1
        reward = (
            world.preferred_berry_reward
            if berry_color == self.preference_color
            else world.other_berry_reward
        )

        # stochastically recolor agent to white
        fractions = world.curr_fractions()
        max_fraction = max(fractions.values()) if fractions else 0.0
        white_prob = 1.0 - max_fraction

        if np.random.random() < white_prob:
            self.set_color("white")

        return reward

    def act(self, world: AllelopathicHarvestWorld, action: int) -> float:
        """Execute an action in the world."""
        # update temporary status timers every step.
        if self.zap_cooldown > 0:
            self.zap_cooldown -= 1
        if self.frozen_steps > 0:
            self.frozen_steps -= 1
        if self.marked_steps > 0:
            self.marked_steps -= 1

        # if frozen, skip all action effects this turn
        if self.frozen_steps > 0:
            return 0.0
        action_name = self.action_spec.get_readable_action(action)

        reward = 0.0
        new_location = self.location
        if action_name == "forward":
            new_location = self.get_forward_location()

        elif action_name == "backward":
            # Move in opposite direction
            y, x, layer = self.location
            if self.orientation == 0:  # North -> move South
                new_location = (y + 1, x, layer)
            elif self.orientation == 1:  # East -> move West
                new_location = (y, x - 1, layer)
            elif self.orientation == 2:  # South -> move North
                new_location = (y - 1, x, layer)
            else:  # West -> move East
                new_location = (y, x + 1, layer)

        elif action_name == "strafe_left":
            # Move perpendicular to facing direction (left)
            y, x, layer = self.location
            if self.orientation == 0:  # North -> move West
                new_location = (y, x - 1, layer)
            elif self.orientation == 1:  # East -> move North
                new_location = (y - 1, x, layer)
            elif self.orientation == 2:  # South -> move East
                new_location = (y, x + 1, layer)
            else:  # West -> move South
                new_location = (y + 1, x, layer)

        elif action_name == "strafe_right":
            # Move perpendicular to facing direction (right)
            y, x, layer = self.location
            if self.orientation == 0:  # North -> move East
                new_location = (y, x + 1, layer)
            elif self.orientation == 1:  # East -> move South
                new_location = (y + 1, x, layer)
            elif self.orientation == 2:  # South -> move West
                new_location = (y, x - 1, layer)
            else:  # West -> move North
                new_location = (y - 1, x, layer)

        elif action_name == "turn_left":
            self.orientation = (self.orientation - 1) % 4
            self.set_color(self.current_color)

        elif action_name == "turn_right":
            self.orientation = (self.orientation + 1) % 4
            self.set_color(self.current_color)

        elif action_name == "plant_red":
            self.spawn_planting_beam(world, "red")

        elif action_name == "plant_green":
            self.spawn_planting_beam(world, "green")

        elif action_name == "plant_blue":
            self.spawn_planting_beam(world, "blue")

        elif action_name == "zap":
            reward += self.spawn_zap_beam(world)

        if new_location != self.location and world.valid_location(new_location):
            obj_loc = (new_location[0], new_location[1], world.object_layer)
            target_entity = world.observe(obj_loc)

            if not target_entity.passable:
                new_location = self.location
            else:
                if isinstance(target_entity, Berry) and target_entity.ripe:
                    reward += self.consume_berry(world, obj_loc)

        # try moving to new location
        if world.valid_location(new_location):
            world.move(self, new_location)

        # update total reward
        world.total_reward += reward

        return reward

    def is_done(self, world: AllelopathicHarvestWorld) -> bool:
        """Check if episode is done."""
        return world.turn >= world.max_turns

    def reset(self) -> None:
        """Reset the agent (and its memory).

        Required by base Agent.
        """
        self.model.reset()
        self.reset_agent_state()

    def reset_agent_state(self):
        """Reset agent's visual state and statistics.

        This is called from env.py during populate_environment().
        """
        self.orientation = 2  # Face down
        self.current_color = "white"
        self.set_color("white")
        self.berries_consumed = {"red": 0, "green": 0, "blue": 0}
        self.berries_planted = {"red": 0, "green": 0, "blue": 0}
        self.zap_cooldown = 0
        self.frozen_steps = 0
        self.marked_steps = 0


# --------------------------------- #
# endregion: Agent                  #
# --------------------------------- #


# --------------------------------- #
# region: Beams                     #
# --------------------------------- #


class PlantingBeam(Entity):
    """Visual beam entity for planting actions."""

    def __init__(self, color: str):
        super().__init__()
        self.color = color
        self.sprite = Path(__file__).parent / "./assets/beam.png"
        self.turn_counter = 0
        self.has_transitions = True
        self.passable = True

    def transition(self, world: Gridworld):
        """Beams persist for one turn, then disappear."""
        if self.turn_counter >= 1:
            world.add(self.location, EmptyEntity())
        else:
            self.turn_counter += 1


class ZapBeam(Entity):
    """Visual beam entity for zap actions."""

    def __init__(self):
        super().__init__()
        self.sprite = Path(__file__).parent / "./assets/beam.png"
        self.turn_counter = 0
        self.has_transitions = True
        self.passable = True

    def transition(self, world: Gridworld):
        """Beams persist for one turn, then disappear."""
        if self.turn_counter >= 1:
            world.add(self.location, EmptyEntity())
        else:
            self.turn_counter += 1


# --------------------------------- #
# endregion: Beams                  #
# --------------------------------- #
