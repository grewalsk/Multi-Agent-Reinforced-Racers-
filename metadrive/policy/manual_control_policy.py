# This file has been modified to ensure both agents in a multi-agent environment
# consistently use their reinforcement learning (RL) models, regardless of which car
# is currently being focused on by the viewer. The key changes are:
#
# 1. Modified ManualControlPolicy.act() to always use RL models for non-tracked vehicles
# 2. Updated TakeoverPolicy.act() to ensure non-tracked vehicles use their RL models
# 3. Added better documentation and tracking of control modes
#
# These changes ensure that when switching between agents with the camera,
# the non-focused agent continues to use its RL model rather than reverting
# to a simple straight-line behavior.

from metadrive.engine.core.manual_controller import KeyboardController, SteeringWheelController, XboxController
from metadrive.engine.engine_utils import get_global_config
from metadrive.engine.logger import get_logger
from metadrive.examples import expert
from metadrive.policy.env_input_policy import EnvInputPolicy

logger = get_logger()

JOYSTICK_DEADZONE = 0.025


def get_controller(controller_name, pygame_control):
    """Get the controller object.

    Args:
        controller_name: The controller name.
        pygame_control: Whether to use Pygame as the interface to receive keyboard signal if using keyboard.

    Returns:
        The instance of controller or None if error.
    """
    controller_name = str(controller_name).lower()
    if controller_name == "keyboard":
        return KeyboardController(pygame_control=pygame_control)
    elif controller_name in ["xboxController", "xboxcontroller", "xbox", "gamepad", "joystick", "steering_wheel",
                             "wheel"]:
        try:
            if controller_name in ["steering_wheel", "wheel"]:
                return SteeringWheelController()
            else:
                return XboxController()
        except Exception:
            return None
    else:
        raise ValueError("No such a controller type: {}".format(controller_name))


class ManualControlPolicy(EnvInputPolicy):
    """
    Control the current track vehicle
    """

    DEBUG_MARK_COLOR = (252, 244, 3, 255)

    def __init__(self, obj, seed, enable_expert=True):
        super(ManualControlPolicy, self).__init__(obj, seed)
        config = self.engine.global_config
        self.enable_expert = enable_expert

        if config["manual_control"] and config["use_render"]:
            self.engine.accept("t", self.toggle_takeover)
            pygame_control = False
        elif config["manual_control"]:
            # Use pygame to accept key strike.
            pygame_control = True
        else:
            pygame_control = False

        # if config["manual_control"] and config["use_render"]:
        if config["manual_control"]:
            self.controller = get_controller(config["controller"], pygame_control=pygame_control)
            if self.controller is None:
                logger.warning("Load Joystick or Steering Wheel Error! Fall back to keyboard control")
                self.controller = KeyboardController(pygame_control=pygame_control)
        else:
            self.controller = None

    def act(self, agent_id):
        # Process controller inputs regardless of which agent is being controlled
        self.controller.process_others(takeover_callback=self.toggle_takeover)

        # Get the current agent
        current_agent = self.engine.agent_manager.get_agent(agent_id)
        is_track_vehicle = current_agent is self.engine.current_track_agent
        not_in_native_bev = (self.engine.main_camera is None) or (not self.engine.main_camera.is_bird_view_camera())

        # Check if this is the tracked vehicle and if expert takeover is enabled
        if is_track_vehicle and self.engine.current_track_agent.expert_takeover and self.enable_expert:
            try:
                # Use expert policy for the tracked vehicle if takeover is enabled
                action = expert(self.engine.current_track_agent)
                self.action_info["manual_control"] = False
                self.action_info["expert_control"] = True
            except (ValueError, AssertionError):
                # If observation doesn't match, fall back to manual control
                print("Current observation does not match the format that expert can accept.")
                self.toggle_takeover()
                # Continue with manual or RL control
                if self.engine.global_config["manual_control"] and not_in_native_bev:
                    action = self.controller.process_input(self.engine.current_track_agent)
                    self.action_info["manual_control"] = True
                    self.action_info["expert_control"] = False
                else:
                    # Use RL model for tracked vehicle if not in manual control
                    action = super(ManualControlPolicy, self).act(agent_id)
                    self.action_info["manual_control"] = False
                    self.action_info["expert_control"] = False
        elif is_track_vehicle and self.engine.global_config["manual_control"] and not_in_native_bev:
            # Manual control for tracked vehicle when not in bird view and manual control is enabled
            action = self.controller.process_input(self.engine.current_track_agent)
            self.action_info["manual_control"] = True
            self.action_info["expert_control"] = False
        else:
            # Always use RL model for non-tracked vehicles or when manual control is disabled
            action = super(ManualControlPolicy, self).act(agent_id)
            self.action_info["manual_control"] = False
            self.action_info["expert_control"] = False

        self.action_info["action"] = action
        return action

    def toggle_takeover(self):
        if self.engine.current_track_agent is not None:
            self.engine.current_track_agent.expert_takeover = not self.engine.current_track_agent.expert_takeover
            print("The expert takeover is set to: ", self.engine.current_track_agent.expert_takeover)


class TakeoverPolicy(EnvInputPolicy):
    """
    Takeover policy shares the control between RL agent (whose action is input via env.step) and
    external control device (whose action is input via controller).
    """
    def __init__(self, obj, seed):
        super(TakeoverPolicy, self).__init__(obj, seed)
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            self.controller = get_controller(config["controller"], pygame_control=False)
        self.takeover = False

    def act(self, agent_id):
        # Always get the RL model action first
        agent_action = super(TakeoverPolicy, self).act(agent_id)

        # Check if this is the tracked vehicle
        is_track_vehicle = self.engine.agent_manager.get_agent(agent_id) is self.engine.current_track_agent

        # Only allow manual takeover for the tracked vehicle
        if is_track_vehicle and self.engine.global_config["manual_control"] and not self.engine.main_camera.is_bird_view_camera():
            # Process controller input for the tracked vehicle
            expert_action = self.controller.process_input(self.engine.current_track_agent)

            # Check for takeover conditions based on controller type
            if isinstance(self.controller, SteeringWheelController) and (self.controller.left_shift_paddle
                                                                         or self.controller.right_shift_paddle):
                self.takeover = True
                return expert_action
            elif isinstance(self.controller, KeyboardController) and (self.controller.takeover
                                                                      or abs(sum(expert_action)) > 0.01):
                self.takeover = True
                return expert_action
            elif isinstance(self.controller, XboxController) and (self.controller.takeover or self.controller.button_a
                                                                  or self.controller.button_b or
                                                                  self.controller.button_x or self.controller.button_y
                                                                  or abs(sum(expert_action)) > JOYSTICK_DEADZONE):
                self.takeover = True
                return expert_action

        # For non-tracked vehicles or when no takeover is detected, use the RL model
        self.takeover = False
        return agent_action


class TakeoverPolicyWithoutBrake(TakeoverPolicy):
    """
    Takeover policy shares the control between RL agent (whose action is input via env.step) and
    external control device (whose action is input via controller).
    Note that this policy will discard brake in human's action.
    """
    def act(self, agent_id):
        action = super(TakeoverPolicyWithoutBrake, self).act(agent_id=agent_id)
        if self.takeover and action[1] < 0.0:
            action[1] = 0.0
        return action
