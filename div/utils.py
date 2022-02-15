import sys

def s():
    sys.exit()    

def load_smb_env(obs_complexity = 1, action_complexity = 1, is_random = False):
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
    game_id = f"SuperMarioBros{'RandomStages' if is_random else ''}-v{3-obs_complexity}"
    env = gym_super_mario_bros.make(game_id)
    env = JoypadSpace(env, [RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT][action_complexity])
    return env