from stable_baselines3.common.env_checker import check_env
from gazebo_car_env import 

# Inicjalizacja środowiska
env = TwojeSrodowisko()

# Sprawdzenie zgodności (wymiary observation_space, action_space itp.)
# Jeśli to przejdzie bez błędów, środowisko jest gotowe do SB3.
check_env(env, warn=True)
