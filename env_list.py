import gymnasium as gym
from rich import pretty, print
from rich.console import Console

print("Hello, [bold red]World[/bold red]!", ":vampire:")
pretty.install()
print(locals())
# print(gym.__version__)
# # toytext = ['Blackjack-v1', 'Taxi-v3', "CliffWalking-v0", "FrozenLake-v1"]
# # for toy in toytext:
# #     env = gym.make(toy)
# #     print(env.reset())
console = Console()
console.print(*gym.envs.registry.keys(), style="bold red")

# print("Hello, [bold magenta]World[/bold magenta]!", ":vampire:", locals())
