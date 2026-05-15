from .base_strategy import BaseSearchStrategy
from .random_strategy import RandomStrategy
from .greedy_strategy import GreedyStrategy
from .beam_strategy import BeamStrategy

STRATEGY_REGISTRY = {
    "random": RandomStrategy,
    "greedy": GreedyStrategy,
    "beam": BeamStrategy,
}
