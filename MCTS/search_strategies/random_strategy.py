import random
from typing import Set, Tuple

from node import SearchNode, SearchRootNode, GraphNode, KGENode, LLMNode
from setup_logger import setup_logger, rank_logger

from .base_strategy import BaseSearchStrategy


FILTER_ACTIONS = [GraphNode, KGENode, LLMNode]


class RandomStrategy(BaseSearchStrategy):
    """
    随机搜索策略：在每一步随机选择过滤器进行状态转移，
    直至满足叶节点阈值后进行评估。重复此过程直到预算耗尽。
    """

    def __init__(self, rank: int = 0, **kwargs):
        super().__init__(rank=rank, **kwargs)
        self.logger = setup_logger(self.__class__.__name__)

    def search(
        self,
        root_node: SearchRootNode,
        budget: int,
    ) -> Tuple[Set[Tuple[str, str, str]], int]:
        discovered = set()
        budget_used = 0

        while budget_used < budget:
            leaf = self._random_rollout(root_node)
            if leaf is None:
                break

            correct, used = leaf.evaluate_candidates()
            budget_used += used
            discovered.update(correct)

            rank_logger(self.logger, self.rank)(
                f"Random rollout: found {len(correct)} triplets, "
                f"budget {budget_used}/{budget}"
            )

            if not root_node.candidate_entities:
                break

        return discovered, budget_used

    def _random_rollout(self, root_node: SearchRootNode) -> SearchNode | None:
        """
        从根节点出发，每步随机选择一个过滤器，逐步过滤候选集，
        直到到达叶节点或无法继续过滤。
        """
        current = root_node

        while not current.is_terminal():
            chosen = random.choice(FILTER_ACTIONS)
            child_context = current._make_child_context()
            child = chosen(child_context)

            if not child.candidate_entities:
                break

            current = child

        return current if current.candidate_entities else None
