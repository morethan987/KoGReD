from typing import Set, Tuple

from node import SearchNode, SearchRootNode, GraphNode, KGENode, LLMNode
from setup_logger import setup_logger, rank_logger

from .base_strategy import BaseSearchStrategy


FILTER_ACTIONS = [GraphNode, KGENode, LLMNode]


class GreedyStrategy(BaseSearchStrategy):
    """
    贪心搜索策略：在每一步展开所有子节点，选择候选集最小
    （即过滤最激进）的单一分支深入，不保留其他分支。
    到达叶节点后评估，然后回根节点重新开始。
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
            leaf = self._greedy_dive(root_node)
            if leaf is None:
                break

            correct, used = leaf.evaluate_candidates()
            budget_used += used
            discovered.update(correct)

            rank_logger(self.logger, self.rank)(
                f"Greedy dive: found {len(correct)} triplets, "
                f"budget {budget_used}/{budget}"
            )

            if not root_node.candidate_entities:
                break

        return discovered, budget_used

    def _greedy_dive(self, root_node: SearchRootNode) -> SearchNode | None:
        """
        从根节点出发，每步展开所有过滤器子节点，
        选择候选集最小的那个继续深入。
        """
        current = root_node

        while not current.is_terminal():
            best_child = None
            best_size = float('inf')

            for action_cls in FILTER_ACTIONS:
                child_context = current._make_child_context()
                child = action_cls(child_context)

                if child.candidate_entities and len(child.candidate_entities) < best_size:
                    best_size = len(child.candidate_entities)
                    best_child = child

            if best_child is None:
                break

            current = best_child

        return current if current.candidate_entities else None
