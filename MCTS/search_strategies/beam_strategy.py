import heapq
from typing import Set, Tuple

from node import SearchNode, SearchRootNode, GraphNode, KGENode, LLMNode
from setup_logger import setup_logger, rank_logger

from .base_strategy import BaseSearchStrategy


FILTER_ACTIONS = [GraphNode, KGENode, LLMNode]


class _BeamPath:
    """束搜索中的一条候选路径"""

    __slots__ = ("node", "cumulative_score", "depth")

    def __init__(self, node: SearchNode, cumulative_score: float, depth: int):
        self.node = node
        self.cumulative_score = cumulative_score
        self.depth = depth

    def __lt__(self, other: "_BeamPath") -> bool:
        return self.cumulative_score > other.cumulative_score


class BeamStrategy(BaseSearchStrategy):
    """
    束搜索策略：设定束宽 beam_width，每一步保留当前累积得分最高的
    beam_width 条过滤路径，并对它们并行扩展。到达叶节点后评估。
    """

    def __init__(self, rank: int = 0, beam_width: int = 3, **kwargs):
        super().__init__(rank=rank, **kwargs)
        self.beam_width = beam_width
        self.logger = setup_logger(self.__class__.__name__)

    def search(
        self,
        root_node: SearchRootNode,
        budget: int,
    ) -> Tuple[Set[Tuple[str, str, str]], int]:
        discovered = set()
        budget_used = 0

        while budget_used < budget:
            if not root_node.candidate_entities:
                break

            leaves, iter_budget = self._beam_search(root_node, budget - budget_used)

            for leaf in leaves:
                if not leaf.candidate_entities:
                    continue
                correct, used = leaf.evaluate_candidates()
                budget_used += used
                discovered.update(correct)

                rank_logger(self.logger, self.rank)(
                    f"Beam leaf evaluation: found {len(correct)} triplets, "
                    f"budget {budget_used}/{budget}"
                )

            budget_used += iter_budget

            if not leaves:
                break

        return discovered, budget_used

    def _estimate_node_score(self, node: SearchNode) -> float:
        """
        对节点的过滤质量进行启发式打分。
        过滤率越高（候选集越小）得分越高，但同时给予候选集
        大小适中的节点一定奖励以避免过早收敛到空集。
        """
        parent_size = len(node.parent.unfiltered_entities) if node.parent else len(node.unfiltered_entities)
        current_size = len(node.candidate_entities)

        if parent_size == 0:
            return 0.0

        filter_ratio = 1.0 - (current_size / parent_size)

        if current_size == 0:
            return -1.0

        size_bonus = min(current_size / node.leaf_threshold, 1.0) if node.leaf_threshold > 0 else 0.0

        return 0.7 * filter_ratio + 0.3 * size_bonus

    def _beam_search(
        self,
        root_node: SearchRootNode,
        remaining_budget: int,
    ) -> Tuple[list, int]:
        """
        执行一轮束搜索：从根节点开始，逐步扩展并保留 top-beam_width 条路径，
        直到所有路径到达叶节点或无法继续。
        """
        active_beams = [_BeamPath(root_node, 0.0, 0)]
        completed_leaves = []
        iter_budget = 0

        while active_beams:
            all_expansions = []

            for beam in active_beams:
                node = beam.node

                if node.is_terminal() or not node.candidate_entities:
                    completed_leaves.append(node)
                    continue

                for action_cls in FILTER_ACTIONS:
                    child_context = node._make_child_context()
                    child = action_cls(child_context)

                    if not child.candidate_entities:
                        continue

                    score = self._estimate_node_score(child)
                    new_cumulative = beam.cumulative_score + score
                    all_expansions.append(
                        _BeamPath(child, new_cumulative, beam.depth + 1)
                    )

            if not all_expansions:
                break

            active_beams = heapq.nsmallest(self.beam_width, all_expansions)

            all_terminal = all(b.node.is_terminal() for b in active_beams)
            if all_terminal:
                completed_leaves.extend(b.node for b in active_beams)
                active_beams = []

        return completed_leaves, iter_budget
