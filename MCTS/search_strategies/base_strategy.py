from abc import ABC, abstractmethod
from typing import Set, Tuple, List
from node import SearchRootNode


class BaseSearchStrategy(ABC):
    """搜索策略抽象基类"""

    def __init__(self, rank: int = 0, **kwargs):
        self.rank = rank

    @abstractmethod
    def search(
        self,
        root_node: SearchRootNode,
        budget: int,
    ) -> Tuple[Set[Tuple[str, str, str]], int]:
        """
        执行搜索，发现正确三元组

        Args:
            root_node: 搜索根节点
            budget: 判别器调用预算上限

        Returns:
            (发现的三元组集合, 实际使用的预算)
        """
        pass

    def _rollout_to_leaf(
        self,
        node,
        action_sequence: List | None = None,
    ):
        """
        从给定节点沿 action_sequence 指定的过滤器路径逐步过滤，
        直至到达叶节点。若 action_sequence 为 None 则由子类自行决定路径。

        Returns:
            (叶节点, 经过的路径)
        """
        raise NotImplementedError
