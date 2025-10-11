import math
from collections import defaultdict
from typing import List, Tuple

from node import SearchNode, GraphNode, KGENode, LLMNode
from rollout_policy import ContextualBanditRolloutPolicy
from setup_logger import setup_logger, rank_logger


class MCTS:
    """Monte Carlo Tree Search实现"""

    def __init__(self, rank: int, exploration_weight: float = 1.0, rollout_policy=None):
        """
        初始化MCTS

        Args:
            exploration_weight: UCT公式中的探索权重参数
        """
        self.exploration_weight = exploration_weight
        self.rollout_policy = rollout_policy
        self.logger = setup_logger(self.__class__.__name__)
        self.reset()
        self.rank = rank

    def reset(self):
        """重置MCTS状态"""
        self.Q = defaultdict(float)  # 节点累计奖励
        self.N = defaultdict(int)    # 节点访问次数
        self.explored = set()        # 已探索节点集合

    def do_iteration(self, root_node: SearchNode) -> Tuple[List[Tuple[str, str, str]], int]:
        """
        执行一次完整的MCTS迭代

        Args:
            root_node: 搜索根节点

        Returns:
            (发现的正确三元组列表, 使用的分类器调用次数)
        """
        rank_logger(self.logger, self.rank)("Starting MCTS iteration")

        # Step 1: Selection - 选择到叶子节点的路径
        path = self._select(root_node)
        leaf = path[-1]

        rank_logger(self.logger, self.rank)(
            f"Selected path length: {len(path)}, leaf type: {leaf.__class__.__name__}")

        # Step 2: Expansion - 扩展叶子节点（如果不是终端节点）
        self._expand(leaf)

        # Step 3: Rollout - 在叶子节点或新扩展的节点上进行评估
        # rollout_results, total_budget_used = self._rollout(leaf)
        rollout_results, total_budget_used, path_taken_in_rollout = self._rollout(leaf)

        # Step 4: Backpropagation
        reward = self._calculate_reward(rollout_results, total_budget_used)
        self._backpropagate(path, reward)

        # Step 5: Update the policy
        if self.rollout_policy and path_taken_in_rollout:
            self.rollout_policy.update(path_taken_in_rollout, reward)

        return rollout_results, total_budget_used

    def _select(self, node: SearchNode) -> List[SearchNode]:
        """
        Selection阶段：从根节点开始，使用UCT策略选择到叶子节点的路径

        Args:
            node: 当前节点

        Returns:
            从根节点到叶子节点的路径
        """
        path = []

        while True:
            path.append(node)

            if node not in self.explored or node.is_terminal():
                # 找到未探索的节点或终端节点
                return path

            # 检查是否有未探索的子节点
            children = node.find_children()
            unexplored = children - self.explored

            if unexplored:
                # 选择一个未探索的子节点
                child = next(iter(unexplored))  # 取第一个未探索的子节点
                path.append(child)
                return path

            # 所有子节点都已探索，使用UCT选择最优子节点
            node = self._uct_select(node)

    def _uct_select(self, node: SearchNode) -> SearchNode:
        """
        使用UCT公式选择最优子节点

        Args:
            node: 父节点

        Returns:
            选中的子节点
        """
        children = node.find_children()

        # 确保所有子节点都被探索过
        assert all(child in self.explored for child in children), \
            "UCT selection called with unexplored children"

        # 使用UCT公式选择
        return max(children, key=self._get_uct_value)

    def _get_uct_value(self, node: SearchNode) -> float:
        """
        计算节点的UCT值

        Args:
            node: 节点

        Returns:
            UCT值
        """
        if self.N[node] == 0:
            return float('inf')  # 未访问的节点具有最高优先级

        exploitation = self.Q[node] / self.N[node]  # 开发项
        exploration = self.exploration_weight * math.sqrt(
            math.log(self.N[node.parent]) / self.N[node]
        )  # 探索项

        return exploitation + exploration

    def _expand(self, node: SearchNode):
        """
        Expansion阶段：扩展节点的子节点

        Args:
            node: 要扩展的节点

        Returns:
            新扩展的子节点列表
        """
        if node in self.explored:
            return

        # 扩展节点
        node.expand()
        self.explored.add(node)

    def _rollout(self, node: SearchNode) -> Tuple[List[Tuple[str, str, str]], int, List[Tuple[SearchNode, type]]]:
        """
        Rollout阶段：使用在线学习策略（或随机策略）来评估节点质量

        Returns:
            (发现的正确三元组列表, 使用的分类器调用次数, 本次Rollout的决策路径)
        """
        current_node = node
        rollout_path = []  # 记录(状态节点, 所选动作类)的决策路径

        while not current_node.is_terminal():
            child_context = current_node._make_child_context()

            # 确定当前所有可能的动作
            potential_actions = [GraphNode, KGENode, LLMNode]

            # 如果配置了学习策略，则使用它来选择动作
            if self.rollout_policy:
                chosen_action_class = self.rollout_policy.get_action(current_node, potential_actions)
            # 否则，退回至纯随机选择
            else:
                chosen_action_class = random.choice(potential_actions)

            # 记录下这次的决策
            rollout_path.append((current_node, chosen_action_class))

            # 执行选择的动作，进入下一个状态
            current_node = chosen_action_class(child_context)

        # 到达终端节点，进行最终评估
        results, budget_used = current_node.evaluate_candidates()

        return results, budget_used, rollout_path

    def _calculate_reward(self, rollout_results: List[Tuple[str, str, str]], total_budget_used: int) -> float:
        """
        根据rollout结果计算奖励值

        Args:
            rollout_results: rollout阶段发现的正确三元组列表

        Returns:
            奖励值
        """
        reward = len(rollout_results) / (total_budget_used + 1)  # 避免除以零

        rank_logger(self.logger, self.rank)(
            f"Calculated reward: {reward} from {len(rollout_results)} correct triplets")

        return reward

    def _backpropagate(self, path: List[SearchNode], reward: float):
        """
        Backpropagation阶段：沿路径向上传播奖励

        Args:
            path: 从根到叶子的节点路径
            reward: 奖励值
        """
        rank_logger(self.logger, self.rank)(
            f"Backpropagating reward {reward} along path of length {len(path)}")

        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def choose_best_child(self, node: SearchNode) -> SearchNode:
        """
        选择最佳子节点（基于平均奖励）

        Args:
            node: 父节点

        Returns:
            最佳子节点
        """
        if node.is_terminal():
            raise RuntimeError("choose_best_child called on terminal node")

        if node not in self.explored:
            return node.find_random_child()

        children = node.find_children()

        if not children:
            return None

        # 选择平均奖励最高的子节点
        best_child = max(children, key=self._get_average_reward)

        rank_logger(self.logger, self.rank)(
            f"Chose best child with average reward: {self._get_average_reward(best_child)}")

        return best_child

    def _get_average_reward(self, node: SearchNode) -> float:
        """
        获取节点的平均奖励

        Args:
            node: 节点

        Returns:
            平均奖励值
        """
        return float('-inf') if self.N[node] == 0 else (self.Q[node] / self.N[node])

    def get_statistics(self) -> dict:
        """获取MCTS统计信息"""
        return {
            "total_nodes_explored": len(self.explored),
            "total_visits": sum(self.N.values()),
            "total_reward": sum(self.Q.values()),
            "average_reward_per_visit": sum(self.Q.values()) / max(sum(self.N.values()), 1)
        }
