from collections import defaultdict
import math
import random
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from node import SearchNode

class ContextualBanditRolloutPolicy:
    """
    一个MCTS的Rollout策略，它使用上下文多臂老虎机算法（基于UCB1）进行在线学习。

    这个类的实例应该在多个MCTS任务之间共享，以便积累知识。
    """
    def __init__(self):
        # 存储每个 (状态, 动作) 对的累计奖励
        # 结构: self.q_values[state_key][action_key] = total_reward
        self.q_values = defaultdict(lambda: defaultdict(float))

        # 存储每个 (状态, 动作) 对被选择的次数
        # 结构: self.counts[state_key][action_key] = count
        self.counts = defaultdict(lambda: defaultdict(int))

        # 存储每个状态被访问的总次数
        self.state_total_counts = defaultdict(int)

    def _get_state_key(self, node: 'SearchNode') -> str:
        """将当前节点的状态转化为一个离散的、可作为字典键的字符串。"""

        # 1. 离散化候选实体的数量 (这是最重要的上下文)
        num_candidates = len(node.unfiltered_entities)
        if num_candidates > 1000:
            size_bucket = "large"  # 候选集 > 1000
        elif num_candidates > 100:
            size_bucket = "medium" # 候选集 101-1000
        else:
            size_bucket = "small"  # 候选集 <= 100

        # 2. (可选) 可以在此加入更多上下文信息，例如在MCTS树中的深度
        #    为此，你需要在 SearchNode 中实现一个 get_depth() 方法
        # depth = node.get_depth()
        # depth_bucket = f"depth:{depth // 2}" # 每2层深度作为一个桶

        return f"size:{size_bucket}"

    def get_action(self, node: 'SearchNode', potential_actions: list) -> type:
        """根据UCB1算法选择最佳动作（即节点类型）。"""
        state_key = self._get_state_key(node)
        total_visits_at_state = self.state_total_counts[state_key]

        # 如果这是一个全新的状态，没有任何历史数据，则随机探索
        if total_visits_at_state == 0:
            return random.choice(potential_actions)

        best_action_class = None
        max_ucb_score = -1.0

        for action_class in potential_actions:
            action_key = action_class.__name__  # 例如 "KGENode", "GraphNode"

            # 如果某个动作在这个状态下从未被尝试过，优先选择它（无限大UCB值）
            if self.counts[state_key][action_key] == 0:
                return action_class

            # 1. 计算“利用”项 (Exploitation): 该动作的平均奖励
            average_reward = self.q_values[state_key][action_key] / self.counts[state_key][action_key]

            # 2. 计算“探索”项 (Exploration): 不确定性带来的奖励加成
            exploration_bonus = math.sqrt(
                2 * math.log(total_visits_at_state) / self.counts[state_key][action_key]
            )

            ucb_score = average_reward + exploration_bonus

            if ucb_score > max_ucb_score:
                max_ucb_score = ucb_score
                best_action_class = action_class

        return best_action_class

    def update(self, rollout_path: List[Tuple['SearchNode', type]], reward: float):
        """使用一次完整Rollout的结果来更新策略模型的权重。"""
        for state_node, action_class in rollout_path:
            state_key = self._get_state_key(state_node)
            action_key = action_class.__name__

            # 更新访问次数
            self.counts[state_key][action_key] += 1
            self.state_total_counts[state_key] += 1

            # 使用增量方式更新Q值（平均奖励），以保证数值稳定性
            # Q_new = Q_old + (reward - Q_old) / N
            q_old = self.q_values[state_key][action_key]
            n = self.counts[state_key][action_key]
            self.q_values[state_key][action_key] += (reward - q_old) / n
