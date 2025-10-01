"""
搜索节点定义：定义MCTS中使用的各种节点类型
每个节点代表搜索空间中的一个状态，包含候选实体的子集
"""

from typing import Set, List
import random
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import re
from collections import Counter

from kg_data_loader import KGDataLoader
from discriminator import TriplesDiscriminator
from model_calls import OpenKEClient
from setup_logger import setup_logger, rank_logger


@dataclass
class Context:
    rank: int
    sparse_entity: str
    position: str
    relation: str
    unfiltered_entities: Set[str]
    output_folder: str
    data_loader: KGDataLoader
    triplet_discriminator: TriplesDiscriminator
    kge_model: OpenKEClient
    leaf_threshold: int
    parent: Optional['SearchNode'] = None


class SearchNode(ABC):
    """搜索节点抽象基类"""

    def __init__(self, context: Context):
        """
        初始化搜索节点

        Args:
            sparse_entity: 稀疏实体ID
            position: 实体在三元组中的位置 ('head' 或 'tail')
            relation: 关系类型
            unfiltered_entities: 候选目标实体集合
            data_loader: 数据加载器
            triplet_discriminator: 三元组判别器
            leaf_threshold: 叶子节点阈值
            parent: 父节点
        """
        self.rank = context.rank
        self.sparse_entity = context.sparse_entity
        self.position = context.position
        self.relation = context.relation
        self.unfiltered_entities = context.unfiltered_entities
        self.output_folder = context.output_folder
        self.candidate_entities = set()
        self.data_loader = context.data_loader
        self.triplet_discriminator = context.triplet_discriminator
        self.kge_model = context.kge_model
        self.leaf_threshold = context.leaf_threshold
        self.parent = context.parent
        self.children = None
        self.logger = setup_logger(f"{self.__class__.__name__}")

        self.llm_limit = 5000

    @abstractmethod
    def _filter(self):
        """过滤候选实体，产生新的候选实体集合"""
        pass

    def find_children(self) -> Set['SearchNode']:
        """查找子节点"""
        if self.children is None:
            return set()
        return self.children

    def find_random_child(self) -> 'SearchNode':
        """随机选择一个子节点"""
        child_context = self._make_child_context()
        if len(child_context.unfiltered_entities) < self.llm_limit:
            return random.choice([KGENode, GraphNode, LLMNode])(child_context)
        else:
            return random.choice([KGENode, GraphNode])(child_context)

    def expand(self):
        """扩展根节点，生成不同的过滤策略子节点"""
        if self.children is not None:
            return

        self.children = set()

        # 生成背景信息
        child_context = self._make_child_context()

        # 1. 基于知识图谱结构的过滤节点
        self.children.add(GraphNode(child_context))

        # 2. 基于KGE模型的过滤节点
        self.children.add(KGENode(child_context))

        # 3. 基于LLM的过滤节点
        if len(child_context.unfiltered_entities) < self.llm_limit:
            self.children.add(LLMNode(child_context))

    def is_terminal(self) -> bool:
        """判断是否为终端节点（候选实体数量小于阈值）"""
        return len(self.candidate_entities) <= self.leaf_threshold

    def evaluate_candidates(self) -> Tuple[List[Tuple[str, str, str]], int]:
        """
        评估候选实体，返回正确的三元组

        Returns:
            (正确的三元组列表, 使用的分类器调用次数)
        """
        correct_triplets = []
        budget_used = 0

        # 构造所有需要评估的三元组
        triplets_to_evaluate = []
        triplet_indices = []  # 记录原始索引，用于后续匹配结果

        for idx, entity in enumerate(self.candidate_entities):
            # 构造三元组
            if self.position == 'head':
                triplet = (self.sparse_entity, self.relation, entity)
            else:  # position == 'tail'
                triplet = (entity, self.relation, self.sparse_entity)

            # 跳过已存在的三元组
            if self.data_loader.triplet_exists(*triplet):
                continue

            triplets_to_evaluate.append(triplet)
            triplet_indices.append(idx)

        # 如果有需要评估的三元组，则使用批量分类器进行判断
        results = []
        if triplets_to_evaluate:
            # 预处理所有三元组为分类器输入格式
            discriminator_inputs = [
                self._preprocess_triplet(triplet)
                for triplet in triplets_to_evaluate
            ]

            # 批量判断
            results = self.triplet_discriminator.judge_batch(discriminator_inputs)

        # 收集正确的三元组
        for i, (triplet, result) in enumerate(zip(triplets_to_evaluate, results)):
            if result["is_correct"]:
                correct_triplets.append(triplet)

        # 更新预算使用量
        budget_used = len(triplets_to_evaluate)

        rank_logger(self.logger, self.rank)(
            f"Evaluated {len(self.candidate_entities)} candidates, "
            f"found {len(correct_triplets)} correct triplets"
        )

        return correct_triplets, budget_used

    def _make_child_context(self) -> Context:
        return Context(
            rank=self.rank,
            sparse_entity=self.sparse_entity,
            position=self.position,
            relation=self.relation,
            unfiltered_entities=self.candidate_entities,
            output_folder=self.output_folder,
            data_loader=self.data_loader,
            triplet_discriminator=self.triplet_discriminator,
            kge_model=self.kge_model,
            leaf_threshold=self.leaf_threshold,
            parent=self
        )

    def _preprocess_triplet(self, triplet: Tuple[str, str, str]) -> dict:
        """
        预处理三元组，转化为dicriminator需要的格式

        Output:
            {
                "input": input text,
                "embedding_ids": [head_id, relation_id, tail_id]
            }
        """
        head_code, rel_code, tail_code = triplet
        input_text = "The input triple: \n( {head}, {rel}, {tail} )\n".format(
            head=self.data_loader.entity2name.get(
                head_code, head_code).replace('_', ' '),
            rel=rel_code.replace('/', ' '),
            tail=self.data_loader.entity2name.get(
                tail_code, tail_code).replace('_', ' ')
        )
        embedding_ids = [self.data_loader.entity2id[head_code],
                         self.data_loader.relation2id[rel_code],
                         self.data_loader.entity2id[tail_code]]
        return {"input": input_text, "embedding_ids": embedding_ids}


class SearchRootNode(SearchNode):
    """搜索根节点"""

    def __init__(self, context: Context):
        super().__init__(context)
        self.candidate_entities = self._filter()

    def _filter(self) -> Set[str]:
        """根节点不进行过滤，直接返回初始候选实体"""
        return self.unfiltered_entities


class KGENode(SearchNode):
    """基于KGE模型的过滤策略节点"""

    def __init__(self, context: Context):
        super().__init__(context)
        self.candidate_entities = self._filter()

    def _filter(self, top_p: float = 0.2) -> Set[str]:
        """基于KGE模型的打分结果进行过滤，保留得分前top_p比例的实体"""
        if not self.unfiltered_entities:
            return set()

        # 获取当前实体和关系的ID
        sparse_entity_id = self.data_loader.entity2id.get(self.sparse_entity)
        relation_id = self.data_loader.relation2id.get(self.relation)

        if sparse_entity_id is None or relation_id is None:
            self.logger.warning(
                "Sparse entity or relation not found in data loader.")
            return set()

        # 候选实体转ID列表
        candidate_ids = [
            self.data_loader.entity2id[ent]
            for ent in self.unfiltered_entities
            if ent in self.data_loader.entity2id
        ]

        if not candidate_ids:
            self.logger.warning(
                "No valid candidate entities to filter with KGE.")
            return set()

        try:
            if self.position == 'head':
                # 固定 head 和 relation，筛选 tail
                top_ids = self.kge_model.get_top_p_tails(
                    head_id=sparse_entity_id,
                    rel_id=relation_id,
                    candidate_tails=candidate_ids,
                    p=top_p
                )
            elif self.position == 'tail':
                # 固定 tail 和 relation，筛选 head
                top_ids = self.kge_model.get_top_p_heads(
                    tail_id=sparse_entity_id,
                    rel_id=relation_id,
                    candidate_heads=candidate_ids,
                    p=top_p
                )
            else:
                raise ValueError(f"Invalid position: {self.position}")
        except Exception as e:
            self.logger.error(f"Error during KGE filtering: {e}")
            return set()

        # 将ID映射回实体名称
        id2entity = self.data_loader.id2entity
        filtered_entities = {
            id2entity[idx] for idx in top_ids if idx in id2entity
        }

        rank_logger(self.logger, self.rank)(
            f"KGENode filtered {len(filtered_entities)} entities from {len(candidate_ids)} candidates.")
        return filtered_entities


class GraphNode(SearchNode):
    """基于图结构的过滤策略节点"""

    def __init__(self, context: Context):
        super().__init__(context)
        self.candidate_entities = self._filter()

    def _filter(self, top_p: float = 0.5) -> Set[str]:
        """基于图结构的启发式方法进行过滤"""

        if not self.unfiltered_entities:
            return set()

        # 在循环外计算一次稀疏实体的一跳邻居，避免重复计算
        sparse_neighbors = set(
            self.data_loader.get_one_hop_neighbors(self.sparse_entity))

        # 计算各种评分
        structural_scores = {}
        semantic_scores = {}
        popularity_scores = {}

        for entity in self.unfiltered_entities:
            # 将 sparse_neighbors 作为参数传入
            structural_scores[entity] = self._calculate_structural_similarity(
                entity, sparse_neighbors)
            # semantic_scores[entity] = self._calculate_semantic_similarity(
            #     entity)
            semantic_scores[entity] = 0  # 暂时不使用语义相似度
            popularity_scores[entity] = self._calculate_relation_popularity(
                entity)

        # 归一化评分
        def normalize_scores(scores_dict):
            if not scores_dict:
                return scores_dict
            max_score = max(scores_dict.values())
            if max_score == 0:
                return scores_dict
            return {k: v / max_score for k, v in scores_dict.items()}

        structural_scores = normalize_scores(structural_scores)
        semantic_scores = normalize_scores(semantic_scores)
        popularity_scores = normalize_scores(popularity_scores)

        # 计算综合评分（可调整权重）
        alpha, beta, gamma = 0.4, 0.3, 0.3  # 结构、语义、热度权重
        combined_scores = {}

        for entity in self.unfiltered_entities:
            combined_scores[entity] = (
                alpha * structural_scores.get(entity, 0) +
                beta * semantic_scores.get(entity, 0) +
                gamma * popularity_scores.get(entity, 0)
            )

        # 排序并选择前p比例的实体
        sorted_entities = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True)
        keep_count = max(1, int(len(sorted_entities) * top_p))

        # 返回候选实体集合
        candidate_entities = {
            entity for entity, _ in sorted_entities[:keep_count]
        }
        rank_logger(self.logger, self.rank)(
            f"GraphNode filtered {len(candidate_entities)} entities from {len(self.unfiltered_entities)} candidates.")
        return candidate_entities

    def _extract_keywords_from_description(self, entity_id: str, top_k: int = 5) -> List[str]:
        """从实体描述中提取关键词"""
        description = self.data_loader.get_entity_description(entity_id)
        if not description:
            return []

        # 去除标点，转小写，分词
        words = re.findall(r'\b\w+\b', description.lower())
        # 常见停用词
        stop_words = {
            'the', 'is', 'in', 'a', 'an', 'and', 'or', 'of', 'to',
            'for', 'with', 'on', 'at', 'by', 'this', 'that', 'these', 'those'
        }
        words = [w for w in words if len(w) > 2 and w not in stop_words]

        # 统计词频并返回高频词
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(top_k)]

    def _calculate_structural_similarity(self, candidate_entity: str, sparse_neighbors: set) -> float:
        """计算结构相似性评分（基于Jaccard系数）"""
        # 获取候选实体的一跳邻居结构
        candidate_neighbors = set(
            self.data_loader.get_one_hop_neighbors(candidate_entity))

        # 直接使用传入的 sparse_neighbors 计算Jaccard相似度
        intersection = sparse_neighbors & candidate_neighbors
        union = sparse_neighbors | candidate_neighbors

        return len(intersection) / len(union) if union else 0.0

    def _calculate_semantic_similarity(self, candidate_entity: str) -> float:
        """计算语义相似性评分（基于描述文本的关键词重叠）"""
        # 提取稀疏实体的关键词
        sparse_keywords = self._extract_keywords_from_description(
            self.sparse_entity)
        if not sparse_keywords:
            return 0.0

        # 提取候选实体的关键词
        candidate_keywords = self._extract_keywords_from_description(
            candidate_entity)
        if not candidate_keywords:
            return 0.0

        # 计算关键词重叠度
        sparse_set = set(sparse_keywords)
        candidate_set = set(candidate_keywords)

        intersection = sparse_set & candidate_set
        union = sparse_set | candidate_set

        return len(intersection) / len(union) if union else 0.0

    def _calculate_relation_popularity(self, candidate_entity: str) -> int:
        """计算关系热度评分"""
        # 统计候选实体在指定关系下的出现频次
        if self.position == 'head':
            # 作为头实体的频次
            count = 0
            for rel, tail in self.data_loader.outgoing_edges.get(candidate_entity, []):
                if rel == self.relation:
                    count += 1
            return count
        else:  # position == 'tail'
            # 作为尾实体的频次
            count = 0
            for rel, head in self.data_loader.incoming_edges.get(candidate_entity, []):
                if rel == self.relation:
                    count += 1
            return count


class LLMNode(SearchNode):
    """基于大语言模型的过滤策略节点"""

    def __init__(self, context: Context):
        super().__init__(context)
        self.candidate_entities = self._filter()

    def _get_target_embedding(self) -> np.ndarray:
        """获取目标实体的嵌入表示"""
        pass

    def _filter(self, top_p: float = 0.2) -> Set[str]:
        """基于LLM的语义分析进行过滤"""
        # feature_vec = sparse_entity_vec +/- relation_vec
        # feature_vec: ndarray, shape (embedding_dim, 1)
        if self.position == 'head':
            feature_embeddings = self.data_loader.relation2embedding[self.relation] + \
                self.data_loader.entity2embedding[self.sparse_entity]
        else:  # position == 'tail'
            feature_embeddings = self.data_loader.entity2embedding[self.sparse_entity] - \
                self.data_loader.relation2embedding[self.relation]
        feature_embeddings = self._get_target_embedding(self.sparse_entity).reshape(-1, 1)

        # 获取候选实体名称的嵌入
        unfiltered_entities_list = list(self.unfiltered_entities)
        entity_embeddings = np.array([
            self.data_loader.entity2embedding[ent] for ent in unfiltered_entities_list
        ])

        # 计算相似度
        scores = np.dot(entity_embeddings, feature_embeddings.T)

        # 选择 top_p 的实体
        num_top = max(1, int(len(unfiltered_entities_list) * top_p))
        top_indices = np.argpartition(scores, -num_top)[-num_top:]
        top_entities = [unfiltered_entities_list[i] for i in top_indices]

        candidate_entities = set(top_entities)
        rank_logger(self.logger, self.rank)(
            f"LLMNode filtered {len(candidate_entities)} entities from {len(self.unfiltered_entities)} candidates.")
        return candidate_entities
