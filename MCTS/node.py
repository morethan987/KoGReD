import random
from abc import ABC, abstractmethod
from typing import Set, List, Tuple, Optional

from kg_data_loader import KGDataLoader
from LLM_Discriminator.discriminator import TriplesDiscriminator
from model_calls import RemoteLLM
from setup_logger import setup_logger
from prompts import SEMANTIC_ANALYSIS


class SearchNode(ABC):
    """æ�œç´¢èŠ‚ç‚¹æŠ½è±¡åŸºç±»"""

    def __init__(self,
                 sparse_entity: str,
                 position: str,
                 relation: str,
                 candidate_entities: Set[str],
                 data_loader: KGDataLoader,
                 triplet_discriminator: TriplesDiscriminator,
                 leaf_threshold: int,
                 parent: Optional['SearchNode'] = None):
        """
        åˆ�å§‹åŒ–æ�œç´¢èŠ‚ç‚¹

        Args:
            sparse_entity: ç¨€ç–�å®žä½“ID
            position: å®žä½“åœ¨ä¸‰å…ƒç»„ä¸­çš„ä½�ç½® ('head' æˆ– 'tail')
            relation: å…³ç³»ç±»åž‹
            candidate_entities: å€™é€‰ç›®æ ‡å®žä½“é›†å�ˆ
            data_loader: æ•°æ�®åŠ è½½å™¨
            triplet_discriminator: ä¸‰å…ƒç»„åˆ¤åˆ«å™¨
            leaf_threshold: å�¶å­�èŠ‚ç‚¹é˜ˆå€¼
            parent: çˆ¶èŠ‚ç‚¹
        """
        self.sparse_entity = sparse_entity
        self.position = position
        self.relation = relation
        self.candidate_entities = candidate_entities
        self.data_loader = data_loader
        self.triplet_discriminator = triplet_discriminator
        self.leaf_threshold = leaf_threshold
        self.parent = parent
        self.children = None
        self.logger = setup_logger(f"{self.__class__.__name__}")

    @abstractmethod
    def find_children(self) -> Set['SearchNode']:
        """æŸ¥æ‰¾å­�èŠ‚ç‚¹"""
        pass

    @abstractmethod
    def find_random_child(self) -> Optional['SearchNode']:
        """éš�æœºé€‰æ‹©ä¸€ä¸ªå­�èŠ‚ç‚¹"""
        pass

    @abstractmethod
    def expand(self):
        """æ‰©å±•èŠ‚ç‚¹ï¼Œç”Ÿæˆ�å­�èŠ‚ç‚¹"""
        pass

    def is_terminal(self) -> bool:
        """åˆ¤æ–­æ˜¯å�¦ä¸ºç»ˆç«¯èŠ‚ç‚¹ï¼ˆå€™é€‰å®žä½“æ•°é‡�å°�äºŽé˜ˆå€¼ï¼‰"""
        return len(self.candidate_entities) <= self.leaf_threshold

    def evaluate_candidates(self) -> Tuple[List[Tuple[str, str, str]], int]:
        """
        è¯„ä¼°å€™é€‰å®žä½“ï¼Œè¿”å›žæ­£ç¡®çš„ä¸‰å…ƒç»„

        Returns:
            (æ­£ç¡®çš„ä¸‰å…ƒç»„åˆ—è¡¨, ä½¿ç”¨çš„åˆ†ç±»å™¨è°ƒç”¨æ¬¡æ•°)
        """
        correct_triplets = []
        budget_used = 0

        for entity in self.candidate_entities:
            # æž„é€ ä¸‰å…ƒç»„
            if self.position == 'head':
                triplet = (self.sparse_entity, self.relation, entity)
            else:  # position == 'tail'
                triplet = (entity, self.relation, self.sparse_entity)

            # è·³è¿‡å·²å­˜åœ¨çš„ä¸‰å…ƒç»„
            if self.data_loader.triplet_exists(*triplet):
                continue

            # ä½¿ç”¨åˆ†ç±»å™¨åˆ¤æ–­
            if self.triplet_discriminator.judge_one(self._preprocess_triplet(triplet)):
                correct_triplets.append(triplet)

            budget_used += 1

        self.logger.debug(f"Evaluated {len(self.candidate_entities)} candidates, "
                          f"found {len(correct_triplets)} correct triplets")

        return correct_triplets, budget_used

    def _preprocess_triplet(self, triplet: Tuple[str, str, str]) -> dict:
        """
        é¢„å¤„ç�†ä¸‰å…ƒç»„ï¼Œè½¬åŒ–ä¸ºdicriminatoréœ€è¦�çš„æ ¼å¼�

        Output:
            {
                "input": input text,
                "embedding_ids": [head_id, relation_id, tail_id]
            }
        """
        head_code, rel_code, tail_code = triplet
        input_text = f"The input triple: \n( {self.data_loader.entity2name.get(head_code, head_code).replace('_', ' ')}, {rel_code.replace('/', ' ')}, {self.data_loader.entity2name.get(tail_code, tail_code).replace('_', ' ')} )\n"
        embedding_ids = [self.data_loader.entity2id[head_code],
                         self.data_loader.relation2id[rel_code],
                         self.data_loader.entity2id[tail_code]]
        return {"input": input_text, "embedding_ids": embedding_ids}


class SearchRootNode(SearchNode):
    """æ�œç´¢æ ¹èŠ‚ç‚¹"""

    def find_children(self) -> Set[SearchNode]:
        """æ ¹èŠ‚ç‚¹çš„å­�èŠ‚ç‚¹æ˜¯å�„ç§�è¿‡æ»¤ç­–ç•¥èŠ‚ç‚¹"""
        if self.children is None:
            return set()
        return self.children

    def find_random_child(self) -> Optional[SearchNode]:
        """éš�æœºé€‰æ‹©ä¸€ä¸ªå­�èŠ‚ç‚¹"""
        if self.children:
            return random.choice(list(self.children))
        return None

    def expand(self):
        """æ‰©å±•æ ¹èŠ‚ç‚¹ï¼Œç”Ÿæˆ�ä¸�å�Œçš„è¿‡æ»¤ç­–ç•¥å­�èŠ‚ç‚¹"""
        if self.children is not None:
            return

        self.children = set()

        # 1. åŸºäºŽçŸ¥è¯†å›¾è°±é‚»å±…çš„è¿‡æ»¤èŠ‚ç‚¹
        self.children.add(NeighborFilterNode(
            sparse_entity=self.sparse_entity,
            position=self.position,
            relation=self.relation,
            candidate_entities=self.candidate_entities,
            data_loader=self.data_loader,
            triplet_discriminator=self.triplet_discriminator,
            leaf_threshold=self.leaf_threshold,
            parent=self
        ))

        # 2. åŸºäºŽå…³ç³»ç±»åž‹çš„è¿‡æ»¤èŠ‚ç‚¹
        self.children.add(RelationFilterNode(
            sparse_entity=self.sparse_entity,
            position=self.position,
            relation=self.relation,
            candidate_entities=self.candidate_entities,
            data_loader=self.data_loader,
            triplet_discriminator=self.triplet_discriminator,
            leaf_threshold=self.leaf_threshold,
            parent=self
        ))

        # 3. åŸºäºŽLLMè¯­ä¹‰çš„è¿‡æ»¤èŠ‚ç‚¹
        self.children.add(SemanticFilterNode(
            sparse_entity=self.sparse_entity,
            position=self.position,
            relation=self.relation,
            candidate_entities=self.candidate_entities,
            data_loader=self.data_loader,
            triplet_discriminator=self.triplet_discriminator,
            leaf_threshold=self.leaf_threshold,
            parent=self
        ))


class NeighborFilterNode(SearchNode):
    """åŸºäºŽé‚»å±…å…³ç³»çš„è¿‡æ»¤èŠ‚ç‚¹"""

    def find_children(self) -> Set[SearchNode]:
        if self.children is None:
            return set()
        return self.children

    def find_random_child(self) -> Optional[SearchNode]:
        if self.children:
            return random.choice(list(self.children))
        return None

    def expand(self):
        """æ ¹æ�®é‚»å±…å…³ç³»åˆ’åˆ†å€™é€‰å®žä½“"""
        if self.children is not None:
            return

        self.children = set()

        # èŽ·å�–ç¨€ç–�å®žä½“çš„ä¸€è·³å’ŒäºŒè·³é‚»å±…
        one_hop_neighbors = self.data_loader.get_one_hop_neighbors(
            self.sparse_entity)
        two_hop_neighbors = self.data_loader.get_two_hop_neighbors(
            self.sparse_entity)

        # æ ¹æ�®ä¸Žç¨€ç–�å®žä½“çš„è·�ç¦»å…³ç³»åˆ’åˆ†å€™é€‰å®žä½“
        one_hop_candidates = self.candidate_entities & one_hop_neighbors
        two_hop_candidates = self.candidate_entities & two_hop_neighbors
        other_candidates = self.candidate_entities - \
            one_hop_neighbors - two_hop_neighbors

        # åˆ›å»ºå­�èŠ‚ç‚¹
        if one_hop_candidates:
            self.children.add(LeafNode(
                sparse_entity=self.sparse_entity,
                position=self.position,
                relation=self.relation,
                candidate_entities=one_hop_candidates,
                data_loader=self.data_loader,
                triplet_discriminator=self.triplet_discriminator,
                leaf_threshold=self.leaf_threshold,
                parent=self,
                node_type="one_hop"
            ))

        if two_hop_candidates:
            self.children.add(LeafNode(
                sparse_entity=self.sparse_entity,
                position=self.position,
                relation=self.relation,
                candidate_entities=two_hop_candidates,
                data_loader=self.data_loader,
                triplet_discriminator=self.triplet_discriminator,
                leaf_threshold=self.leaf_threshold,
                parent=self,
                node_type="two_hop"
            ))

        if other_candidates:
            # å¦‚æžœå…¶ä»–å€™é€‰å®žä½“è¿‡å¤šï¼Œè¿›ä¸€æ­¥éš�æœºåˆ’åˆ†
            if len(other_candidates) > self.leaf_threshold * 3:
                # éš�æœºåˆ’åˆ†ä¸ºå¤šä¸ªå­�é›†
                candidates_list = list(other_candidates)
                random.shuffle(candidates_list)

                chunk_size = len(candidates_list) // 3
                for i in range(3):
                    start_idx = i * chunk_size
                    end_idx = len(candidates_list) if i == 2 else (
                        i + 1) * chunk_size
                    chunk = set(candidates_list[start_idx:end_idx])

                    if chunk:
                        self.children.add(LeafNode(
                            sparse_entity=self.sparse_entity,
                            position=self.position,
                            relation=self.relation,
                            candidate_entities=chunk,
                            data_loader=self.data_loader,
                            triplet_discriminator=self.triplet_discriminator,
                            leaf_threshold=self.leaf_threshold,
                            parent=self,
                            node_type=f"other_chunk_{i}"
                        ))
            else:
                self.children.add(LeafNode(
                    sparse_entity=self.sparse_entity,
                    position=self.position,
                    relation=self.relation,
                    candidate_entities=other_candidates,
                    data_loader=self.data_loader,
                    triplet_discriminator=self.triplet_discriminator,
                    leaf_threshold=self.leaf_threshold,
                    parent=self,
                    node_type="other"
                ))


class RelationFilterNode(SearchNode):
    """åŸºäºŽå…³ç³»ç±»åž‹çš„è¿‡æ»¤èŠ‚ç‚¹"""

    def find_children(self) -> Set[SearchNode]:
        if self.children is None:
            return set()
        return self.children

    def find_random_child(self) -> Optional[SearchNode]:
        if self.children:
            return random.choice(list(self.children))
        return None

    def expand(self):
        """æ ¹æ�®å€™é€‰å®žä½“åœ¨å…¶ä»–å…³ç³»ä¸­çš„å‡ºçŽ°é¢‘çŽ‡åˆ’åˆ†"""
        if self.children is not None:
            return

        self.children = set()

        # èŽ·å�–ä¸Žå½“å‰�å…³ç³»ç›¸å…³çš„å®žä½“ï¼ˆåœ¨å…¶ä»–ä¸‰å…ƒç»„ä¸­å‡ºçŽ°è¿‡è¯¥å…³ç³»çš„å®žä½“ï¼‰
        relation_entities = self.data_loader.get_entities_with_relation(
            self.relation, self.position
        )

        # åˆ’åˆ†å€™é€‰å®žä½“
        related_candidates = self.candidate_entities & relation_entities
        unrelated_candidates = self.candidate_entities - relation_entities

        if related_candidates:
            self.children.add(LeafNode(
                sparse_entity=self.sparse_entity,
                position=self.position,
                relation=self.relation,
                candidate_entities=related_candidates,
                data_loader=self.data_loader,
                triplet_discriminator=self.triplet_discriminator,
                leaf_threshold=self.leaf_threshold,
                parent=self,
                node_type="relation_related"
            ))

        if unrelated_candidates:
            # å¦‚æžœæ— å…³å®žä½“è¿‡å¤šï¼Œéš�æœºåˆ’åˆ†
            if len(unrelated_candidates) > self.leaf_threshold * 2:
                candidates_list = list(unrelated_candidates)
                random.shuffle(candidates_list)

                mid_point = len(candidates_list) // 2
                chunk1 = set(candidates_list[:mid_point])
                chunk2 = set(candidates_list[mid_point:])

                for i, chunk in enumerate([chunk1, chunk2]):
                    if chunk:
                        self.children.add(LeafNode(
                            sparse_entity=self.sparse_entity,
                            position=self.position,
                            relation=self.relation,
                            candidate_entities=chunk,
                            data_loader=self.data_loader,
                            triplet_discriminator=self.triplet_discriminator,
                            leaf_threshold=self.leaf_threshold,
                            parent=self,
                            node_type=f"relation_unrelated_{i}"
                        ))
            else:
                self.children.add(LeafNode(
                    sparse_entity=self.sparse_entity,
                    position=self.position,
                    relation=self.relation,
                    candidate_entities=unrelated_candidates,
                    data_loader=self.data_loader,
                    triplet_discriminator=self.triplet_discriminator,
                    leaf_threshold=self.leaf_threshold,
                    parent=self,
                    node_type="relation_unrelated"
                ))


class SemanticFilterNode(SearchNode):
    """åŸºäºŽLLMè¯­ä¹‰ç�†è§£çš„è¿‡æ»¤èŠ‚ç‚¹"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_client = RemoteLLM()

    def find_children(self) -> Set[SearchNode]:
        if self.children is None:
            return set()
        return self.children

    def find_random_child(self) -> Optional[SearchNode]:
        if self.children:
            return random.choice(list(self.children))
        return None

    def expand(self):
        """ä½¿ç”¨LLMè¿›è¡Œè¯­ä¹‰è¿‡æ»¤"""
        if self.children is not None:
            return

        self.children = set()

        # å¦‚æžœå€™é€‰å®žä½“è¾ƒå°‘ï¼Œç›´æŽ¥ä½œä¸ºå�¶å­�èŠ‚ç‚¹
        if len(self.candidate_entities) <= self.leaf_threshold * 2:
            self.children.add(LeafNode(
                sparse_entity=self.sparse_entity,
                position=self.position,
                relation=self.relation,
                candidate_entities=self.candidate_entities,
                data_loader=self.data_loader,
                triplet_discriminator=self.triplet_discriminator,
                leaf_threshold=self.leaf_threshold,
                parent=self,
                node_type="semantic_all"
            ))
            return

        # ä½¿ç”¨LLMè¿›è¡Œè¯­ä¹‰åˆ†æž�å’Œåˆ†ç»„
        try:
            semantic_groups = self._semantic_grouping()

            for i, group in enumerate(semantic_groups):
                if group:
                    self.children.add(LeafNode(
                        sparse_entity=self.sparse_entity,
                        position=self.position,
                        relation=self.relation,
                        candidate_entities=group,
                        data_loader=self.data_loader,
                        triplet_discriminator=self.triplet_discriminator,
                        leaf_threshold=self.leaf_threshold,
                        parent=self,
                        node_type=f"semantic_group_{i}"
                    ))

        except Exception as e:
            self.logger.warning(
                f"LLM semantic grouping failed: {e}, using random split")
            # å¦‚æžœLLMè°ƒç”¨å¤±è´¥ï¼Œå›žé€€åˆ°éš�æœºåˆ’åˆ†
            self._random_split()

    def _semantic_grouping(self) -> List[Set[str]]:
        """ä½¿ç”¨LLMè¿›è¡Œè¯­ä¹‰åˆ†ç»„"""
        # èŽ·å�–å®žä½“æ��è¿°ä¿¡æ�¯
        sparse_name = self.data_loader.get_entity_name(self.sparse_entity)
        sparse_desc = self.data_loader.get_entity_description(
            self.sparse_entity)

        # éš�æœºé‡‡æ ·ä¸€äº›å€™é€‰å®žä½“è¿›è¡Œåˆ†æž�ï¼ˆé�¿å…�è¾“å…¥è¿‡é•¿ï¼‰
        sample_size = min(20, len(self.candidate_entities))
        sample_entities = random.sample(
            list(self.candidate_entities), sample_size)

        # æž„é€ LLMè¾“å…¥
        prompt = self._build_semantic_prompt(
            sparse_name, sparse_desc, sample_entities)

        # è°ƒç”¨LLM
        response = self.llm_client.get_output(prompt)

        # è§£æž�å“�åº”å¹¶åˆ†ç»„
        groups = self._parse_semantic_response(response, sample_entities)

        # ä¸ºæœªé‡‡æ ·çš„å®žä½“éš�æœºåˆ†é…�ç»„
        remaining_entities = self.candidate_entities - set(sample_entities)
        if remaining_entities and groups:
            # å°†å‰©ä½™å®žä½“éš�æœºåˆ†é…�åˆ°çŽ°æœ‰ç»„ä¸­
            remaining_list = list(remaining_entities)
            random.shuffle(remaining_list)

            group_count = len(groups)
            for i, entity in enumerate(remaining_list):
                groups[i % group_count].add(entity)

        return groups

    def _build_semantic_prompt(self, sparse_name: str, sparse_desc: str, sample_entities: List[str]) -> str:
        """æž„é€ è¯­ä¹‰åˆ†æž�çš„LLMæ��ç¤º"""
        entity_info = []
        for entity in sample_entities:
            name = self.data_loader.get_entity_name(entity)
            desc = self.data_loader.get_entity_description(entity)
            entity_info.append(f"- {entity}: {name} ({desc[:100]}...)" if len(
                desc) > 100 else f"- {entity}: {name} ({desc})")

        prompt = SEMANTIC_ANALYSIS.format(
            sparse_entity=self.sparse_entity,
            sparse_name=sparse_name,
            sparse_desc=sparse_desc,
            relation=self.relation,
            position=self.position,
            candidate_entities=chr(10).join(entity_info)
        )
        return prompt

    def _parse_semantic_response(self, response: str, sample_entities: List[str]) -> List[Set[str]]:
        """è§£æž�LLMçš„è¯­ä¹‰åˆ†ç»„å“�åº”"""
        groups = []

        try:
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith('Group'):
                    # æ��å�–ç»„ä¸­çš„å®žä½“
                    if ':' in line:
                        entities_str = line.split(':', 1)[1].strip()
                        entities = [e.strip() for e in entities_str.split(',')]
                        # å�ªä¿�ç•™æœ‰æ•ˆçš„å®žä½“ID
                        valid_entities = set(
                            e for e in entities if e in sample_entities)
                        if valid_entities:
                            groups.append(valid_entities)

        except Exception as e:
            self.logger.warning(f"Failed to parse semantic response: {e}")

        # å¦‚æžœè§£æž�å¤±è´¥æˆ–æ²¡æœ‰æœ‰æ•ˆåˆ†ç»„ï¼Œéš�æœºåˆ†ç»„
        if not groups:
            groups = self._random_grouping(sample_entities)

        return groups

    def _random_grouping(self, entities: List[str]) -> List[Set[str]]:
        """éš�æœºåˆ†ç»„"""
        random.shuffle(entities)
        group_size = len(entities) // 2

        group1 = set(entities[:group_size])
        group2 = set(entities[group_size:])

        return [group1, group2]

    def _random_split(self):
        """éš�æœºåˆ’åˆ†å€™é€‰å®žä½“"""
        candidates_list = list(self.candidate_entities)
        random.shuffle(candidates_list)

        mid_point = len(candidates_list) // 2
        chunk1 = set(candidates_list[:mid_point])
        chunk2 = set(candidates_list[mid_point:])

        for i, chunk in enumerate([chunk1, chunk2]):
            if chunk:
                self.children.add(LeafNode(
                    sparse_entity=self.sparse_entity,
                    position=self.position,
                    relation=self.relation,
                    candidate_entities=chunk,
                    data_loader=self.data_loader,
                    triplet_discriminator=self.triplet_discriminator,
                    leaf_threshold=self.leaf_threshold,
                    parent=self,
                    node_type=f"random_split_{i}"
                ))


class LeafNode(SearchNode):
    """å�¶å­�èŠ‚ç‚¹ï¼šå€™é€‰å®žä½“æ•°é‡�è¾ƒå°‘ï¼Œå�¯ä»¥ç›´æŽ¥è¿›è¡Œåˆ†ç±»å™¨è¯„ä¼°"""

    def __init__(self, *args, node_type: str = "leaf", **kwargs):
        super().__init__(*args, **kwargs)
        self.node_type = node_type

    def find_children(self) -> Set[SearchNode]:
        return set()  # å�¶å­�èŠ‚ç‚¹æ²¡æœ‰å­�èŠ‚ç‚¹

    def find_random_child(self) -> Optional[SearchNode]:
        return None  # å�¶å­�èŠ‚ç‚¹æ²¡æœ‰å­�èŠ‚ç‚¹

    def expand(self):
        """å�¶å­�èŠ‚ç‚¹ä¸�éœ€è¦�æ‰©å±•"""
        self.children = set()

    def is_terminal(self) -> bool:
        """å�¶å­�èŠ‚ç‚¹å§‹ç»ˆæ˜¯ç»ˆç«¯èŠ‚ç‚¹"""
        return True
