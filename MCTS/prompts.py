CHECK_RELATION_DIRECTION = """
You are an expert in Knowledge Graph schema.Your task is to determine whether a given entity can validly appear as the head (subject) or tail (object) of a relation.

Example1:
Relation: /tv/tv_writer/tv_programs./tv/tv_program_writer_relationship/tv_program
Position: head
Entity: Paris
Output: False

Example2:
Relation: /location/country/capital
Position: tail
Entity: Beijing
Output: True

Actual task:
Relation: {relation}
Position: {position}
Entity: {entity}
Output:
"""

SEMANTIC_ANALYSIS = """
Given a sparse entity and a relation, please group the following candidate entities into 2-3 semantic groups based on their relevance and semantic similarity.

Sparse Entity: {sparse_entity} - {sparse_name}
Description: {sparse_desc}
Relation: {relation}
Position: {osition}

Candidate Entities:
{candidate_entities}

Please group these entities into 2-3 groups and respond in the following format:
Group 1: entity1, entity2, entity3
Group 2: entity4, entity5, entity6
Group 3: entity7, entity8, entity9

Base your grouping on semantic similarity and relevance to the sparse entity and relation.
"""
