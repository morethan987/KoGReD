import argparse
import torch
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util

"""
generate similarity matrix of entities
"""
def similarity_matrix(dataset):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.max_seq_length = 512
    description_text = []
    with open("./{}/entity2des.txt".format(dataset), "r") as file:
        for line in file:
            description = line.split("\t")[-1]
            description_text.append(description)
    embeddings = model.encode(description_text, convert_to_tensor=True) 

    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embeddings, embeddings)
    # print(cosine_scores[0])
    return cosine_scores

def topk_matrix(similarity_numpy, dataset, topk: int=4):
    top_k = topk
    sorted_indices = np.argsort(similarity_numpy, axis=1)[:, ::-1]
    top_k_scores = similarity_numpy[np.arange(similarity_numpy.shape[0])[:, np.newaxis], sorted_indices[:, :top_k]]
    top_k_indices = sorted_indices[:, :top_k]
    torch.save(top_k_indices, "./{}/top_{}_indices.pt".format(dataset, topk))
    return top_k_indices

def replace_entity(dataset, triple, top_k_indices, few_type="head"):
    ent_list = []
    with open("./{}/entity2des.txt".format(dataset), "r") as file:
        for line in file:
            entity = line.split("\t")[0]
            ent_list.append(entity)
    replace_list = []
    head = triple[0]
    rel = triple[1]
    tail = triple[2]
    if few_type=="head":
        tail_index = ent_list.index(tail)
        for i in range(1, 2): # choose top 1 most similar entity to generate auxiliary triples
            replace_ent_id = top_k_indices[tail_index][i]
            replace_list.append([head, rel, ent_list[replace_ent_id]])
    if few_type=="tail":
        head_index = ent_list.index(head)
        for i in range(1, 2):
            replace_ent_id = top_k_indices[head_index][i]
            replace_list.append([ent_list[replace_ent_id], rel, tail])
    return replace_list

"""
"""
def sparse_entity(dataset, sparse_threshold: float=0.009):
    count = {}  
    with open("./{}/train.txt".format(dataset), "r") as file:
        triple_num = len(list(file))
        for line in file:
            triple = line.strip().split('\t')
            head = triple[0]
            tail = triple[-1]
            rel = triple[1]
            if head in count:
                count[head] += 1
            else:
                count[head] = 1
            if tail in count:
                count[tail] += 1
            else:
                count[tail] = 1
    entity_freq = {}
    sparse_ent = []
    for entity, num in count.items():
        frequency = round(num / (triple_num * 2) * 100, 4)
        entity_freq[entity] = frequency
        if frequency < sparse_threshold:
            sparse_ent.append(entity)
    return sparse_ent

def build_auxiliary_triples(dataset, top_k_indices, sparse_ent):
    auxiliary_triples = []
    with open("./{}/train.txt".format(dataset), "r") as file:
        for line in file:
            triple = line.strip().split("\t")
            head = triple[0]
            tail = triple[2]
            if head in sparse_ent:
                replace_list = replace_entity(dataset, triple, top_k_indices, few_type="head")
                auxiliary_triples.extend(replace_list)
            if tail in sparse_ent:
                replace_list = replace_entity(dataset, triple, top_k_indices, few_type="tail")
                auxiliary_triples.extend(replace_list)

    with open("./{}/auxiliary_triples_raw.txt", "w") as file:
        for i in auxiliary_triples:
            file.write("{}\t{}\t{}\n".format(i[0], i[1], i[2]))

def main():
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dataset',	default='FB15K-237N', help='Select dataset name for constructing auxiliary triples.')
    args = parser.parse_args()
    matrix = similarity_matrix(args.dataset)
    similarity_numpy = matrix.cpu().numpy()
    top_k_indices = topk_matrix(similarity_numpy, args.dataset)
    sparse_entities = sparse_entity(args.dataset)
    build_auxiliary_triples(args.dataset, top_k_indices, sparse_entities)

if __name__ == '__main__':
    main()
