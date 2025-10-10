import json


def process_checkpoints(folder: str, num_checkpoints: int = 3):
    results = set()
    for i in range(num_checkpoints):
        with open(f"{folder}/checkpoint_rank_{i}.json", "r") as f:
            data = json.load(f)

        triplets = data.get("discovered_triplets", [])
        results.update(tuple(triplet) for triplet in triplets)

    with open(f"{folder}/temp.txt", "w") as f:
        for triplet in results:
            f.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")

    print(f"Total unique triplets: {len(results)}")


if __name__ == "__main__":
    # cdko && ackopa && python MCTS/process_checkpoints.py
    # mv MCTS/output/fb15k-237n/checkpoints/temp.txt data/FB15K-237N/auxiliary_triples.txt
    # mv MCTS/output/codex-s/checkpoints/temp.txt data/CoDEx-S/auxiliary_triples.txt
    # process_checkpoints("MCTS/output/fb15k-237n/checkpoints")
    process_checkpoints("MCTS/output/fb15k-237n/checkpoints")
