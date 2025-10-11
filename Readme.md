# KoGReD

monitor checkpoint:

```bash
cdko && ackopa && python MCTS/process_checkpoints.py && mv MCTS/output/fb15k-237n/checkpoints/temp.txt data/FB15K-237N/auxiliary_triples.txt && acko && cdko && python data/data_preview.py

cdko && for i in {0..2}; do tail -n 4 MCTS/output/fb15k-237n/checkpoints/checkpoint_rank_$i.json; done
```

### Dependencies
- Dependencies can be installed using `requirements.txt`.

### Dataset:

- We use FB15k-237N and CoDEx-S dataset for knowledge graph reasoning.
- FB15k-237N and CoDEx-S are included in the `data` directory.

### Run:

1. Install all the requirements from `requirements.txt.`

2. Execute `./data/build_auxiliary_triples.py` for construct auxiliary triples for datasets

3. Move into floder `LLM_Discriminator`, run `./scripts/finetune_fb15k237n.sh` for finetune the LLM to obtain finetuned LLM as a triple discriminator for filtering auxiliary triples. Due to the size of the data, you need to download and unzip the data file data.zip from ([this link](https://drive.google.com/file/d/1J1Ioi23jTMaBkBDYzfIy2MAZYMUIjFWW/view)) and put them in the `./LLM_Discriminator/scripts/data/`

4. Feed the auxiliary triples build in step 2 into finetuned LLM discriminator by run `discriminator.py`

5. Put the auxiliary triples filtered by the discriminator under the dataset folder `./data/FB15K-237N` named `auxiliary_triples.txt`. Run `python run.py -score_func conve -opn corr -data fb15k-N -adapt_aggr 1 -loss_delta 0.002`
