# KoGReD

## Dependencies

- Dependencies can be installed using `requirements.txt`.

## Dataset:

- We use FB15k-237N and CoDEx-S dataset for knowledge graph reasoning.
- FB15k-237N and CoDEx-S are included in the `data` directory.

## Run:

1. Install all the requirements from `requirements.txt`

2. Execute `scripts/run_embedding.sh` to build the `entity2embedding.pth` file in data folder.

3. Run `scripts/run_finetune_fb15k237n.sh` for finetune the LLM to obtain finetuned LLM as a triple discriminator for filtering auxiliary triples. Due to the size of the data, you need to download and unzip the data file data.zip from ([this link](https://drive.google.com/file/d/1J1Ioi23jTMaBkBDYzfIy2MAZYMUIjFWW/view)) and put them in the `LLM_Discriminator/data/`

4. Start the MCTS by running `scripts/run_mcts.sh`, the output would be put at `MCTS/output/fb15k-237n/discovered_triplets.txt` for example. The exact path depends on your dataset. Then move the `discovered_triplets.txt` into the dataset folder, the same as the `train.txt`, and rename it as `auxiliary_triples.txt`

5. Run `scripts/train_loss_restrain_kge.sh` to build the final kge model, which is evaluated on the `test.txt` automatically and saved into the `loss_restraint_KGE_model/output` folder. The test result can be found in the trainning logs.

## Other

In the `data_preview.py` are many useful tools you may need.
