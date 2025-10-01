import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import argparse
import os


def setup_npu():
    if 'NPU_VISIBLE_DEVICES' in os.environ:
    device_id = int(os.environ['NPU_VISIBLE_DEVICES'])
    torch.npu.set_device(f'npu:{device_id}')
    print(f"Using NPU device: {device_id}")
    else:
    # 如果没有设置环境变量，可以默认使用npu:0
    torch.npu.set_device('npu:0')
    print("NPU_VISIBLE_DEVICES not set, using default NPU device: 0")

def main(args):
    print("Command-line arguments:")
    for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
    # Setup NPU device
    setup_npu()

    # Ensure output directory exists
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # 1. Dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=args.in_path,
        batch_size=args.batch_size,
        threads=args.threads,
        sampling_mode="cross",
        bern_flag=args.bern,
        filter_flag=1,
        neg_ent=args.neg_ent,
        neg_rel=args.neg_rel
    )

    # 2. Define the model
    model_map = {
        'RotatE': RotatE
    }

    if args.model in model_map:
        if args.model == 'RotatE':
            kge_model = model_map[args.model](
                ent_tot=train_dataloader.get_ent_tot(),
                rel_tot=train_dataloader.get_rel_tot(),
                dim=args.dimension,
                margin=args.margin,
                epsilon=2.0,  # Default epsilon for RotatE
            )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # 3. Define the loss function and training strategy
    model = NegativeSampling(
        model=kge_model,
        loss=SigmoidLoss(adv_temperature=2),
        batch_size=train_dataloader.get_batch_size()
    )

    # 4. Train the model
    print(f"\nStarting training for {args.model} model...")
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=args.epochs,
        alpha=args.learning_rate,
        use_gpu=True,  # Set to True to use NPU/GPU
        opt_method=args.optimizer
    )
    trainer.run()

    # 5. Save the model
    checkpoint_path = os.path.join(
        args.out_path, f"{args.model}_{args.time_stamp}.pth")
    kge_model.save_checkpoint(checkpoint_path)
    print(f"Model training completed and saved to {checkpoint_path}")

    # 6. (Optional) Evaluation after training
    if args.test_link or args.test_triple:
        print("\nLoading checkpoint for evaluation...")
        kge_model.load_checkpoint(checkpoint_path)

        if args.test_link:
            print("\nStarting link prediction evaluation...")
            test_dataloader_link = TestDataLoader(args.in_path, "link")
            tester_link = Tester(
                model=kge_model, data_loader=test_dataloader_link, use_gpu=True)
            tester_link.run_link_prediction(type_constrain=False)

        if args.test_triple:
            print("\nStarting triple classification evaluation...")
            test_dataloader_triple = TestDataLoader(args.in_path, "triple")
            tester_triple = Tester(
                model=kge_model, data_loader=test_dataloader_triple, use_gpu=True)
            tester_triple.run_triple_classification()


if name == "main":
parser = argparse.ArgumentParser(
description="Train KGE with OpenKE (PyTorch)")
# Paths
parser.add_argument('--in_path', type=str,
                    default='./benchmark/FB15K-237/', help='Input dataset path')
parser.add_argument('--out_path', type=str, default='./res/',
                    help='Output directory for models and results')
parser.add_argument('--root_dir', type=str, default='./',
                    help='Root directory for the project')

# Model
parser.add_argument('--model', type=str, default='RotatE', choices=[
                    'TransE', 'TransH', 'TransR', 'TransD', 'SimplE', 'RotatE', 'RESCAL', 'DistMult', 'ComplEx', 'Analogy'], help='KGE model to train')
parser.add_argument('--dimension', type=int,
                    default=512, help='Embedding dimension')

# Training Hyperparameters
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int,
                    default=12, help='Batch size')
parser.add_argument('--learning_rate', type=float,
                    default=1e-4, help='Learning rate (alpha)')
parser.add_argument('--margin', type=float, default=6.0,
                    help='Margin for margin-based loss or models')
parser.add_argument('--optimizer', type=str, default='Adam',
                    choices=['SGD', 'Adam', 'Adagrad', 'RMSprop'], help='Optimizer')
parser.add_argument('--bern', type=int, default=0,
                    choices=[0, 1], help='Negative sampling strategy (0: unif, 1: bern)')
parser.add_argument('--neg_ent', type=int, default=32,
                    help='Number of negative entities to sample')
parser.add_argument('--neg_rel', type=int, default=0,
                    help='Number of negative relations to sample')

# System
parser.add_argument('--threads', type=int, default=1,
                    help='Number of worker threads')
parser.add_argument('--time_stamp', type=str,
                    default='timestamp', help='Timestamp for output files')

# Evaluation
parser.add_argument('--test_link', action='store_true',
                    help='Perform link prediction after training')
parser.add_argument('--test_triple', action='store_true',
                    help='Perform triple classification after training')

args = parser.parse_args()

# lazy import
import sys
sys.path.append(args.root_dir)
from openke.data import TrainDataLoader, TestDataLoader
from openke.module.strategy import NegativeSampling
from openke.module.loss import SigmoidLoss
from openke.module.model import RotatE
from openke.config import Trainer, Tester

main(args)
