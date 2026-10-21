import time
import argparse
import torch
import os
from pathlib import Path
import datetime
import yaml
from omegaconf import OmegaConf
import hydra
from hydra.experimental import compose
from hydra import initialize_config_dir


from pytorch_lightning import seed_everything
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from eval_utils import lattices_to_params_shape

def generator(
    loader,
    model,
    num_evals,
    step_lr=5,
    infer_timesteps=200,
    local_rank=0,
    world_size=1,
    diffusion=False,
):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    gt_frac_coords = []
    gt_atom_types = []
    gt_lengths = []
    gt_angles = []
    gt_num_atoms = []

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch = batch.to(f"cuda:{local_rank}")
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []

        
        gt_frac_coords.append(batch.frac_coords.cpu())
        gt_atom_types.append(batch.atom_types.cpu())
        gt_lengths.append(batch.lengths.cpu())
        gt_angles.append(batch.angles.cpu())
        gt_num_atoms.append(batch.num_atoms)

        for eval_idx in range(num_evals):
            print(f"Rank {local_rank}: batch {idx+1} / {len(loader)}, sample {eval_idx+1} / {num_evals}")
            if diffusion:
                outputs, _ = model.module.sample(batch, step_lr=step_lr)
            else:
                outputs = model.module.sample(batch, step_lr=step_lr, infer_timesteps=infer_timesteps)
            batch_frac_coords.append(outputs["frac_coords"].detach().cpu())
            batch_num_atoms.append(outputs["num_atoms"].detach().cpu())
            batch_atom_types.append(outputs["atom_types"].detach().cpu())
            batch_lattices.append(outputs["lattices"].detach().cpu())

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1) if frac_coords else torch.tensor([])
    num_atoms = torch.cat(num_atoms, dim=1) if num_atoms else torch.tensor([])
    atom_types = torch.cat(atom_types, dim=1) if atom_types else torch.tensor([])
    lattices = torch.cat(lattices, dim=1) if lattices else torch.tensor([])
    lengths, angles = lattices_to_params_shape(lattices) if lattices.numel() > 0 else (torch.tensor([]), torch.tensor([]))


    gt_frac_coords = torch.cat(gt_frac_coords, dim=0)
    gt_atom_types = torch.cat(gt_atom_types, dim=0)
    gt_lengths = torch.cat(gt_lengths, dim=0)
    gt_angles = torch.cat(gt_angles, dim=0)
    gt_num_atoms = torch.tensor(gt_num_atoms)

    device = f"cuda:{local_rank}"

    # Gather results from all ranks
    frac_coords = frac_coords.to(device)
    num_atoms = num_atoms.to(device)
    atom_types = atom_types.to(device)
    lattices = lattices.to(device)
    lengths = lengths.to(device)
    angles = angles.to(device)

    all_frac_coords = [None] * world_size
    all_num_atoms = [None] * world_size
    all_atom_types = [None] * world_size
    all_lattices = [None] * world_size
    all_lengths = [None] * world_size
    all_angles = [None] * world_size
    all_gt_frac_coords = [None] * world_size
    all_gt_atom_types = [None] * world_size
    all_gt_lengths = [None] * world_size
    all_gt_angles = [None] * world_size
    all_gt_num_atoms = [None] * world_size

    dist.all_gather_object(all_frac_coords, frac_coords.cpu())
    dist.all_gather_object(all_num_atoms, num_atoms.cpu())
    dist.all_gather_object(all_atom_types, atom_types.cpu())
    dist.all_gather_object(all_lattices, lattices.cpu())
    dist.all_gather_object(all_lengths, lengths.cpu())
    dist.all_gather_object(all_angles, angles.cpu())
    dist.all_gather_object(all_gt_frac_coords, gt_frac_coords.cpu())
    dist.all_gather_object(all_gt_atom_types, gt_atom_types.cpu())
    dist.all_gather_object(all_gt_lengths, gt_lengths.cpu())
    dist.all_gather_object(all_gt_angles, gt_angles.cpu())
    dist.all_gather_object(all_gt_num_atoms, gt_num_atoms.cpu())

    if local_rank == 0:
        frac_coords = torch.cat(all_frac_coords, dim=1)
        num_atoms = torch.cat(all_num_atoms, dim=1)
        atom_types = torch.cat(all_atom_types, dim=1)
        lattices = torch.cat(all_lattices, dim=1)
        lengths = torch.cat(all_lengths, dim=1)
        angles = torch.cat(all_angles, dim=1)
        gt_frac_coords = torch.cat(all_gt_frac_coords, dim=0)
        gt_atom_types = torch.cat(all_gt_atom_types, dim=0)
        gt_lengths = torch.cat(all_gt_lengths, dim=0)
        gt_angles = torch.cat(all_gt_angles, dim=0)
        gt_num_atoms = torch.cat(all_gt_num_atoms, dim=0)

    return dict(
        frac_coords=frac_coords,
        num_atoms=num_atoms,
        atom_types=atom_types,
        lattices=lattices,
        lengths=lengths,
        angles=angles,
        gt_frac_coords=gt_frac_coords,
        gt_atom_types=gt_atom_types,
        gt_lengths=gt_lengths,
        gt_angles=gt_angles,
        gt_num_atoms=gt_num_atoms,
    )


def load_model_ckpt(model_path, ckpt_path, load_data=False, testing=True, switch=True, test_path=None):
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
        model = hydra.utils.get_class(
            cfg.model._target_,
        )
        model = model.load_from_checkpoint(ckpt_path, strict=True)

        if load_data:
            cfg.data.datamodule.batch_size.test = 1
            cfg.data.switch = switch
            cfg.data.datamodule.datasets.test[0].switch = switch
            if test_path is not None:
                cfg.data.datamodule.datasets.test[0].path = test_path
            datamodule = hydra.utils.instantiate(
                cfg.data.datamodule, _recursive_=False, scaler_path=model_path
            )
            if testing:
                datamodule.setup('test')
                test_loader = datamodule.test_dataloader()[0]
            else:
                datamodule.setup()
                train_loader = datamodule.train_dataloader(shuffle=False)
                val_loader = datamodule.val_dataloader()[0]
                test_loader = (train_loader, val_loader)
        else:
            test_loader = None

    return model, test_loader, cfg


def save_cfg_to_yaml(cfg, output_path):
    if isinstance(cfg, dict):
        cfg_dict = cfg
    else:
        try:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # Use OmegaConf to convert to a standard dictionary
        except Exception as e:
            print(f"Failed to convert cfg: {e}")
            cfg_dict = {}

    with open(output_path, 'w') as file:
        yaml.dump(cfg_dict, file, default_flow_style=False, sort_keys=False)
    print(f"Configuration saved to {output_path}")


def main(args):
    dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=7200))  # 2h
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    seed_everything(42 + local_rank)

    model_path = Path(os.path.dirname(args.generator_ckpt_path))
    save_path = Path(args.save_path)
    if local_rank == 0:
        os.makedirs(save_path, exist_ok=True)

    unified_model, test_loader, cfg = load_model_ckpt(
        model_path, args.generator_ckpt_path, load_data=True, switch=args.noswitch, test_path=args.test_path
    )
    if local_rank == 0:
        cfg_output_path = os.path.join(save_path, "config.yaml")
        save_cfg_to_yaml(cfg, cfg_output_path)

    if args.ae:
        model = unified_model.self_module
    else:
        model = unified_model.cross_module
    model.eval()

    dataset = test_loader.dataset  # Extract the underlying dataset from the loaded DataLoader
    if args.end_idx != -1:
        dataset = [dataset[i] for i in range(args.begin_idx, args.end_idx)]
    collate_fn = test_loader.collate_fn

    test_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers if hasattr(cfg.data, 'num_workers') else 0,
    )

    torch.cuda.set_device(local_rank)
    model.to(f"cuda:{local_rank}")
    model = DDP(model, device_ids=[local_rank])

    print(f"Rank {local_rank}: Evaluate the generator model.")

    start_time = time.time()
    out_dict = generator(
        test_loader,
        model,
        num_evals=args.num_evals,
        step_lr=args.step_lr,
        infer_timesteps=args.infer_timesteps,
        local_rank=local_rank,
        world_size=world_size,
        diffusion=args.diffusion,
    )

    if local_rank == 0:
        generator_out_name = f"eval_generator_{args.label}_number-eval{args.num_evals}_b{args.begin_idx}_e{args.end_idx}.pt"
        torch.save(
            {
                "eval_setting": args,
                "frac_coords": out_dict["frac_coords"],
                "num_atoms": out_dict["num_atoms"],
                "atom_types": out_dict["atom_types"],
                "lattices": out_dict["lattices"],
                "lengths": out_dict["lengths"],
                "angles": out_dict["angles"],
                "gt_frac_coords": out_dict["gt_frac_coords"],
                "gt_num_atoms": out_dict["gt_num_atoms"],
                "gt_atom_types": out_dict["gt_atom_types"],
                "gt_lengths": out_dict["gt_lengths"],
                "gt_angles": out_dict["gt_angles"],
                "time": time.time() - start_time,
            },
            save_path / generator_out_name,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_ckpt_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--step_lr", default=5, type=float)
    parser.add_argument("--infer_timesteps", default=200, type=int)
    parser.add_argument("--num_evals", default=1, type=int)
    parser.add_argument("--label", default="")
    parser.add_argument("--begin_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=-1, type=int)
    parser.add_argument("--noswitch", action="store_false", help="whether to switch views") # default true
    parser.add_argument("--test_path", default=None, type=str)
    parser.add_argument("--ae", action="store_true", help="test if it is an self-generator") # default false
    parser.add_argument("--diffusion", action="store_true", help="test if it uses a diffusion model as the generator") # default false
    args = parser.parse_args()
    main(args)
