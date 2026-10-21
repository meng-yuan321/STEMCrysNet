import torch
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from collections import Counter
import math
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

def reduce_to_ratio(arr):
    counter = Counter(arr)
    counts = list(counter.values())
    # 求所有计数的最大公约数
    def gcd_list(lst):
        from functools import reduce
        return reduce(math.gcd, lst)
    factor = gcd_list(counts)
    # 用最大公约数约分
    reduced = []
    for k, v in counter.items():
        reduced.extend([k] * (v // factor))
    return tuple(sorted(reduced))

def main(args):
    csv_file_path = 'data/mp_50/mp_50_7system_20250710.csv'
    df = pd.read_csv(csv_file_path)
    cifs = df["CIF"].values
    material_id = df["Material ID"].values

    atom_data = torch.load(args.atom_path)
    stem_feat = []
    atom_feat = []
    atom_id_list = []
    stem_id_list = []
    atom_type_list = []
    stem_type_list = []
    atom_pos_list = []
    # for i in [0,1,3,5]:
    for i in range(len(atom_data['atom_feat'])):
        atom_id_list.append(atom_data['ids'][i].squeeze(0))
        atom_feat.append(atom_data['atom_feat'][i].squeeze(0))
        atom_type_list.append(atom_data['atom_types'][i].squeeze(0))

    for i in range(len(atom_data['stem_feat'])):
        stem_id_list.append(atom_data['ids'][i].squeeze(0))
        stem_feat.append(atom_data['stem_feat'][i].squeeze(0))
        stem_type_list.append(atom_data['atom_types'][i].squeeze(0))
        # atom_pos_list.append(data['frac_coords'][i].squeeze(0))
    stem_tensor = torch.cat(stem_feat, dim=0) # [len(test_loader), 512]
    atom_tensor = torch.cat(atom_feat, dim=0) # [len(test_loader), 512]
    # per_stem = stem_tensor @ atom_tensor.T # [len(test_loader), len(test_loader)]
    # print(atom_type_list)
    # Compute similarity only between matching atom types
    type_to_indices = defaultdict(list)
    for j in range(len(atom_type_list)):
        if args.option == 0:
            # Elemental Query
            arr = set(atom_type_list[j].tolist())
        elif args.option == 1:
            # Stoichiometric Query
            arr = reduce_to_ratio(atom_type_list[j].tolist())
        elif args.option == 2:
            # Unit Cell Content Query
            arr = atom_type_list[j].tolist()
        else:
            # Unconstrained Query
            # don't need to fill type_to_indices, later match all indices
            continue
        key = tuple(sorted(arr))
        # print(key)
        type_to_indices[key].append(j)
    
    # if Unconstrained Query, mark type_to_indices as None
    if args.option == 3:
        type_to_indices = None
    
    print("first stage done")

    # Initialize similarity score matrix
    # per_stem = torch.zeros(stem_tensor.shape[0], atom_tensor.shape[0])
    per_stem = torch.full((stem_tensor.shape[0], atom_tensor.shape[0]), -1.1) #相似度的范围是-1~1，因此应该初始化为最小值-1，而不是0
    for i in range(stem_tensor.shape[0]):
        # print(f' {i} / {stem_tensor.shape[0]}', end='\r')
        if args.option == 0:
            # Elemental Query
            arr = set(stem_type_list[i].tolist())
        elif args.option == 1:
            # Stoichiometric Query
            arr = reduce_to_ratio(stem_type_list[i].tolist())
        elif args.option == 2:
            # Unit Cell Content Query
            arr = stem_type_list[i].tolist()
        else:
            # Unconstrained Query
            current_type = None

        if args.option != 3:
            current_type = tuple(sorted(arr))
            # get all matched j indices
            matched_js = type_to_indices.get(current_type, [])
        else:
            matched_js = list(range(len(atom_type_list)))

        # get all matched j indices
        # matched_js = type_to_indices.get(current_type, [])
        if matched_js:
            # batch compute dot product (vectorized)
            # per_stem[i, matched_js] = stem_tensor[i] @ atom_tensor[matched_js].T
            atom_matched = atom_tensor[matched_js]  # [n, 512]
            # print(atom_matched.shape)
            # normalize
            stem_vec = F.normalize(stem_tensor[i].unsqueeze(0), dim=1)  # [1, 512]
            # print(stem_vec.shape)
            atom_matched_norm = F.normalize(atom_matched, dim=1)        # [n, 512]
            # print(atom_matched_norm.shape)
            # compute cosine similarity
            sim = torch.mm(stem_vec, atom_matched_norm.T).squeeze(0)     # [n]
            # print(sim)
            per_stem[i, matched_js] = sim
            

    # after computation, release intermediate variables
    # del stem_data
    del atom_data
    del stem_feat
    del atom_feat
    del atom_type_list
    del stem_tensor
    del atom_tensor
    print("second stage done")

    
    def topK_rate(k, save_correct_mp_ids=False, save_wrong_mp_ids=False):
        topk_idx = per_stem.topk(k, dim=1)[1] # [len(test_loader), K]
        topk_hit = 0
        correct_indices = []
        wrong_indices = []

        # helper to resolve cif by mp id or by index
        def resolve_cif(mp_val):
            # try to coerce torch/numpy types to python scalar/string
            try:
                if hasattr(mp_val, "item"):
                    mp_val = mp_val.item()
            except Exception:
                pass
            # if it's bytes, decode
            if isinstance(mp_val, (bytes, bytearray)):
                try:
                    mp_val = mp_val.decode()
                except Exception:
                    mp_val = str(mp_val)
            # try direct lookup in material_id array (string match)
            if isinstance(mp_val, str):
                matches = np.where(material_id == mp_val)[0]
                if matches.size > 0:
                    return cifs[matches[0]]
            # if it's integer index into csv rows
            try:
                idx_int = int(mp_val)
                if 0 <= idx_int < len(cifs):
                    return cifs[idx_int]
            except Exception:
                pass
            return None

        matcher = StructureMatcher()

        for idx, t in enumerate(topk_idx):
            gt_mp = stem_id_list[idx]
            matched = False

            # resolve ground-truth cif once
            cif_gt = resolve_cif(gt_mp)
            struct_gt = None
            if cif_gt is not None:
                try:
                    struct_gt = Structure.from_str(cif_gt, fmt="cif")
                except Exception:
                    struct_gt = None

            for j in t.tolist():
                pred_mp = atom_id_list[j]
                # quick equality check first
                try:
                    if hasattr(gt_mp, "item"):
                        gt_val = gt_mp.item()
                    else:
                        gt_val = gt_mp
                    if hasattr(pred_mp, "item"):
                        pred_val = pred_mp.item()
                    else:
                        pred_val = pred_mp
                except Exception:
                    gt_val = gt_mp
                    pred_val = pred_mp

                if gt_val == pred_val:
                    topk_hit += 1
                    correct_indices.append(idx)
                    matched = True
                    break

                # otherwise try structure matching
                cif_pred = resolve_cif(pred_mp)
                if cif_gt is None or cif_pred is None:
                    continue
                try:
                    struct_pred = Structure.from_str(cif_pred, fmt="cif")
                except Exception:
                    continue

                try:
                    if struct_gt is None:
                        struct_gt = Structure.from_str(cif_gt, fmt="cif")
                except Exception:
                    struct_gt = None

                if struct_gt is None:
                    continue

                try:
                    if matcher.fit(struct_gt, struct_pred):
                        topk_hit += 1
                        correct_indices.append(idx)
                        matched = True
                        break
                except Exception:
                    # fallback: not matched
                    pass

            if not matched:
                wrong_indices.append(idx)
                # print predictions for debugging
                # pred_ids = [atom_id_list[i].item() if hasattr(atom_id_list[i], "item") else atom_id_list[i] for i in t.tolist()]
                # print(f"wrong sample index: {idx}, true id: {gt_val}, predicted id: {pred_ids}")

        topk_rate = topk_hit / per_stem.shape[0]
        if save_correct_mp_ids:
            correct_ids = [stem_id_list[i].cpu().numpy() if hasattr(stem_id_list[i], "cpu") else stem_id_list[i] for i in correct_indices]
            correct_ids = np.array(correct_ids)
            np.save(os.path.join(save_dir, f"top{k}_correct_ids.npy"), correct_ids)
        if save_wrong_mp_ids:
            wrong_ids = [stem_id_list[i].cpu().numpy() if hasattr(stem_id_list[i], "cpu") else stem_id_list[i] for i in wrong_indices]
            wrong_ids = np.array(wrong_ids)
            np.save(os.path.join(save_dir, f"top{k}_wrong_ids.npy"), wrong_ids)
        return topk_rate
    
    save = True
    save_dir = None
    if save:
        save_dir = os.path.join(os.path.dirname(args.atom_path), f"option{args.option}")
        os.makedirs(save_dir)
    top1_rate = topK_rate(k=1, save_correct_mp_ids=save, save_wrong_mp_ids=save)
    print("top 1 rate: ",top1_rate)    
    top3_rate = topK_rate(k=3, save_correct_mp_ids=save, save_wrong_mp_ids=save)
    print("top 3 rate: ",top3_rate)
    top5_rate = topK_rate(k=5, save_correct_mp_ids=save, save_wrong_mp_ids=save)
    print("top 5 rate: ",top5_rate)
    top10_rate = topK_rate(k=10, save_correct_mp_ids=save, save_wrong_mp_ids=save)
    print("top 10 rate: ",top10_rate)

    with open(os.path.join(save_dir, f"rate_option{args.option}.txt"), "w") as f:
        f.write(f"top 1 rate: {top1_rate}\ntop 3 rate: {top3_rate}\ntop 5 rate: {top5_rate}\ntop 10 rate: {top10_rate}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=int, default=0, help="0: Elemental Query, 1: Stoichiometric Query, 2: Unit Cell Content Query, 3: Unconstrained Query")
    parser.add_argument('--atom_path', required=True)
    args = parser.parse_args()
    main(args)
