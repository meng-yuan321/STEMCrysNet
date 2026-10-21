from collections import Counter
import argparse
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map 
import time

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

import torch



from eval_utils import structure_validity, load_data, get_crystals_list

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

def center_structure(structure, vacuum=None, axis=(0, 1, 2), about=None):
    """
    Center atoms in unit cell.

    Centers the atoms in the unit cell, so there is the same
    amount of vacuum on all sides.

    vacuum: float (default: None)
        If specified adjust the amount of vacuum when centering.
        If vacuum=10.0 there will thus be 10 Angstrom of vacuum
        on each side.
    axis: int or sequence of ints
        Axis or axes to act on.  Default: Act on all axes.
    about: float or array (default: None)
        If specified, center the atoms about <about>.
        I.e., about=(0., 0., 0.) (or just "about=0.", interpreted
        identically), to center about the origin.
    """

    # Find the orientations of the faces of the unit cell
    cell = structure.lattice.matrix # base vectors
    dirs = np.zeros_like(cell)

    lengths = np.linalg.norm(cell, axis=1) # the length of each base vector
    for i in range(3):
        dirs[i] = np.cross(cell[i - 1], cell[i - 2])
        dirs[i] /= np.linalg.norm(dirs[i]) # the normal vector of the face
        if np.dot(dirs[i], cell[i]) < 0.0:
            dirs[i] *= -1

    if isinstance(axis, int):
        axes = (axis,)
    else:
        axes = axis

    # Now, decide how much each basis vector should be made longer
    pos = structure.cart_coords
    longer = np.zeros(3)
    shift = np.zeros(3)
    for i in axes:
        if len(pos):
            scalarprod = np.dot(pos, dirs[i])
            p0 = scalarprod.min()
            p1 = scalarprod.max()
        else:
            p0 = 0
            p1 = 0
        height = np.dot(cell[i], dirs[i])
        if vacuum is not None:
            lng = (p1 - p0 + 2 * vacuum) - height
        else:
            lng = 0.0  # Do not change unit cell size!
        top = lng + height - p1
        shf = 0.5 * (top - p0)
        cosphi = np.dot(cell[i], dirs[i]) / lengths[i]
        longer[i] = lng / cosphi
        shift[i] = shf / cosphi

    # Now, do it!
    translation = np.zeros(3)
    new_cell = cell.copy()
    for i in axes:
        nowlen = lengths[i]
        if vacuum is not None:
            new_cell[i] = cell[i] * (1 + longer[i] / nowlen)
        translation += shift[i] * cell[i] / nowlen    
        # We calculated translations using the completed cell,
        # so directions without cell vectors will have been centered
        # along a "fake" vector of length 1.
        # Therefore, we adjust by -0.5:
        if not any(new_cell[i]):
            translation[i] -= 0.5

    # Optionally, translate to center about a point in space.
    if about is not None:
        for n, vector in enumerate(new_cell):
            if n in axes:
                translation -= vector / 2.0
                translation[n] += about[n]
    structure.translate_sites(range(len(structure)), translation, frac_coords=False)
    
    
    # new fractional coordinates
    cart_coords = structure.cart_coords
    new_lattice = Lattice(new_cell)
    new_frac_coords = new_lattice.get_fractional_coords(cart_coords)

    # update the lattice and fractional coordinates
    structure.lattice = Lattice(new_cell)
    for i, site in enumerate(structure):
        structure.replace(i, site.species, new_frac_coords[i])

class Crystal(object):

    def __init__(self, crys_array_dict, i):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)

        self.get_structure(i)
        self.get_composition()
        self.get_validity()
        # for double-layer structures
        # if self.structure.num_sites == 6:
        #     self.remove_vacuum_and_center() 

    def get_structure(self, i):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'            
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
                # formula = self.structure.formula.replace(" ", "")
                # save_dir = 'evaluate_results/exp_cifs2/'
                # os.makedirs(save_dir, exist_ok=True)
                # self.structure.to(filename=f"{save_dir}/{i}_pred.cif")
                
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = True
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.struct_valid = True
        self.valid = self.comp_valid and self.struct_valid

    def remove_vacuum_and_center(self):
        coords = self.structure.frac_coords
        c_length = self.structure.lattice.c 
        # sort atoms by z coordinates
        sorted_coords = sorted(coords, key=lambda x: x[2])
        z_coords = [coord[2] for coord in sorted_coords]
        z_diff = [z_coords[i+1] - z_coords[i] for i in range(len(z_coords)-1)]
        max_vacuum_index = z_diff.index(max(z_diff))
        first_largest_z_diff = z_diff[max_vacuum_index] * c_length
        # case 1: vacuum region at the ends
        if sorted_coords[0][2] * c_length + (1 - sorted_coords[-1][2]) * c_length > 10:
            center_structure(self.structure, vacuum=first_largest_z_diff/2, axis=(2))
        
        else:     
            # case 2: vacuum region in the middle
            if first_largest_z_diff > 10:
                top = z_coords[max_vacuum_index] 
                translation = np.array([0, 0, -top-0.05])
                self.structure.translate_sites(range(len(self.structure.sites)), translation, frac_coords=True)
                sorted_z_diff = sorted(z_diff, reverse=True)
                second_largest_z_diff = sorted_z_diff[1]
                center_structure(self.structure, vacuum=second_largest_z_diff/2, axis=(2))



class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.3, angle_tol=5, ltol=0.1):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def get_match_rate_and_rms(self, dir):
        def process_one(pred, gt, is_valid, i):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0] 
                if rms_dist is None:
                    save_dir = os.path.join(dir,'error_cifs/')
                    os.makedirs(save_dir, exist_ok=True)
                    pred.structure.to(filename=f"{save_dir}/{i}_pred.cif")
                    gt.structure.to(filename=f"{save_dir}/{i}_gt.cif")
                else:
                    save_dir = os.path.join(dir,'correct_cifs/')
                    os.makedirs(save_dir, exist_ok=True)
                    pred.structure.to(filename=f"{save_dir}/{i}_pred.cif")
                    gt.structure.to(filename=f"{save_dir}/{i}_gt.cif")
                return rms_dist
            except Exception:
                return None
        validity = [c1.valid and c2.valid for c1,c2 in zip(self.preds, self.gts)]
        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(process_one(
                self.preds[i], self.gts[i], validity[i], i))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist,
                'rms_dists':rms_dists}     

    def get_metrics(self, dir):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms(dir))
        return metrics

def process_one(pred, gt, matcher):
    try:
        rms_dist = matcher.get_rms_dist(
            pred.structure, gt.structure)
        rms_dist = None if rms_dist is None else rms_dist[0]
        return rms_dist
    except Exception:
        return None

def process_batch(preds, gt, matcher):
    rms_dists = np.zeros(len(preds))
    for i in (range(len(preds))):
        rms_dists[i] = (process_one(
            preds[i], gt, matcher))
    if np.isnan(rms_dists).all():
        return rms_dists, None
    else:
        return rms_dists, np.nanmin(rms_dists)

class RecEvalBatch(object):
#0.7, 5, 0.2
    def __init__(self, pred_crys, gt_crys, rank=None, stol=0.3, angle_tol=5, ltol=0.1, bsz=10000, strict=True):
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol, attempt_supercell=True, comparator=ElementComparator())
        self.preds = pred_crys
        self.gts = gt_crys
        self.batch_size = min(len(self.preds), bsz)
        # self.rank = rank.cpu().numpy() if rank is not None else None
        self.rank = rank if rank is not None else None
        self.strict = strict

    def get_randomk_match_rate_and_rms(self, dists, k=1):
        all_dists = np.array(dists)[:k, :]
        rmse_list = []
        for i in range(len(all_dists[0])):
            if (np.isnan(all_dists[:, i])).all():
                pass
            else:
                rmse_list.append(np.nanmin(all_dists[:, i]))
        match_rate = len(rmse_list) / len(all_dists[0])
        mean_rms_dist = np.mean(rmse_list)
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}
    
    def get_match_rate_and_rms(self, dir):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                if self.strict:
                    if self.matcher.fit(pred.structure, gt.structure):
                        rms_dist = self.matcher.get_rms_dist(
                            pred.structure, gt.structure)
                        rms_dist = None if rms_dist is None else rms_dist[0]
                    else:
                        rms_dist = None
                else:
                    rms_dist = self.matcher.get_rms_dist(
                        pred.structure, gt.structure)
                    rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        rms_dists = []
        self.all_rms_dis = np.zeros((self.batch_size, len(self.gts)))
        for i in tqdm(range(len(self.preds[0]))): # materials
            tmp_rms_dists = []
            for j in range(self.batch_size): # sample times
                save_dir = os.path.join(dir,'pred_cifs_multi_eval/')
                os.makedirs(save_dir, exist_ok=True)
                self.gts[i].structure.to(filename=f"{save_dir}/{i}_gt.cif")
                self.preds[j][i].structure.to(filename=f"{save_dir}/{i}_pred_{j}.cif")
                
                rmsd = process_one(self.preds[j][i], self.gts[i], self.preds[j][i].valid)
                self.all_rms_dis[j][i] = rmsd
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
                    save_dir = os.path.join(dir,'correct_cifs_multi_eval/')
                    os.makedirs(save_dir, exist_ok=True)
                    self.gts[i].structure.to(filename=f"{save_dir}/{i}_gt.cif")
                    self.preds[j][i].structure.to(filename=f"{save_dir}/{i}_pred_{j}.cif")
            if len(tmp_rms_dists) == 0:
                rms_dists.append(None)
                save_dir = os.path.join(dir,'error_cifs_multi_eval/')
                os.makedirs(save_dir, exist_ok=True)
                self.gts[i].structure.to(filename=f"{save_dir}/{i}_gt.cif")
                for j in range(self.batch_size):
                    self.preds[j][i].structure.to(filename=f"{save_dir}/{i}_pred_{j}.cif")

            else:
                rms_dists.append(np.min(tmp_rms_dists))
                
        
        result = {}
        rms_dists = np.array(rms_dists)
        print(rms_dists.shape)
        none_indices = [i for i, d in enumerate(rms_dists) if d is None]
        # print("Indices where rms_dists is None:", none_indices)

        match_rate = sum(rms_dists != None) / len(self.preds[0])
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        result['match_rate'] = match_rate
        result['rms_dist'] = self.all_rms_dis
        
        for i in [1,3,5,10,20]:
            if i > self.batch_size:
                break
            result[f'match_rate_random_{i}'] = self.get_randomk_match_rate_and_rms(self.all_rms_dis, k=i)['match_rate']
            result[f'rms_dist_random_{i}'] = self.get_randomk_match_rate_and_rms(self.all_rms_dis, k=i)['rms_dist']
        if self.rank is not None:
            self.rank_all_rms_dist = np.take_along_axis(np.transpose(self.all_rms_dis, (1,0)), self.rank, axis=1)
            result['rank_all_rms_dist'] = self.rank_all_rms_dist
            for i in [1,3,5,10, 20]:
                none_indices = []
                if i > self.batch_size:
                    break
                rms_dists_i = self.rank_all_rms_dist[:, :i]
                min_rmse = []
                for j in range(len(rms_dists_i)):
                    if (np.isnan(rms_dists_i[j])).all():
                        none_indices.append(j)
                        pass
                    else:
                        min_rmse.append(np.nanmin(rms_dists_i[j])) 
                match_rate_i = len(min_rmse) / len(self.preds[0])
                mean_rms_dist_i = np.mean(min_rmse)
                result[f'match_rate_top_{i}'] = match_rate_i
                result[f'rms_dist_top_{i}'] = mean_rms_dist_i
                result[f"wrong idx when top_{i}"] = none_indices
        
        return result

    def get_metrics(self, dir):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms(dir))
        return metrics

def get_file_paths(root_path, task, label='', suffix='pt'):
    if args.label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data['frac_coords'][i],
                data['atom_types'][i],
                data['lengths'][i],
                data['angles'][i],
                data['num_atoms'][i])
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        crys_array_list = get_crystals_list(
            data['frac_coords'],
            data['atom_types'],
            data['lengths'],
            data['angles'],
            data['num_atoms'])        
    else:
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx])
    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    
    elif 'gt_frac_coords' in data and 'gt_atom_types' in data and 'gt_lengths' in data and 'gt_angles' in data and 'gt_num_atoms' in data:
        true_crystal_array_list = get_crystals_list(
            data['gt_frac_coords'], 
            data['gt_atom_types'], 
            data['gt_lengths'], 
            data['gt_angles'], 
            data['gt_num_atoms']
        )
    else:
        true_crystal_array_list = None

    return data["time"], crys_array_list, true_crystal_array_list


def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif,fmt='cif')
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles)
    }
    return Crystal(crys_array_dict) 

def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    all_metrics = {}

    recon_file_path = args.root_path
    batch_idx = -1 if args.multi_eval else 0
    generate_time, crys_array_list, true_crystal_array_list = get_crystal_array_list(
        recon_file_path, batch_idx = batch_idx)
    print(f"Number of predicted crystals: {len(true_crystal_array_list)}")

    enumerated_list = list(enumerate(true_crystal_array_list))
    gt_crys = p_map(lambda x: Crystal(x[1], x[0]), enumerated_list)

    if not args.multi_eval:
        pred_crys = p_map(lambda x: Crystal(x[1], x[0]), enumerate(crys_array_list))
    else:
        pred_crys = [] # sample times, materials (20, 100)
        for i in range(min(len(crys_array_list), args.bsz)): # num_evals
            print(f"Processing batch {i}")
            enumerated_list = list(enumerate(crys_array_list[i]))
            pred_crys.append(p_map(lambda x: Crystal(x[1], x[0]), enumerated_list))   

    if args.multi_eval:
        if args.sort:
            print(len(crys_array_list), len(crys_array_list[0]))
            energy = np.full((len(crys_array_list), len(crys_array_list[0])), np.inf)
            
            from ase import Atoms
            from ase.geometry import cellpar_to_cell
            from ase.io import read, write
            import sys
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)

            if args.sort_mode == 'dp':
                from deepmd.calculator import DP
                cal = DP("frozen_model.pth")
            elif args.sort_mode == 'mattersim':
                from mattersim.forcefield import MatterSimCalculator
                device = "cuda" if torch.cuda.is_available() else "cpu"
                cal = MatterSimCalculator(load_path='mattersim-v1.0.0-1M.pth', device = device)
            else:
                raise ValueError(f"Unsupported sort mode: {args.sort_mode}, only 'dp' and 'mattersim' are supported.")        
            
            start = time.time()
            for i in range(len(crys_array_list)): 
                print(f"Processing batch {i}")
                crystal_batch = crys_array_list[i] 
                for j in tqdm(range(len(crystal_batch))):
                    # print(i,j)
                    crys_array_dict = crystal_batch[j]
                    
                    try:
                        cell = cellpar_to_cell(np.concatenate([crys_array_dict['lengths'], crys_array_dict['angles']]))  # input [a, b, c, alpha, beta, gamma]，output 3x3 matrix
                    except AssertionError:
                        print(f"[Warning] Invalid cell parameters at index {i} {j}: {np.concatenate([crys_array_dict['lengths'], crys_array_dict['angles']])}")
                        invalid_cif_dir = os.path.join(args.save_path, 'invalid_cell_cifs/')
                        os.makedirs(invalid_cif_dir, exist_ok=True)
                        write(os.path.join(invalid_cif_dir, f'tmp_{i}_{j}.cif'), atoms)
                        continue  
                    
                    atoms = Atoms(
                        numbers=crys_array_dict['atom_types'],     
                        scaled_positions=crys_array_dict['frac_coords'],  
                        cell=cell,               
                        pbc=True                 
                    )
                    
                    vol = atoms.get_volume() / len(crys_array_dict['atom_types'])
                    # print(vol, crys_array_dict['lengths'], crys_array_dict['angles'])
                    if vol is None or vol <= 2 or crys_array_dict['angles'][0]>170 or crys_array_dict['angles'][1]>170 or crys_array_dict['angles'][2]>170:
                        print(f"[Warning] Invalid cell parameters at index {i} {j}: {np.concatenate([crys_array_dict['lengths'], crys_array_dict['angles']])}")
                        invalid_cif_dir = os.path.join(args.save_path, 'invalid_cell_cifs/')
                        os.makedirs(invalid_cif_dir, exist_ok=True)
                        write(os.path.join(invalid_cif_dir, f'tmp_{i}_{j}.cif'), atoms)
                        continue
                    atoms.calc = cal
                    # from ase.optimize import BFGS
                    # dyn = BFGS(atoms)
                    # dyn.run(steps=10, fmax=1e-3)
                    # print(atoms.get_potential_energy())
                    energy[i][j] = atoms.get_potential_energy()
            np.save(os.path.join(args.save_path, f'energy_{args.label}.npy'), energy)
            sorted_energy = np.sort(energy.transpose(1, 0), axis=1)
            rank = energy.transpose(1,0).argsort(axis=1) 
            np.save(os.path.join(args.save_path, f'sorted_energy_{args.label}.npy'), sorted_energy)
            np.save(os.path.join(args.save_path, f'rank_energy_{args.label}.npy'), rank)
            print("rank shape:", rank.shape) # (materials, sample times)
            cal_time = time.time() - start
        else:
            rank = None
        rec_evaluator = RecEvalBatch(pred_crys, gt_crys, rank, bsz=args.bsz, strict=args.strict)
    else:
        rec_evaluator = RecEval(pred_crys, gt_crys)

    recon_metrics = rec_evaluator.get_metrics(args.save_path)

    all_metrics.update(recon_metrics)

    rms_dists = all_metrics.pop('rms_dist', None)
    rank_all_rms_dist = all_metrics.pop('rank_all_rms_dist', None)
    # print(all_metrics)
    if rms_dists is not None:
        np.save(os.path.join(args.save_path, f'rms_dists_{args.label}.npy'), rms_dists)
    if rank_all_rms_dist is not None:
        np.save(os.path.join(args.save_path, f'rank_all_rms_dist_{args.label}.npy'), rank_all_rms_dist)
    if args.label == '':
        metrics_out_file = 'eval_metrics.json'
    else:
        metrics_out_file = f'eval_metrics_{args.label}.json'
    metrics_out_file = os.path.join(args.save_path, metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)

    print("\n" + "="*60)
    print("⏱️  Time consumption")
    print("="*60)

    total_number = len(crys_array_list) * len(crys_array_list[0])
    print(f"1. Generate structure:        {(generate_time/total_number):.2f}s")
    print(f"2. Calculate energy:        {(cal_time/total_number):.2f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True, help='the path to the reconstructed crystal data, should be a pt file containing frac_coords, atom_types, lengths, angles, num_atoms, and gt_* fields for true crystal structures')
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--multi_eval',action='store_true')
    parser.add_argument('--strict',action='store_true')
    parser.add_argument("--bsz", default='10000', type=int)
    parser.add_argument('--sort',action='store_true',help='whether to sort the predictions by energy, only works when --multi_eval is set')
    parser.add_argument("--sort_mode", default='mattersim', type=str, help='the mode to sort the predictions, can be either "mattersim" or "dp", only works when --sort is set')
    
    args = parser.parse_args()
    main(args)
