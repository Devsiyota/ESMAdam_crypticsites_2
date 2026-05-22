import os
import tempfile
import esm
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
import numpy as np

from torchmd_amber_energy import TorchMDAmberEnergy
from torchmd_amber_energy import energy_to_float
from torchmd_amber_energy import print_energy_details

from hydrogens_template import (
    make_atom14_atomic_numbers,
    add_template_hydrogens_to_output_nn,
    extract_heavy_and_template_hydrogen_pos_z,
    write_pdb_with_template_hydrogens,
)

from esmadam_cryptic_optimization_amberFF import * 



def seed_everything(seed=1234):
    import random

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_esmfold_model():
    print("Loading ESMFold...")
    #torch.hub.set_dir(ESMFOLD_CACHE_DIR)


    model_esm = esm.pretrained.esmfold_v1()
    model_esm = model_esm.eval().to(DEVICE)

    for p in model_esm.parameters():
        p.requires_grad_(False)

    return model_esm


def build_amber_energy_backend():
    amber_energy = TorchMDAmberEnergy(
        prmtop=AMBER_PRMTOP,
        pdb_file=PL_AMBER_PDB,
        device=DEVICE,
        precision=PRECISION,
        terms=AMBER_TERMS,
    )

    amber_energy.print_system_info()

    return amber_energy


def parse_ranges(ranges_spec):
    ranges = []

    for chunk in ranges_spec.split(","):
        chunk = chunk.strip()

        if chunk == "":
            continue

        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            lo = int(lo)
            hi = int(hi)

            if lo > hi:
                lo, hi = hi, lo

            ranges.append((lo, hi))
        else:
            resid = int(chunk)
            ranges.append((resid, resid))

    return ranges


def resid_in_ranges(resid, ranges):
    for lo, hi in ranges:
        if lo <= resid <= hi:
            return True

    return False


def read_pdb_atom_records(pdb_path,device, precision, include_hetatm=False):
    records = []

    allowed = ("ATOM", "HETATM") if include_hetatm else ("ATOM",)

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith(allowed):
                continue

            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21:22].strip()
            resid = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            records.append(
                {
                    "line": line.rstrip("\n"),
                    "atom_name": atom_name,
                    "res_name": res_name,
                    "chain_id": chain_id,
                    "resid": resid,
                    "xyz": [x, y, z],
                }
            )
        coords = [record["xyz"] for record in records]
        resids = [record["resid"] for record in records]
        ref_pos = torch.tensor(coords,dtype=precision,device=device)

    return ref_pos, resids




def make_atom_mask_from_resids(resids, ranges_spec, device):
    ranges = parse_ranges(ranges_spec)
    mask = [resid_in_ranges(resid, ranges) for resid in resids]

    return torch.tensor(
        mask,
        dtype=torch.bool,
        device=device,
    )


def compress_resids_to_ranges_spec(resids):
    resids = sorted(set(int(resid) for resid in resids))

    if len(resids) == 0:
        raise RuntimeError("No residues selected for restraint")

    ranges = []
    start = resids[0]
    previous = resids[0]

    for resid in resids[1:]:
        if resid == previous + 1:
            previous = resid
            continue

        if start == previous:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{previous}")

        start = resid
        previous = resid

    if start == previous:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{previous}")

    return ",".join(ranges)


def find_resids_not_within_ligand(
    protein_pos,
    protein_resids,
    ligand_pos,
    cutoff,
):
    """
    Select protein residues where NO atom of that residue is within cutoff Å
    of any ligand atom.

    This means:
    - residues within cutoff of ligand are left flexible
    - residues outside cutoff are selected for backbone restraint

    Hydrogens are included if they exist in protein_pos and ligand_pos.
    """
    if protein_pos.dim() != 2 or protein_pos.shape[1] != 3:
        raise RuntimeError(f"Expected protein_pos shape [N, 3], got {tuple(protein_pos.shape)}")

    if ligand_pos.dim() != 2 or ligand_pos.shape[1] != 3:
        raise RuntimeError(f"Expected ligand_pos shape [M, 3], got {tuple(ligand_pos.shape)}")

    if protein_pos.shape[0] != len(protein_resids):
        raise RuntimeError(
            f"protein_pos/protein_resids mismatch: "
            f"{protein_pos.shape[0]} atoms vs {len(protein_resids)} residue ids"
        )

    distances = torch.cdist(protein_pos, ligand_pos)

    atom_close_mask = torch.any(distances <= cutoff, dim=1)

    close_resids = set()

    for atom_is_close, resid in zip(atom_close_mask.detach().cpu().tolist(), protein_resids):
        if atom_is_close:
            close_resids.add(int(resid))

    all_resids = sorted(set(int(resid) for resid in protein_resids))

    selected_resids = [
        resid for resid in all_resids
        if resid not in close_resids
    ]

    if len(selected_resids) == 0:
        raise RuntimeError(
            f"No residues found outside {cutoff:.2f} Å of ligand. "
            f"That means every residue has at least one atom close to the ligand, which is suspicious."
        )

    print("Residues within ligand cutoff, left flexible:", sorted(close_resids))
    print("Residues outside ligand cutoff, restrained:", selected_resids)

    return selected_resids



def parse_pdb_backbone_in_ranges(pdb_file, ranges_spec, require_full=False):
    backbone = ("N", "CA", "C", "O")
    wanted = parse_ranges(ranges_spec)
    bb_coords = {}
    seen = set()

    with open(pdb_file, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()

            if atom_name not in backbone:
                continue

            resid = int(line[22:26].strip())

            if not resid_in_ranges(resid, wanted):
                continue

            if (resid, atom_name) in seen:
                continue

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            bb_coords.setdefault(resid, {})[atom_name] = np.array(
                [x, y, z],
                dtype=np.float32,
            )
            seen.add((resid, atom_name))

    ordered_keys = []
    coords_list = []

    for resid in sorted(bb_coords.keys()):
        if require_full and any(atom_name not in bb_coords[resid] for atom_name in backbone):
            continue

        for atom_name in backbone:
            if atom_name in bb_coords[resid]:
                ordered_keys.append((resid, atom_name))
                coords_list.append(bb_coords[resid][atom_name])

    if len(coords_list) == 0:
        raise RuntimeError(
            f"No backbone atoms found in {pdb_file} for ranges {ranges_spec}"
        )

    coords_array = np.vstack(coords_list).astype(np.float32)
    print("ordered keys", ordered_keys)


    return ordered_keys, coords_array


def select_output_backbone_by_keys(output_positions, ordered_keys, step=-1, batch=0):
    if output_positions.dim() == 4:
        output_positions = output_positions.unsqueeze(1)

    block = output_positions[step, batch]

    atom_to_idx = {
        "N": 0,
        "CA": 1,
        "C": 2,
        "O": 3,
    }

    coords = []

    for resid, atom_name in ordered_keys:
        res_idx = resid - 1
        atom_idx = atom_to_idx[atom_name]
        coords.append(block[res_idx, atom_idx, :])

    return torch.stack(coords, dim=0)


def kabsch_torch(P, Q, eps=1e-8):
    """
    Rigidly align P onto Q using Kabsch.

    Returns:
        R: [3, 3] rotation matrix
        t: [3] translation vector
        rmsd: fitted RMSD after applying P @ R.T + t

    Apply as:
        P_aligned = P @ R.T + t
    """
    if P.shape != Q.shape:
        raise RuntimeError(f"Kabsch shape mismatch: P={tuple(P.shape)} Q={tuple(Q.shape)}")

    centroid_P = P.mean(dim=0, keepdim=True)
    centroid_Q = Q.mean(dim=0, keepdim=True)

    p = P - centroid_P
    q = Q - centroid_Q

    H = p.transpose(0, 1) @ q
    U, S, Vt = torch.linalg.svd(H, full_matrices=False)
    V = Vt.mT

    det_vu = torch.det(V @ U.mT)
    sign = torch.where(
        det_vu < 0.0,
        torch.tensor(-1.0, device=H.device, dtype=H.dtype),
        torch.tensor(1.0, device=H.device, dtype=H.dtype),
    )

    E = torch.diag(
        torch.stack(
            [
                torch.ones((), device=H.device, dtype=H.dtype),
                torch.ones((), device=H.device, dtype=H.dtype),
                sign,
            ]
        )
    )

    R = V @ E @ U.mT
    t = centroid_Q - centroid_P @ R.mT
    t = t.squeeze(0)

    P_aligned = P @ R.mT + t
    rmsd = torch.sqrt(torch.mean(torch.sum((P_aligned - Q) ** 2, dim=-1)) + eps)

    return R, t, rmsd


def apply_rigid_transform(coords, R, t):
    if coords.dim() != 2 or coords.shape[1] != 3:
        raise RuntimeError(f"Expected coords shape [N, 3], got {tuple(coords.shape)}")

    return coords @ R.mT + t


def make_ligand_follow_generated_protein(
    ligand_ref_pos,
    restraint_ref_coords,
    restraint_current_coords,
):
    """
    Move the ligand from the reference protein frame into the generated protein frame.

    The transform is computed from restrained backbone atoms:
        reference restrained backbone -> current generated restrained backbone

    Then the same rigid transform is applied to every ligand atom.
    """
    R_ref_to_current, t_ref_to_current, rmsd_fit = kabsch_torch(
        P=restraint_ref_coords,
        Q=restraint_current_coords,
    )

    ligand_current_pos = apply_rigid_transform(
        coords=ligand_ref_pos,
        R=R_ref_to_current,
        t=t_ref_to_current,
    )

    return ligand_current_pos, rmsd_fit


def compute_ca_distances(coords):
    diff_i1 = coords[:, :-1, :] - coords[:, 1:, :]
    dist_i1 = torch.norm(diff_i1, dim=-1)

    return dist_i1


def compute_local_ca_loss(output_positions):
    if output_positions.dim() == 4:
        output_positions = output_positions.unsqueeze(1)

    ca = output_positions[-1, 0, :, 1, :]

    ca_all = ca.unsqueeze(0)
    cur_ca_dist = compute_ca_distances(ca_all)
    target = torch.ones_like(cur_ca_dist) * 3.8

    return F.mse_loss(cur_ca_dist, target)


def get_amber_ordered_pos_z_from_output(output_nn, sequence, z14):
    output_nn_H = add_template_hydrogens_to_output_nn(
        output_nn=output_nn,
        sequence=sequence,
        amber_pdb_path=PROTEIN_AMBER_PDB,
    )

    pos_all, z_all = extract_heavy_and_template_hydrogen_pos_z(
        output_nn_H=output_nn_H,
        z14=z14,
    )

    return output_nn_H, pos_all, z_all


def compute_amber_energy(amber_energy, pos_all):
    if pos_all.dim() != 2 or pos_all.shape[1] != 3:
        raise RuntimeError(f"Expected pos_all shape [natoms, 3], got {tuple(pos_all.shape)}")

    pos_amber = pos_all.unsqueeze(0).to(DEVICE)
    energy = amber_energy.energy(pos_amber)

    return energy




def guess_element_from_atom_name(atom_name):
    name = atom_name.strip()

    if len(name) == 0:
        raise RuntimeError("Empty atom name")

    two = name[:2].upper()
    one = name[0].upper()

    if two in ELEMENT_TO_Z:
        return two

    if one in ELEMENT_TO_Z:
        return one

    raise RuntimeError(f"Could not guess element from atom name: {atom_name}")


def read_ligand_pdb_as_pos_z(pdb_path, device, precision=torch.float32):
    ligand_lines = []
    coords = []
    z_list = []

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            atom_name = line[12:16].strip()

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            element = line[76:78].strip().upper()

            if element == "":
                element = guess_element_from_atom_name(atom_name)

            atomic_number = ELEMENT_TO_Z[element]

            ligand_lines.append(line.rstrip("\n"))
            coords.append([x, y, z])
            z_list.append(atomic_number)

    if len(coords) == 0:
        raise RuntimeError(f"No ATOM/HETATM records found in ligand PDB: {pdb_path}")

    ligand_pos = torch.tensor(coords, dtype=precision, device=device)
    ligand_z = torch.tensor(z_list, dtype=torch.long, device=device)

    return ligand_pos, ligand_z, ligand_lines


def make_complex_pos_z(protein_pos, protein_z, ligand_pos, ligand_z):
    pos_complex = torch.cat(
        [protein_pos, ligand_pos],
        dim=0,
    )

    z_complex = torch.cat(
        [protein_z, ligand_z],
        dim=0,
    )

    return pos_complex, z_complex


def write_protein_H_ligand_pdb_using_template_writer(
    output_nn_H,
    sequence,
    ligand_lines,
    output_pdb,
    amber_pdb_path=None,
    chain_id="A",
    ligand_pos_override=None,
):
    tmp_dir = os.path.dirname(output_pdb)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".pdb",
        dir=tmp_dir,
        delete=False,
    ) as tmp:
        tmp_protein_pdb = tmp.name

    write_pdb_with_template_hydrogens(
        output_nn_H=output_nn_H,
        sequence=sequence,
        filename=tmp_protein_pdb,
        amber_pdb_path=amber_pdb_path,
        chain_id=chain_id,
    )

    protein_lines = []

    with open(tmp_protein_pdb, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                protein_lines.append(line.rstrip("\n"))

    os.remove(tmp_protein_pdb)

    with open(output_pdb, "w") as f:
        serial = 1

        for line in protein_lines:
            f.write(f"{line[:6]}{serial:5d}{line[11:]}\n")
            serial += 1

        f.write("TER\n")

        ligand_index = 0
        for line in ligand_lines:
            if line.startswith(("ATOM", "HETATM")):
                if ligand_pos_override is None:
                    f.write(f"HETATM{serial:5d}{line[11:]}\n")
                else:
                    xyz = ligand_pos_override[ligand_index].detach().cpu().tolist()
                    x, y, z = xyz
                    new_line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                    f.write(f"HETATM{serial:5d}{new_line[11:]}\n")
                serial += 1
                ligand_index += 1

        f.write("TER\n")
        f.write("END\n")

