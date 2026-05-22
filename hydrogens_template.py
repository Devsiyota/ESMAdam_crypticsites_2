import copy
import math

import torch


# ============================================================
# Residue and atom definitions
# ============================================================

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

atom14_names = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
}

element_to_z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "S": 16,
}


# ============================================================
# Heavy atom bond templates
# ============================================================

residue_heavy_bonds = {
    "ALA": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")],
    "ARG": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
    "ASN": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
    "ASP": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
    "CYS": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "SG")],
    "GLN": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")],
    "GLU": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
    "GLY": [("N", "CA"), ("CA", "C"), ("C", "O")],
    "HIS": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"), ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2")],
    "ILE": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")],
    "LEU": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "LYS": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
    "MET": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
    "PHE": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ")],
    "PRO": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")],
    "SER": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "OG")],
    "THR": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")],
    "TRP": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"), ("NE1", "CE2"), ("CE2", "CD2"), ("CD2", "CE3"), ("CE3", "CZ3"), ("CZ3", "CH2"), ("CH2", "CZ2"), ("CZ2", "CE2")],
    "TYR": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH")],
    "VAL": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")],
}


# ============================================================
# Hydrogen templates
# ============================================================

hydrogen_templates = {
    "ALA": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB1", "HB2", "HB3"], "sp3_3", "CA"),
    },
    "ARG": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "CG": (["HG2", "HG3"], "sp3_2", None),
        "CD": (["HD2", "HD3"], "sp3_2", None),
        "NE": (["HE"], "sp2_1", None),
        "NH1": (["HH11", "HH12"], "guanidinium_2", "NE"),
        "NH2": (["HH21", "HH22"], "guanidinium_2", "NE"),
    },
    "ASN": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "ND2": (["HD21", "HD22"], "amide_2", "OD1"),
    },
    "ASP": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
    },
    "CYS": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "SG": (["HG"], "thiol_1", "CA"),
    },
    "GLN": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "CG": (["HG2", "HG3"], "sp3_2", None),
        "NE2": (["HE21", "HE22"], "amide_2", "OE1"),
    },
    "GLU": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "CG": (["HG2", "HG3"], "sp3_2", None),
    },
    "GLY": {
        "CA": (["HA2", "HA3"], "sp3_2", None),
    },
    "HIS": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "ND1": (["HD1"], "sp2_1", None),
        "NE2": (["HE2"], "sp2_1", None), # this line updated or extra if Histidine is of type HIE
        "CD2": (["HD2"], "sp2_1", None),
        "CE1": (["HE1"], "sp2_1", None),
    },
    "ILE": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB"], "sp3_1", None),
        "CG1": (["HG12", "HG13"], "sp3_2", None),
        "CG2": (["HG21", "HG22", "HG23"], "sp3_3", "CB"),
        "CD1": (["HD11", "HD12", "HD13"], "sp3_3", "CG1"),
    },
    "LEU": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "CG": (["HG"], "sp3_1", None),
        "CD1": (["HD11", "HD12", "HD13"], "sp3_3", "CG"),
        "CD2": (["HD21", "HD22", "HD23"], "sp3_3", "CG"),
    },
    "LYS": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "CG": (["HG2", "HG3"], "sp3_2", None),
        "CD": (["HD2", "HD3"], "sp3_2", None),
        "CE": (["HE2", "HE3"], "sp3_2", None),
        "NZ": (["HZ1", "HZ2", "HZ3"], "sp3_3", "CE"),
    },
    "MET": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "CG": (["HG2", "HG3"], "sp3_2", None),
        "CE": (["HE1", "HE2", "HE3"], "sp3_3", "SD"),
    },
    "PHE": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "CD1": (["HD1"], "sp2_1", None),
        "CD2": (["HD2"], "sp2_1", None),
        "CE1": (["HE1"], "sp2_1", None),
        "CE2": (["HE2"], "sp2_1", None),
        "CZ": (["HZ"], "sp2_1", None),
    },
    "PRO": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "CG": (["HG2", "HG3"], "sp3_2", None),
        "CD": (["HD2", "HD3"], "sp3_2", None),
    },
    "SER": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "OG": (["HG"], "hydroxyl_1", "CA"),
    },
    "THR": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB"], "sp3_1", None),
        "OG1": (["HG1"], "hydroxyl_1", "CA"),
        "CG2": (["HG21", "HG22", "HG23"], "sp3_3", "CB"),
    },
    "TRP": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "CD1": (["HD1"], "sp2_1", None),
        "NE1": (["HE1"], "sp2_1", None),
        "CE3": (["HE3"], "sp2_1", None),
        "CZ2": (["HZ2"], "sp2_1", None),
        "CZ3": (["HZ3"], "sp2_1", None),
        "CH2": (["HH2"], "sp2_1", None),
    },
    "TYR": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB2", "HB3"], "sp3_2", None),
        "CD1": (["HD1"], "sp2_1", None),
        "CD2": (["HD2"], "sp2_1", None),
        "CE1": (["HE1"], "sp2_1", None),
        "CE2": (["HE2"], "sp2_1", None),
        "OH": (["HH"], "hydroxyl_1", "CE1"),
    },
    "VAL": {
        "CA": (["HA"], "sp3_1", None),
        "CB": (["HB"], "sp3_1", None),
        "CG1": (["HG11", "HG12", "HG13"], "sp3_3", "CB"),
        "CG2": (["HG21", "HG22", "HG23"], "sp3_3", "CB"),
    },
}


# ============================================================
# General utilities
# ============================================================

def atom_name_to_atomic_number(atom_name):
    if atom_name == "":
        return 0

    first_letter = atom_name[0]

    if first_letter not in element_to_z:
        raise ValueError(f"Unknown atom element for atom name: {atom_name}")

    return element_to_z[first_letter]


def make_atom14_atomic_numbers(sequence, device):
    z_all = []

    for aa in sequence:
        if aa not in restype_1to3:
            raise ValueError(f"Unknown amino acid: {aa}")

        resname = restype_1to3[aa]
        names = atom14_names[resname]
        z_res = [atom_name_to_atomic_number(name) for name in names]
        z_all.append(z_res)

    return torch.tensor(z_all, dtype=torch.long, device=device)


def extract_pos_z(output, z14):
    pos14 = output["positions"][-1, 0]
    mask14 = output["atom14_atom_exists"][0].bool()

    z14 = z14.to(pos14.device)

    valid = mask14 & (z14 > 0)

    pos = pos14[valid]
    z = z14[valid]

    return pos, z


def output_to_pdb_string(model_esm, output_nn):
    output_cpu = {}

    for k, v in output_nn.items():
        if torch.is_tensor(v):
            output_cpu[k] = v.detach().cpu()
        else:
            output_cpu[k] = v

    pdb_str = model_esm.output_to_pdb(output_cpu)[0]
    return pdb_str


def write_output_nn_pdb(model_esm, output_nn, filename):
    pdb_str = output_to_pdb_string(model_esm, output_nn)

    with open(filename, "w") as f:
        f.write(pdb_str)


# ============================================================
# Diagnostics
# ============================================================

def diagnose_positions(pos, z, name="pos"):
    print(f"{name} shape:", pos.shape)
    print(f"{name} finite:", torch.isfinite(pos).all().item())
    print(f"{name} min:", pos.min().item())
    print(f"{name} max:", pos.max().item())
    print(f"{name} z unique:", torch.unique(z).detach().cpu().tolist())

    if not torch.isfinite(pos).all():
        bad = ~torch.isfinite(pos)
        print(f"{name} bad coordinate count:", bad.sum().item())
        raise RuntimeError(f"{name} contains NaN or Inf")

    with torch.no_grad():
        d = torch.cdist(pos.detach(), pos.detach())
        eye = torch.eye(d.shape[0], dtype=torch.bool, device=d.device)
        d[eye] = 1e6

        min_dist = d.min().item()
        idx = torch.nonzero(d == d.min(), as_tuple=False)[0]

        print(f"{name} minimum interatomic distance:", min_dist)
        print(f"{name} closest atom indices:", idx.tolist())
        print(f"{name} closest atom z:", z[idx[0]].item(), z[idx[1]].item())

        if min_dist < 0.35:
            raise RuntimeError(
                f"Atom clash detected in {name}: minimum distance {min_dist:.4f} Å"
            )


# ============================================================
# Differentiable template hydrogen geometry
# ============================================================

def safe_normalize(v, eps=1e-8):
    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / (norm + eps)


def get_atom_position_by_name(pos14, mask14, residue_index, atom_name, sequence):
    aa = sequence[residue_index]
    resname = restype_1to3[aa]
    names = atom14_names[resname]

    if atom_name not in names:
        return None

    atom_index = names.index(atom_name)

    if not bool(mask14[residue_index, atom_index]):
        return None

    return pos14[residue_index, atom_index]


def get_heavy_neighbors_from_template(resname, atom_name):
    neighbors = []

    for a, b in residue_heavy_bonds[resname]:
        if atom_name == a:
            neighbors.append(b)
        elif atom_name == b:
            neighbors.append(a)

    return neighbors


def get_neighbor_positions(pos14, mask14, sequence, residue_index, atom_name):
    aa = sequence[residue_index]
    resname = restype_1to3[aa]

    neighbor_names = get_heavy_neighbors_from_template(
        resname=resname,
        atom_name=atom_name
    )

    neighbor_positions = []

    for neigh_atom in neighbor_names:
        neigh_pos = get_atom_position_by_name(
            pos14=pos14,
            mask14=mask14,
            residue_index=residue_index,
            atom_name=neigh_atom,
            sequence=sequence
        )

        if neigh_pos is not None:
            neighbor_positions.append(neigh_pos)

    if atom_name == "N" and residue_index > 0:
        prev_c = get_atom_position_by_name(
            pos14=pos14,
            mask14=mask14,
            residue_index=residue_index - 1,
            atom_name="C",
            sequence=sequence
        )

        if prev_c is not None:
            neighbor_positions.append(prev_c)

    if atom_name == "C" and residue_index < len(sequence) - 1:
        next_n = get_atom_position_by_name(
            pos14=pos14,
            mask14=mask14,
            residue_index=residue_index + 1,
            atom_name="N",
            sequence=sequence
        )

        if next_n is not None:
            neighbor_positions.append(next_n)

    return neighbor_positions


def make_perpendicular_basis(axis, ref=None, eps=1e-6):
    axis = safe_normalize(axis)

    dtype = axis.dtype
    device = axis.device

    ref_x = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    ref_y = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
    ref_z = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

    candidate_refs = []

    if ref is not None:
        candidate_refs.append(ref)

    candidate_refs.extend([ref_x, ref_y, ref_z])

    best_ref = None
    best_norm = None

    for candidate in candidate_refs:
        candidate = candidate.to(dtype=dtype, device=device)
        projected = candidate - torch.sum(candidate * axis) * axis
        projected_norm = torch.linalg.norm(projected)

        if best_norm is None or bool(projected_norm > best_norm):
            best_norm = projected_norm
            best_ref = projected

    if best_ref is None or bool(best_norm < eps):
        raise RuntimeError("Could not construct perpendicular basis for hydrogen geometry")

    e1 = safe_normalize(best_ref)
    e2 = torch.cross(axis, e1, dim=-1)
    e2 = safe_normalize(e2)

    return e1, e2


def hydrogen_bond_length(parent_atom_name):
    first = parent_atom_name[0]

    if first == "C":
        return 1.09
    if first == "N":
        return 1.01
    if first == "O":
        return 0.96
    if first == "S":
        return 1.34

    raise ValueError(f"Unknown hydrogen parent atom type: {parent_atom_name}")


def geometry_sp2_single(parent, neighbor_positions):
    if len(neighbor_positions) < 2:
        repel = torch.zeros_like(parent)

        for neigh in neighbor_positions:
            repel = repel - safe_normalize(neigh - parent)

        if len(neighbor_positions) == 0 or bool(torch.linalg.norm(repel) < 1e-6):
            repel = torch.tensor(
                [1.0, 0.0, 0.0],
                dtype=parent.dtype,
                device=parent.device
            )

        return [safe_normalize(repel)]

    d1 = safe_normalize(neighbor_positions[0] - parent)
    d2 = safe_normalize(neighbor_positions[1] - parent)

    h_dir = -(d1 + d2)

    if bool(torch.linalg.norm(h_dir) < 1e-6):
        e1, e2 = make_perpendicular_basis(d1)
        h_dir = e1

    h_dir = safe_normalize(h_dir)

    return [h_dir]


def geometry_sp3_single(parent, neighbor_positions):
    repel = torch.zeros_like(parent)

    for neigh in neighbor_positions:
        repel = repel - safe_normalize(neigh - parent)

    if len(neighbor_positions) == 0 or bool(torch.linalg.norm(repel) < 1e-6):
        repel = torch.tensor(
            [1.0, 0.0, 0.0],
            dtype=parent.dtype,
            device=parent.device
        )

    return [safe_normalize(repel)]


def geometry_sp3_two(parent, neighbor_positions):
    dtype = parent.dtype
    device = parent.device

    if len(neighbor_positions) == 0:
        return [
            torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device),
            torch.tensor([-1.0, 0.0, 0.0], dtype=dtype, device=device),
        ]

    if len(neighbor_positions) == 1:
        axis_to_heavy = safe_normalize(neighbor_positions[0] - parent)
        axis_away = -axis_to_heavy

        e1, e2 = make_perpendicular_basis(axis_away)

        angle = math.radians(70.5)

        h1 = math.cos(angle) * axis_away + math.sin(angle) * e1
        h2 = math.cos(angle) * axis_away - math.sin(angle) * e1

        return [safe_normalize(h1), safe_normalize(h2)]

    d1 = safe_normalize(neighbor_positions[0] - parent)
    d2 = safe_normalize(neighbor_positions[1] - parent)

    axis_away_raw = -(d1 + d2)

    if bool(torch.linalg.norm(axis_away_raw) < 1e-6):
        axis_away, _ = make_perpendicular_basis(d1)
    else:
        axis_away = safe_normalize(axis_away_raw)

    plane_normal = torch.cross(d1, d2, dim=-1)

    if bool(torch.linalg.norm(plane_normal) < 1e-6):
        e1, e2 = make_perpendicular_basis(axis_away)
        plane_normal = e1
    else:
        plane_normal = safe_normalize(plane_normal)

    angle = math.radians(55.0)

    h1 = math.cos(angle) * axis_away + math.sin(angle) * plane_normal
    h2 = math.cos(angle) * axis_away - math.sin(angle) * plane_normal

    return [safe_normalize(h1), safe_normalize(h2)]


def geometry_sp3_three(parent, neighbor_positions, ref_position=None):
    dtype = parent.dtype
    device = parent.device

    if len(neighbor_positions) == 0:
        axis_away = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    else:
        axis_to_heavy = safe_normalize(neighbor_positions[0] - parent)
        axis_away = -axis_to_heavy

    ref_vec = None

    if ref_position is not None:
        ref_vec = safe_normalize(ref_position - parent)

    e1, e2 = make_perpendicular_basis(axis_away, ref=ref_vec)

    angle = math.radians(70.5)

    phi1 = 0.0
    phi2 = 2.0 * math.pi / 3.0
    phi3 = 4.0 * math.pi / 3.0

    h1 = math.cos(angle) * axis_away + math.sin(angle) * (
        math.cos(phi1) * e1 + math.sin(phi1) * e2
    )

    h2 = math.cos(angle) * axis_away + math.sin(angle) * (
        math.cos(phi2) * e1 + math.sin(phi2) * e2
    )

    h3 = math.cos(angle) * axis_away + math.sin(angle) * (
        math.cos(phi3) * e1 + math.sin(phi3) * e2
    )

    return [safe_normalize(h1), safe_normalize(h2), safe_normalize(h3)]


def geometry_sp2_two(parent, anchor_position, plane_ref_position):
    axis_away = safe_normalize(parent - anchor_position)

    ref_vec = plane_ref_position - anchor_position
    ref_vec = ref_vec - torch.sum(ref_vec * axis_away) * axis_away

    if bool(torch.linalg.norm(ref_vec) < 1e-6):
        ref_vec, _ = make_perpendicular_basis(axis_away)
    else:
        ref_vec = safe_normalize(ref_vec)

    angle = math.radians(60.0)

    h1 = math.cos(angle) * axis_away + math.sin(angle) * ref_vec
    h2 = math.cos(angle) * axis_away - math.sin(angle) * ref_vec

    return [safe_normalize(h1), safe_normalize(h2)]


def geometry_hydroxyl(parent, neighbor_positions, ref_position=None, angle_deg=120.0):
    dtype = parent.dtype
    device = parent.device

    if len(neighbor_positions) == 0:
        return [
            torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        ]

    heavy_dir = safe_normalize(neighbor_positions[0] - parent)
    axis_away = -heavy_dir

    ref_vec = None

    if ref_position is not None:
        ref_vec = ref_position - parent
    elif len(neighbor_positions) >= 2:
        ref_vec = neighbor_positions[1] - parent

    e1, e2 = make_perpendicular_basis(axis_away, ref=ref_vec)

    theta_from_axis_away = math.radians(180.0 - angle_deg)

    h_dir = (
        math.cos(theta_from_axis_away) * axis_away
        + math.sin(theta_from_axis_away) * e1
    )

    return [safe_normalize(h_dir)]


def get_backbone_hydrogen_template(resname, residue_index):
    if residue_index == 0:
        if resname == "PRO":
            return ["H1", "H2"], "sp3_2"
        return ["H1", "H2", "H3"], "sp3_3"

    if resname == "PRO":
        return [], None

    return ["H"], "peptide_N"


def make_hydrogens_for_parent(
    pos14,
    mask14,
    sequence,
    residue_index,
    parent_atom_name,
    hydrogen_names,
    geometry_type,
    ref_atom_name=None
):
    parent = get_atom_position_by_name(
        pos14=pos14,
        mask14=mask14,
        residue_index=residue_index,
        atom_name=parent_atom_name,
        sequence=sequence
    )

    if parent is None:
        return [], []

    neighbor_positions = get_neighbor_positions(
        pos14=pos14,
        mask14=mask14,
        sequence=sequence,
        residue_index=residue_index,
        atom_name=parent_atom_name
    )

    ref_position = None

    if ref_atom_name is not None:
        ref_position = get_atom_position_by_name(
            pos14=pos14,
            mask14=mask14,
            residue_index=residue_index,
            atom_name=ref_atom_name,
            sequence=sequence
        )

    if geometry_type == "sp2_1":
        h_dirs = geometry_sp2_single(parent, neighbor_positions)

    elif geometry_type == "sp3_1":
        h_dirs = geometry_sp3_single(parent, neighbor_positions)

    elif geometry_type == "sp3_2":
        h_dirs = geometry_sp3_two(parent, neighbor_positions)

    elif geometry_type == "sp3_3":
        h_dirs = geometry_sp3_three(
            parent=parent,
            neighbor_positions=neighbor_positions,
            ref_position=ref_position
        )

    elif geometry_type == "peptide_N":
        h_dirs = geometry_sp2_single(parent, neighbor_positions)

    elif geometry_type == "hydroxyl_1":
        h_dirs = geometry_hydroxyl(
            parent=parent,
            neighbor_positions=neighbor_positions,
            ref_position=ref_position,
            angle_deg=120.0
        )

    elif geometry_type == "thiol_1":
        h_dirs = geometry_hydroxyl(
            parent=parent,
            neighbor_positions=neighbor_positions,
            ref_position=ref_position,
            angle_deg=105.0
        )

    elif geometry_type == "amide_2":
        if len(neighbor_positions) == 0 or ref_position is None:
            h_dirs = geometry_sp3_two(parent, neighbor_positions)
        else:
            anchor_position = neighbor_positions[0]
            h_dirs = geometry_sp2_two(parent, anchor_position, ref_position)

    elif geometry_type == "guanidinium_2":
        if len(neighbor_positions) == 0 or ref_position is None:
            h_dirs = geometry_sp3_two(parent, neighbor_positions)
        else:
            anchor_position = neighbor_positions[0]
            h_dirs = geometry_sp2_two(parent, anchor_position, ref_position)

    else:
        raise ValueError(f"Unknown hydrogen geometry type: {geometry_type}")

    bond_length = hydrogen_bond_length(parent_atom_name)

    h_positions = []

    for h_dir in h_dirs[:len(hydrogen_names)]:
        h_positions.append(parent + bond_length * h_dir)

    return hydrogen_names[:len(h_positions)], h_positions


def add_template_hydrogens_to_output_nn(output_nn, sequence):
    pos14 = output_nn["positions"][-1, 0]
    mask14 = output_nn["atom14_atom_exists"][0].bool()

    dtype = pos14.dtype
    device = pos14.device

    all_residue_h_positions = []
    all_residue_h_names = []

    for residue_index, aa in enumerate(sequence):
        resname = restype_1to3[aa]

        residue_h_names = []
        residue_h_positions = []

        bb_h_names, bb_geometry_type = get_backbone_hydrogen_template(
            resname=resname,
            residue_index=residue_index
        )

        if len(bb_h_names) > 0:
            h_names, h_positions = make_hydrogens_for_parent(
                pos14=pos14,
                mask14=mask14,
                sequence=sequence,
                residue_index=residue_index,
                parent_atom_name="N",
                hydrogen_names=bb_h_names,
                geometry_type=bb_geometry_type,
                ref_atom_name="CA"
            )

            residue_h_names.extend(h_names)
            residue_h_positions.extend(h_positions)

        residue_template = hydrogen_templates[resname]

        for parent_atom_name, template_data in residue_template.items():
            hydrogen_names, geometry_type, ref_atom_name = template_data

            h_names, h_positions = make_hydrogens_for_parent(
                pos14=pos14,
                mask14=mask14,
                sequence=sequence,
                residue_index=residue_index,
                parent_atom_name=parent_atom_name,
                hydrogen_names=hydrogen_names,
                geometry_type=geometry_type,
                ref_atom_name=ref_atom_name
            )

            residue_h_names.extend(h_names)
            residue_h_positions.extend(h_positions)

        all_residue_h_names.append(residue_h_names)
        all_residue_h_positions.append(residue_h_positions)

    max_h = max(len(x) for x in all_residue_h_positions)

    padded_positions_per_residue = []
    padded_mask_per_residue = []
    padded_names_per_residue = []

    zero_pos = torch.zeros(3, dtype=dtype, device=device)

    for residue_h_names, residue_h_positions in zip(
        all_residue_h_names,
        all_residue_h_positions
    ):
        padded_positions = []
        padded_mask = []
        padded_names = []

        for h_name, h_pos in zip(residue_h_names, residue_h_positions):
            padded_names.append(h_name)
            padded_positions.append(h_pos)
            padded_mask.append(torch.tensor(True, dtype=torch.bool, device=device))

        while len(padded_positions) < max_h:
            padded_names.append("")
            padded_positions.append(zero_pos)
            padded_mask.append(torch.tensor(False, dtype=torch.bool, device=device))

        padded_positions_per_residue.append(torch.stack(padded_positions, dim=0))
        padded_mask_per_residue.append(torch.stack(padded_mask, dim=0))
        padded_names_per_residue.append(padded_names)

    hydrogen_positions = torch.stack(padded_positions_per_residue, dim=0).unsqueeze(0)
    hydrogen_atom_exists = torch.stack(padded_mask_per_residue, dim=0).unsqueeze(0)

    output_nn_H = copy.copy(output_nn)

    output_nn_H["hydrogen_positions"] = hydrogen_positions
    output_nn_H["hydrogen_atom_exists"] = hydrogen_atom_exists
    output_nn_H["hydrogen_atom_names"] = padded_names_per_residue

    return output_nn_H


def extract_hydrogen_pos_z(output_nn_H):
    h_pos_all = output_nn_H["hydrogen_positions"]
    h_mask_all = output_nn_H["hydrogen_atom_exists"]

    h_pos = h_pos_all[h_mask_all]

    h_z = torch.ones(
        h_pos.shape[0],
        dtype=torch.long,
        device=h_pos.device
    )

    return h_pos, h_z


def extract_heavy_and_template_hydrogen_pos_z(output_nn_H, z14):
    heavy_pos, heavy_z = extract_pos_z(output_nn_H, z14)
    h_pos, h_z = extract_hydrogen_pos_z(output_nn_H)

    all_pos = torch.cat([heavy_pos, h_pos], dim=0)
    all_z = torch.cat([heavy_z, h_z], dim=0)

    return all_pos, all_z


# ============================================================
# Debug PDB writer for hydrogens
# ============================================================

def write_pdb_with_template_hydrogens(output_nn_H, sequence, filename):
    pos14 = output_nn_H["positions"][-1, 0].detach().cpu()
    mask14 = output_nn_H["atom14_atom_exists"][0].detach().cpu().bool()

    h_pos = output_nn_H["hydrogen_positions"][0].detach().cpu()
    h_mask = output_nn_H["hydrogen_atom_exists"][0].detach().cpu().bool()
    h_names = output_nn_H["hydrogen_atom_names"]

    atom_serial = 1
    lines = []

    L = pos14.shape[0]

    for i in range(L):
        aa = sequence[i]
        resname = restype_1to3[aa]
        resid = i + 1
        chain_id = "A"

        atom_names = atom14_names[resname]

        for j, atom_name in enumerate(atom_names):
            if atom_name == "":
                continue

            if not bool(mask14[i, j]):
                continue

            x, y, z = pos14[i, j].tolist()
            element = atom_name[0]

            line = (
                f"ATOM  {atom_serial:5d} {atom_name:<4s} {resname:>3s} {chain_id}"
                f"{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}"
            )

            lines.append(line)
            atom_serial += 1

        for j in range(h_pos.shape[1]):
            if not bool(h_mask[i, j]):
                continue

            atom_name = h_names[i][j]
            x, y, z = h_pos[i, j].tolist()
            element = "H"

            line = (
                f"ATOM  {atom_serial:5d} {atom_name:<4s} {resname:>3s} {chain_id}"
                f"{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}"
            )

            lines.append(line)
            atom_serial += 1

    lines.append("END")

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")




# ============================================================
# AMBER PDB atom-order support
# These definitions intentionally override the simpler versions above.
# Use amber_pdb_path="protein_amber.pdb" in add_template_hydrogens_to_output_nn()
# to store an AMBER-ordered all-atom tensor inside output_nn_H.
# ============================================================

def parse_amber_pdb_atom_order(amber_pdb_path):
    residues = []
    current_key = None
    current = None

    with open(amber_pdb_path, "r") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            atom_name = line[12:16].strip()
            resname = line[17:20].strip()

            try:
                resid = int(line[22:26])
            except ValueError:
                parts = line.split()
                resid = int(parts[4])

            key = (resid, resname)

            if key != current_key:
                current = {
                    "resid": resid,
                    "resname": resname,
                    "atom_names": [],
                }
                residues.append(current)
                current_key = key

            current["atom_names"].append(atom_name)

    if len(residues) == 0:
        raise ValueError(f"No ATOM/HETATM records found in AMBER PDB: {amber_pdb_path}")

    return residues


def infer_element_from_atom_name(atom_name):
    name = atom_name.strip()

    if name == "":
        return ""

    if name[0].isdigit() and len(name) > 1:
        return name[1].upper()

    if name.startswith("Cl") or name.startswith("CL"):
        return "Cl"

    if name.startswith("Br") or name.startswith("BR"):
        return "Br"

    return name[0].upper()


def atom_name_to_z_from_amber_name(atom_name):
    element = infer_element_from_atom_name(atom_name)

    if element == "H":
        return 1
    if element == "C":
        return 6
    if element == "N":
        return 7
    if element == "O":
        return 8
    if element == "S":
        return 16

    raise ValueError(f"Unknown element for AMBER atom name: {atom_name}")


def build_atom_coordinate_map_for_residue(output_nn_H, sequence, residue_index):
    pos14 = output_nn_H["positions"][-1, 0]
    mask14 = output_nn_H["atom14_atom_exists"][0].bool()

    h_pos = output_nn_H["hydrogen_positions"][0]
    h_mask = output_nn_H["hydrogen_atom_exists"][0].bool()
    h_names = output_nn_H["hydrogen_atom_names"]

    aa = sequence[residue_index]
    resname = restype_1to3[aa]

    coord_map = {}

    for atom_index, atom_name in enumerate(atom14_names[resname]):
        if atom_name == "":
            continue

        if bool(mask14[residue_index, atom_index]):
            coord_map[atom_name] = pos14[residue_index, atom_index]

    for atom_index, atom_name in enumerate(h_names[residue_index]):
        if atom_name == "":
            continue

        if bool(h_mask[residue_index, atom_index]):
            coord_map[atom_name] = h_pos[residue_index, atom_index]

    if residue_index == len(sequence) - 1:
        if "OXT" not in coord_map and all(k in coord_map for k in ["CA", "C", "O"]):
            coord_map["OXT"] = make_terminal_oxt_position(
                ca=coord_map["CA"],
                c=coord_map["C"],
                o=coord_map["O"],
            )

    return coord_map


def make_terminal_oxt_position(ca, c, o, bond_length=1.25):
    dtype = c.dtype
    device = c.device

    axis_to_ca = safe_normalize(ca - c)
    axis_away_from_ca = -axis_to_ca

    o_dir = safe_normalize(o - c)
    perpendicular = o_dir - torch.sum(o_dir * axis_away_from_ca) * axis_away_from_ca

    if torch.linalg.norm(perpendicular).item() < 1e-6:
        e1, _ = make_perpendicular_basis(axis_away_from_ca)
        perpendicular = e1
    else:
        perpendicular = safe_normalize(perpendicular)

    angle = math.radians(60.0)

    oxt_dir = (
        math.cos(angle) * axis_away_from_ca
        - math.sin(angle) * perpendicular
    )

    return c + bond_length * safe_normalize(oxt_dir)


def add_amber_ordered_atoms_to_output_nn_H(output_nn_H, sequence, amber_pdb_path):
    amber_residues = parse_amber_pdb_atom_order(amber_pdb_path)

    if len(amber_residues) != len(sequence):
        raise ValueError(
            f"AMBER PDB has {len(amber_residues)} residues, but sequence has {len(sequence)} residues"
        )

    dtype = output_nn_H["positions"].dtype
    device = output_nn_H["positions"].device

    ordered_positions_per_residue = []
    ordered_mask_per_residue = []
    ordered_z_per_residue = []
    ordered_names_per_residue = []
    ordered_resnames = []

    max_atoms = max(len(r["atom_names"]) for r in amber_residues)
    zero_pos = torch.zeros(3, dtype=dtype, device=device)

    for residue_index, amber_residue in enumerate(amber_residues):
        expected_resname = restype_1to3[sequence[residue_index]]
        amber_resname = amber_residue["resname"]

        if amber_resname != expected_resname:
            raise ValueError(
                f"Residue mismatch at index {residue_index + 1}: "
                f"AMBER PDB has {amber_resname}, sequence expects {expected_resname}"
            )

        coord_map = build_atom_coordinate_map_for_residue(
            output_nn_H=output_nn_H,
            sequence=sequence,
            residue_index=residue_index,
        )

        residue_positions = []
        residue_mask = []
        residue_z = []
        residue_names = []

        missing_names = []

        for atom_name in amber_residue["atom_names"]:
            residue_names.append(atom_name)

            if atom_name in coord_map:
                residue_positions.append(coord_map[atom_name])
                residue_mask.append(torch.tensor(True, dtype=torch.bool, device=device))
                residue_z.append(torch.tensor(atom_name_to_z_from_amber_name(atom_name), dtype=torch.long, device=device))
            else:
                missing_names.append(atom_name)
                residue_positions.append(zero_pos)
                residue_mask.append(torch.tensor(False, dtype=torch.bool, device=device))
                residue_z.append(torch.tensor(0, dtype=torch.long, device=device))

        if len(missing_names) > 0:
            raise ValueError(
                f"Missing atoms for residue {residue_index + 1} {amber_resname}: {missing_names}. "
                f"Available generated atoms: {sorted(coord_map.keys())}"
            )

        while len(residue_positions) < max_atoms:
            residue_names.append("")
            residue_positions.append(zero_pos)
            residue_mask.append(torch.tensor(False, dtype=torch.bool, device=device))
            residue_z.append(torch.tensor(0, dtype=torch.long, device=device))

        ordered_positions_per_residue.append(torch.stack(residue_positions, dim=0))
        ordered_mask_per_residue.append(torch.stack(residue_mask, dim=0))
        ordered_z_per_residue.append(torch.stack(residue_z, dim=0))
        ordered_names_per_residue.append(residue_names)
        ordered_resnames.append(amber_resname)

    output_nn_H["amber_atom_positions"] = torch.stack(ordered_positions_per_residue, dim=0).unsqueeze(0)
    output_nn_H["amber_atom_exists"] = torch.stack(ordered_mask_per_residue, dim=0).unsqueeze(0)
    output_nn_H["amber_atom_z"] = torch.stack(ordered_z_per_residue, dim=0).unsqueeze(0)
    output_nn_H["amber_atom_names"] = ordered_names_per_residue
    output_nn_H["amber_residue_names"] = ordered_resnames
    output_nn_H["amber_reference_pdb_path"] = amber_pdb_path

    return output_nn_H


def add_template_hydrogens_to_output_nn(output_nn, sequence, amber_pdb_path=None):
    pos14 = output_nn["positions"][-1, 0]
    mask14 = output_nn["atom14_atom_exists"][0].bool()

    dtype = pos14.dtype
    device = pos14.device

    all_residue_h_positions = []
    all_residue_h_names = []

    for residue_index, aa in enumerate(sequence):
        resname = restype_1to3[aa]

        residue_h_names = []
        residue_h_positions = []

        bb_h_names, bb_geometry_type = get_backbone_hydrogen_template(
            resname=resname,
            residue_index=residue_index
        )

        if len(bb_h_names) > 0:
            h_names, h_positions = make_hydrogens_for_parent(
                pos14=pos14,
                mask14=mask14,
                sequence=sequence,
                residue_index=residue_index,
                parent_atom_name="N",
                hydrogen_names=bb_h_names,
                geometry_type=bb_geometry_type,
                ref_atom_name="CA"
            )

            residue_h_names.extend(h_names)
            residue_h_positions.extend(h_positions)

        residue_template = hydrogen_templates[resname]

        for parent_atom_name, template_data in residue_template.items():
            hydrogen_names, geometry_type, ref_atom_name = template_data

            h_names, h_positions = make_hydrogens_for_parent(
                pos14=pos14,
                mask14=mask14,
                sequence=sequence,
                residue_index=residue_index,
                parent_atom_name=parent_atom_name,
                hydrogen_names=hydrogen_names,
                geometry_type=geometry_type,
                ref_atom_name=ref_atom_name
            )

            residue_h_names.extend(h_names)
            residue_h_positions.extend(h_positions)

        all_residue_h_names.append(residue_h_names)
        all_residue_h_positions.append(residue_h_positions)

    max_h = max(len(x) for x in all_residue_h_positions)

    padded_positions_per_residue = []
    padded_mask_per_residue = []
    padded_names_per_residue = []

    zero_pos = torch.zeros(3, dtype=dtype, device=device)

    for residue_h_names, residue_h_positions in zip(
        all_residue_h_names,
        all_residue_h_positions
    ):
        padded_positions = []
        padded_mask = []
        padded_names = []

        for h_name, h_pos in zip(residue_h_names, residue_h_positions):
            padded_names.append(h_name)
            padded_positions.append(h_pos)
            padded_mask.append(torch.tensor(True, dtype=torch.bool, device=device))

        while len(padded_positions) < max_h:
            padded_names.append("")
            padded_positions.append(zero_pos)
            padded_mask.append(torch.tensor(False, dtype=torch.bool, device=device))

        padded_positions_per_residue.append(torch.stack(padded_positions, dim=0))
        padded_mask_per_residue.append(torch.stack(padded_mask, dim=0))
        padded_names_per_residue.append(padded_names)

    hydrogen_positions = torch.stack(padded_positions_per_residue, dim=0).unsqueeze(0)
    hydrogen_atom_exists = torch.stack(padded_mask_per_residue, dim=0).unsqueeze(0)

    output_nn_H = copy.copy(output_nn)

    output_nn_H["hydrogen_positions"] = hydrogen_positions
    output_nn_H["hydrogen_atom_exists"] = hydrogen_atom_exists
    output_nn_H["hydrogen_atom_names"] = padded_names_per_residue

    if amber_pdb_path is not None:
        output_nn_H = add_amber_ordered_atoms_to_output_nn_H(
            output_nn_H=output_nn_H,
            sequence=sequence,
            amber_pdb_path=amber_pdb_path,
        )

    return output_nn_H


def extract_amber_ordered_pos_z(output_nn_H):
    pos_all = output_nn_H["amber_atom_positions"]
    mask_all = output_nn_H["amber_atom_exists"].bool()
    z_all = output_nn_H["amber_atom_z"]

    pos = pos_all[mask_all]
    z = z_all[mask_all]

    return pos, z


def extract_heavy_and_template_hydrogen_pos_z(output_nn_H, z14):
    if "amber_atom_positions" in output_nn_H:
        return extract_amber_ordered_pos_z(output_nn_H)

    heavy_pos, heavy_z = extract_pos_z(output_nn_H, z14)
    h_pos, h_z = extract_hydrogen_pos_z(output_nn_H)

    all_pos = torch.cat([heavy_pos, h_pos], dim=0)
    all_z = torch.cat([heavy_z, h_z], dim=0)

    return all_pos, all_z


def format_pdb_atom_line(atom_serial, atom_name, resname, resid, x, y, z, element, chain_id=""):
    if chain_id == "":
        return (
            f"ATOM  {atom_serial:5d} {atom_name:>4s} {resname:>3s}  {resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00          {element:>2s}"
        )

    return (
        f"ATOM  {atom_serial:5d} {atom_name:>4s} {resname:>3s} {chain_id}"
        f"{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00  0.00          {element:>2s}"
    )


def write_pdb_from_amber_ordered_output(output_nn_H, filename, chain_id=""):
    atom_positions = output_nn_H["amber_atom_positions"][0].detach().cpu()
    atom_exists = output_nn_H["amber_atom_exists"][0].detach().cpu().bool()
    atom_names = output_nn_H["amber_atom_names"]
    resnames = output_nn_H["amber_residue_names"]

    atom_serial = 1
    lines = []

    for residue_index in range(atom_positions.shape[0]):
        resid = residue_index + 1
        resname = resnames[residue_index]

        for atom_index in range(atom_positions.shape[1]):
            if not bool(atom_exists[residue_index, atom_index]):
                continue

            atom_name = atom_names[residue_index][atom_index]
            element = infer_element_from_atom_name(atom_name)
            x, y, z = atom_positions[residue_index, atom_index].tolist()

            lines.append(
                format_pdb_atom_line(
                    atom_serial=atom_serial,
                    atom_name=atom_name,
                    resname=resname,
                    resid=resid,
                    x=x,
                    y=y,
                    z=z,
                    element=element,
                    chain_id=chain_id,
                )
            )
            atom_serial += 1

    lines.append("TER")
    lines.append("END")

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_pdb_with_template_hydrogens(output_nn_H, sequence, filename, amber_pdb_path=None, chain_id=""):
    if amber_pdb_path is not None and "amber_atom_positions" not in output_nn_H:
        output_nn_H = add_amber_ordered_atoms_to_output_nn_H(
            output_nn_H=output_nn_H,
            sequence=sequence,
            amber_pdb_path=amber_pdb_path,
        )

    if "amber_atom_positions" in output_nn_H:
        write_pdb_from_amber_ordered_output(
            output_nn_H=output_nn_H,
            filename=filename,
            chain_id=chain_id,
        )
        return

    pos14 = output_nn_H["positions"][-1, 0].detach().cpu()
    mask14 = output_nn_H["atom14_atom_exists"][0].detach().cpu().bool()

    h_pos = output_nn_H["hydrogen_positions"][0].detach().cpu()
    h_mask = output_nn_H["hydrogen_atom_exists"][0].detach().cpu().bool()
    h_names = output_nn_H["hydrogen_atom_names"]

    atom_serial = 1
    lines = []

    L = pos14.shape[0]

    for i in range(L):
        aa = sequence[i]
        resname = restype_1to3[aa]
        resid = i + 1

        atom_names = atom14_names[resname]

        for j, atom_name in enumerate(atom_names):
            if atom_name == "":
                continue

            if not bool(mask14[i, j]):
                continue

            x, y, z = pos14[i, j].tolist()
            element = infer_element_from_atom_name(atom_name)

            lines.append(
                format_pdb_atom_line(
                    atom_serial=atom_serial,
                    atom_name=atom_name,
                    resname=resname,
                    resid=resid,
                    x=x,
                    y=y,
                    z=z,
                    element=element,
                    chain_id=chain_id,
                )
            )
            atom_serial += 1

        for j in range(h_pos.shape[1]):
            if not bool(h_mask[i, j]):
                continue

            atom_name = h_names[i][j]
            x, y, z = h_pos[i, j].tolist()
            element = "H"

            lines.append(
                format_pdb_atom_line(
                    atom_serial=atom_serial,
                    atom_name=atom_name,
                    resname=resname,
                    resid=resid,
                    x=x,
                    y=y,
                    z=z,
                    element=element,
                    chain_id=chain_id,
                )
            )
            atom_serial += 1

    lines.append("END")

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")
