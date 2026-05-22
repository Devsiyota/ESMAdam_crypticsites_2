import os
import torch
import torch.nn.utils as utils
import tempfile
import esm
import torch.nn.functional as F
import numpy as np


from cryptic_utilities import *



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRECISION = torch.float32

#ESMFOLD_CACHE_DIR = "/depot/chen4116/data/deven_esmfold_checkpoints"
SEQUENCE = "LASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"

AMBER_PRMTOP = "PL_dry.prmtop"
PROTEIN_AMBER_PDB = "protein_dry.pdb"
PL_AMBER_PDB = "PL_dry.pdb"
LIGAND_PDB = "DRG.pdb"

OUTPUT_DIR = "Structures"

NUM_STEPS = 10000
LEARNING_RATE = 0.005
LATENT_NOISE_SCALE = 0.5
RESTRAINT_LIGAND_CUTOFF = 6.0

CA_WEIGHT = 10.0
RMSD_WEIGHT = 1000.0
MAX_LATENT_GRAD_NORM = 10.0

AMBER_TERMS = [
    "bonds",
    "angles",
    "dihedrals",
    "impropers",
    "1-4",
    "electrostatics",
    "lj",
]

ELEMENT_TO_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "CL": 17,
    "BR": 35,
    "I": 53,
}






def main():
    seed_everything(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_esm = load_esmfold_model()

    with torch.no_grad():
        output, esm_s_output = model_esm.infer(SEQUENCE)

    z14 = make_atom14_atomic_numbers(SEQUENCE, DEVICE)

    output_H, init_pos_all, init_z_all = get_amber_ordered_pos_z_from_output(
        output_nn=output,
        sequence=SEQUENCE,
        z14=z14,
    )

    protein_ref_pos, protein_ref_resids = read_pdb_atom_records(pdb_path=PROTEIN_AMBER_PDB,device=DEVICE,precision=PRECISION,include_hetatm=False)
    

    if init_pos_all.shape[0] != protein_ref_pos.shape[0]:
        raise RuntimeError(
            f"Protein atom count mismatch between template output and {PROTEIN_AMBER_PDB}: "
            f"{init_pos_all.shape[0]} vs {protein_ref_pos.shape[0]}"
        )

    #local_atom_mask = make_atom_mask_from_resids(resids=protein_ref_resids,ranges_spec=LOCAL_ENERGY_RANGES,device=DEVICE)

    ##print("local_atom_mask", local_atom_mask)

    ligand_pos, ligand_z, ligand_lines = read_ligand_pdb_as_pos_z(
        pdb_path=LIGAND_PDB,
        device=DEVICE,
        precision=PRECISION,
    )


    restraint_resids = find_resids_not_within_ligand(
    protein_pos=protein_ref_pos,
    protein_resids=protein_ref_resids,
    ligand_pos=ligand_pos,
    cutoff=RESTRAINT_LIGAND_CUTOFF,
    )

    RESTRAINT_RANGES_AUTO = compress_resids_to_ranges_spec(restraint_resids)

    #print("distance restraint cutoff:", RESTRAINT_LIGAND_CUTOFF)
    #print("distance  restraint residues:", restraint_resids)
    print("distance restraint ranges:", RESTRAINT_RANGES_AUTO)

    restrained_atom_mask = make_atom_mask_from_resids(
        resids=protein_ref_resids,
        ranges_spec=RESTRAINT_RANGES_AUTO,
        device=DEVICE,
    )

    restraint_keys, restraint_ref_xyz = parse_pdb_backbone_in_ranges(
        pdb_file=PROTEIN_AMBER_PDB,
        ranges_spec=RESTRAINT_RANGES_AUTO,
        require_full=False,
    )

    restraint_ref_coords = torch.tensor(
        restraint_ref_xyz,
        dtype=PRECISION,
        device=DEVICE,
    )

    pos_complex, z_complex = make_complex_pos_z(
        protein_pos=init_pos_all,
        protein_z=init_z_all,
        ligand_pos=ligand_pos,
        ligand_z=ligand_z,
    )

    #print("Protein atoms:", init_pos_all.shape[0])
    #print("Ligand atoms:", ligand_pos.shape[0])
    #print("Complex atoms:", pos_complex.shape[0])
    #print("Complex z shape:", z_complex.shape)
    #print("Restrained protein atoms:", int(restrained_atom_mask.sum().item()))
    #print("Restrained backbone atoms for RMSD:", restraint_ref_coords.shape[0])

    initial_complex_pdb = os.path.join(
        OUTPUT_DIR,
        "structure_initial_protein_H_ligand.pdb",
    )

    init_restraint_coords = select_output_backbone_by_keys(
        output_positions=output["positions"],
        ordered_keys=restraint_keys,
        step=-1,
        batch=0,
    )

    init_ligand_pos_current, init_rmsd_fit = make_ligand_follow_generated_protein(
        ligand_ref_pos=ligand_pos,
        restraint_ref_coords=restraint_ref_coords,
        restraint_current_coords=init_restraint_coords,
    )

    #print("Initial fitted restraint RMSD for ligand placement:", init_rmsd_fit.item())

    write_protein_H_ligand_pdb_using_template_writer(
        output_nn_H=output_H,
        sequence=SEQUENCE,
        ligand_lines=ligand_lines,
        output_pdb=initial_complex_pdb,
        amber_pdb_path=PROTEIN_AMBER_PDB,
        chain_id="A",
        ligand_pos_override=init_ligand_pos_current,
    )

    #print("Initial protein-H + ligand PDB:", initial_complex_pdb)

    amber_energy = build_amber_energy_backend()

    #init_protein_for_energy = make_local_energy_protein_pos(protein_pos=init_pos_all,protein_ref_pos=protein_ref_pos,local_atom_mask=local_atom_mask)

    init_pos_complex_for_energy, _ = make_complex_pos_z(
        protein_pos=init_pos_all,
        protein_z=init_z_all,
        ligand_pos=ligand_pos,
        ligand_z=ligand_z,
    )

    energy = compute_amber_energy(
        amber_energy=amber_energy,
        pos_all=init_pos_complex_for_energy,
    )

    print("Initial masked-local AMBER energy:", energy)

    initial_esm_s = LATENT_NOISE_SCALE * torch.randn_like(esm_s_output)
    initial_esm_s = initial_esm_s.detach().to(DEVICE).requires_grad_(True)

    optimizer = torch.optim.Adam(
        [initial_esm_s],
        lr=LEARNING_RATE,
    )



    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.8,
    patience=100,
    min_lr=5e-3,
    verbose=True,
    )



    best_loss = None
    best_step = None
    best_output_nn_H = None
    best_ligand_pos = None

    for step in range(NUM_STEPS):
        optimizer.zero_grad()

        output_nn, esm_s_nn = model_esm.infer(
            SEQUENCE,
            esm_s_input=initial_esm_s,
        )

        output_nn_H, protein_pos, protein_z = get_amber_ordered_pos_z_from_output(
            output_nn=output_nn,
            sequence=SEQUENCE,
            z14=z14,
        )


        restraint_coords = select_output_backbone_by_keys(
            output_positions=output_nn["positions"],
            ordered_keys=restraint_keys,
            step=-1,
            batch=0,
        )

        ligand_pos_current, rmsd_restraint = make_ligand_follow_generated_protein(
            ligand_ref_pos=ligand_pos,
            restraint_ref_coords=restraint_ref_coords,
            restraint_current_coords=restraint_coords,
        )

        #protein_pos_for_energy = make_local_energy_protein_pos(protein_pos=protein_pos,protein_ref_pos=protein_ref_pos,local_atom_mask=local_atom_mask)

        pos_complex_for_energy, z_complex = make_complex_pos_z(
            protein_pos=protein_pos,
            protein_z=protein_z,
            ligand_pos=ligand_pos_current,
            ligand_z=ligand_z,
        )

        energy = compute_amber_energy(
            amber_energy=amber_energy,
            pos_all=pos_complex_for_energy,
        )


        loss_ca = compute_local_ca_loss(
            output_positions=output_nn["positions"],
        )



        loss = energy + RMSD_WEIGHT * rmsd_restraint + CA_WEIGHT * loss_ca

        loss.backward()

        print(
            f"Step {step:04d} | "
            f"loss={loss.item():.6f} | "
            f"local_energy={energy.item():.6f} | "
            f"rmsd_restraint={rmsd_restraint.item():.6f} | "
            f"loss_ca_local={loss_ca.item():.6f}"
        )

        if initial_esm_s.grad is None:
            raise RuntimeError("initial_esm_s.grad is None")

        utils.clip_grad_norm_(
            [initial_esm_s],
            max_norm=MAX_LATENT_GRAD_NORM,
        )

        current_loss = float(loss.detach().cpu())

        if best_loss is None or current_loss < best_loss:
            best_loss = current_loss
            best_step = step
            best_output_nn_H = output_nn_H
            best_ligand_pos = ligand_pos_current.detach()

        optimizer.step()
        scheduler.step(current_loss)

        complex_pdb = os.path.join(
            OUTPUT_DIR,
            f"complex_{step:01d}.pdb",
        )

        write_protein_H_ligand_pdb_using_template_writer(
            output_nn_H=output_nn_H,
            sequence=SEQUENCE,
            ligand_lines=ligand_lines,
            output_pdb=complex_pdb,
            amber_pdb_path=PROTEIN_AMBER_PDB,
            chain_id="A",
            ligand_pos_override=ligand_pos_current,
        )

    best_complex_pdb = os.path.join(
        OUTPUT_DIR,
        f"LowestE_cpmplex{best_step:01d}.pdb",
    )

    write_protein_H_ligand_pdb_using_template_writer(
        output_nn_H=best_output_nn_H,
        sequence=SEQUENCE,
        ligand_lines=ligand_lines,
        output_pdb=best_complex_pdb,
        amber_pdb_path=PROTEIN_AMBER_PDB,
        chain_id="A",
        ligand_pos_override=best_ligand_pos,
    )

    #print("done")
    print("best_step:", best_step)
    print("best_loss:", best_loss)
    print("best_pdb:", best_complex_pdb)


if __name__ == "__main__":
    main()
