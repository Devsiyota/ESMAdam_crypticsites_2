import sys
import torch

from moleculekit.molecule import Molecule
from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
from torchmd.systems import System
from torchmd.forces import Forces


class TorchMDEnergyWithForceGrad(torch.autograd.Function):
    """
    Custom autograd wrapper for TorchMD classic force fields.

    TorchMD classic force computation gives us forces directly.

    Force relation:

    F = -dE/dx

    Therefore:

    dE/dx = -F

    This wrapper returns energy as a torch scalar and supplies the correct
    coordinate gradient during backward().
    """

    @staticmethod
    def forward(ctx, pos, box, forces_object):
        force_buffer = torch.zeros_like(pos)

        E_raw = forces_object.compute(
            pos,
            box,
            force_buffer,
        )

        energy_float = energy_to_float(E_raw)

        ctx.save_for_backward(force_buffer)

        energy_tensor = torch.tensor(
            energy_float,
            dtype=pos.dtype,
            device=pos.device,
        )

        return energy_tensor

    @staticmethod
    def backward(ctx, grad_output):
        force_buffer, = ctx.saved_tensors

        grad_pos = -force_buffer
        grad_pos = grad_output * grad_pos

        grad_box = None
        grad_forces_object = None

        return grad_pos, grad_box, grad_forces_object


def value_to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().sum())

    if isinstance(x, float) or isinstance(x, int):
        return float(x)

    raise TypeError(f"Unsupported value type: {type(x)}")


def energy_to_float(E):
    if isinstance(E, list):
        total = 0.0

        for value in E:
            total += value_to_float(value)

        return total

    if isinstance(E, dict):
        total = 0.0

        for key, value in E.items():
            total += value_to_float(value)

        return total

    return value_to_float(E)


def print_energy_details(E):
    print("type(E):", type(E))

    if isinstance(E, list):
        print("len(E):", len(E))

        total = 0.0

        for i, value in enumerate(E):
            e_i = value_to_float(value)
            total += e_i
            print(f"energy_item_{i}:", e_i, "| type:", type(value))

        print("total_energy_sum:", total)

    elif isinstance(E, dict):
        total = 0.0

        for key, value in E.items():
            e_i = value_to_float(value)
            total += e_i
            print(key, e_i, "| type:", type(value))

        print("total_energy_sum:", total)

    else:
        print("energy:", value_to_float(E), "| type:", type(E))


class TorchMDAmberEnergy:
    """
    Energy object for one AMBER prmtop + PDB coordinate system.

    This class hides all TorchMD setup from the optimization code.
    The optimizer should only see:
    - self.pos
    - self.box
    - self.energy(pos)
    - self.raw_energy_and_force(pos)
    - self.write_pdb(pos, output_pdb)
    """

    def __init__(
        self,
        prmtop,
        pdb_file,
        device=None,
        precision=torch.float32,
        terms=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if terms is None:
            terms = [
                "bonds",
                "angles",
                "dihedrals",
                "impropers",
                "1-4",
                "electrostatics",
                "lj",
            ]

        self.prmtop = prmtop
        self.pdb_file = pdb_file
        self.device = device
        self.precision = precision
        self.terms = terms

        self.mol = Molecule()
        self.mol.read(prmtop)

        self.natoms_from_prmtop = self.mol.numAtoms

        self.mol.read(pdb_file)

        self._validate_molecule()

        self.ff = ForceField.create(self.mol, prmtop)

        self.parameters = Parameters(
            self.ff,
            self.mol,
            precision=precision,
            device=device,
        )

        self.system = System(
            self.mol.numAtoms,
            nreplicas=1,
            precision=precision,
            device=device,
        )

        self.system.set_positions(self.mol.coords)
        self.system.set_box(self.mol.box)

        self.forces_object = Forces(
            self.parameters,
            terms=terms,
        )

        self.pos = self.system.pos.detach().clone().to(device)
        self.box = self.system.box.detach().clone().to(device)

    def _validate_molecule(self):
        if self.mol.numAtoms != self.natoms_from_prmtop:
            raise ValueError(
                "Atom count changed after reading PDB. "
                f"Atoms from prmtop: {self.natoms_from_prmtop}; "
                f"atoms after PDB read: {self.mol.numAtoms}. "
                "Your PDB is not compatible with this prmtop."
            )

        if self.mol.coords is None:
            raise ValueError("No coordinates were loaded from PDB.")

        if self.mol.coords.shape[0] != self.mol.numAtoms:
            raise ValueError(
                "Coordinate atom count does not match molecule atom count. "
                f"mol.coords shape: {self.mol.coords.shape}; "
                f"mol.numAtoms: {self.mol.numAtoms}"
            )

    def print_system_info(self):
        print("device:", self.device)
        print("prmtop:", self.prmtop)
        print("pdb:", self.pdb_file)
        print("terms:", self.terms)
        print("atoms from prmtop:", self.natoms_from_prmtop)
        print("atoms after pdb read:", self.mol.numAtoms)
        print("box:", self.mol.box)
        print("coords shape:", self.mol.coords.shape)

    def energy(self, pos):
        return TorchMDEnergyWithForceGrad.apply(
            pos,
            self.box,
            self.forces_object,
        )

    def raw_energy_and_force(self, pos):
        force_buffer = torch.zeros_like(pos)

        E_raw = self.forces_object.compute(
            pos,
            self.box,
            force_buffer,
        )

        return E_raw, force_buffer

    def check_energy_and_force(self, pos, label="Energy/force check"):
        print()
        print(label)

        E_raw, force_buffer = self.raw_energy_and_force(pos)

        print_energy_details(E_raw)

        print("force tensor shape:", force_buffer.shape)
        print("force norm:", float(torch.linalg.norm(force_buffer).detach().cpu()))
        print("force min:", float(force_buffer.detach().cpu().min()))
        print("force max:", float(force_buffer.detach().cpu().max()))

        if torch.isnan(force_buffer).any():
            print("WARNING: forces contain NaN")
        else:
            print("No NaN in forces")

        if torch.isinf(force_buffer).any():
            print("WARNING: forces contain Inf")
        else:
            print("No Inf in forces")

        return E_raw, force_buffer

    def check_custom_backward(self, pos):
        print()
        print("Testing custom autograd wrapper:")

        if not pos.requires_grad:
            raise ValueError("pos.requires_grad must be True for backward test.")

        if pos.grad is not None:
            pos.grad.zero_()

        test_energy = self.energy(pos)

        print("test_energy:", float(test_energy.detach().cpu()))
        print("test_energy requires_grad:", test_energy.requires_grad)
        print("test_energy grad_fn:", test_energy.grad_fn)

        test_energy.backward()

        if pos.grad is None:
            raise RuntimeError("Custom backward did not create pos.grad.")

        print("test pos.grad norm:", float(torch.linalg.norm(pos.grad).detach().cpu()))

        pos.grad.zero_()

    def write_pdb(self, pos, output_pdb):
        optimized_coords = pos.detach().cpu().numpy()[0]
        self.mol.coords = optimized_coords.reshape(self.mol.numAtoms, 3, 1)
        self.mol.write(output_pdb)
        print("Wrote structure:", output_pdb)


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("python torchmd_amber_energy.py system.prmtop coordinates.pdb")
        sys.exit(1)

    prmtop = sys.argv[1]
    pdb_file = sys.argv[2]

    amber_energy = TorchMDAmberEnergy(
        prmtop=prmtop,
        pdb_file=pdb_file,
    )

    amber_energy.print_system_info()

    pos = amber_energy.pos.detach().clone()
    pos.requires_grad_(True)

    amber_energy.check_energy_and_force(
        pos,
        label="Initial TorchMD energy/force check",
    )

    amber_energy.check_custom_backward(pos)

    print()
    print("DONE")


if __name__ == "__main__":
    main()
