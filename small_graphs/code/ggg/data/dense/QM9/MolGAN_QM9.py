import os
import pickle
from warnings import warn

import numpy as np
import subprocess as sp

from torch.utils.data.dataset import Dataset

from ggg.utils.utils import kcycles as kcl
from ggg.data.dense.utils.helpers import _data_helper

try:
    from rdkit import Chem
except:
    warn(f"No rdkit found, won't be able to use molgan")
from datetime import datetime

import torch


class QM9preprocess(Dataset):
    def __init__(
        self,
        data_dir=None,
        filename=None,
        size=5000,
        name="5k",
        k_=None,
        gdb_sdf_file="gdb9.sdf",
        create_rand=False,
    ):
        super().__init__()
        if data_dir is None:
            data_dir = "molganQM9"
        self.Py_data = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        if filename is None:
            filename = "QM9_{}.sparsedataset".format(name)
        self.filename = filename
        self.gdb_sdf_file = gdb_sdf_file
        self.size = size
        self.k_ = k_
        self.name = name

        if not (
            os.path.isfile(os.path.join(self.data_dir, self.filename))
            and self.filename.endswith(".sparsedataset")
        ):
            if not os.path.exists(os.path.join(self.data_dir, self.gdb_sdf_file)):
                curdir = os.getcwd()
                os.makedirs(self.data_dir, exist_ok=True)
                os.chdir(self.data_dir)
                # HACK: should not usually do relative paths but
                script_path = os.path.join(
                    os.path.dirname(__file__), "download_dataset.sh"
                )
                sp.run(["bash", script_path])
                os.chdir(curdir)
            adapted = True
            self.generate(
                filters=lambda x: x.GetNumAtoms() <= 9,
                size=self.size,
                adapted=adapted,
                k_=self.k_,
                create_rand=create_rand,
            )
            self.save(self.data_dir, self.filename)
        else:
            self.load(self.data_dir, self.filename)

    def save(self, data_dir, filename):
        with open(os.path.join(data_dir, filename), "wb") as f:
            pickle.dump(self.Py_data, f)

    def load(self, data_dir, filename):
        with open(os.path.join(data_dir, filename), "rb") as f:
            self.Py_data = pickle.load(f)

    def generate(
        self,
        add_h=False,
        filters=lambda x: True,
        size=None,
        adapted=False,
        k_=None,
        create_rand=False,
    ):
        filename = os.path.join(self.data_dir, self.gdb_sdf_file)
        self.data = list(filter(lambda x: x is not None, Chem.SDMolSupplier(filename)))
        self.log("Extracting {}..".format(filename))

        self.data = list(map(Chem.AddHs, self.data)) if add_h else self.data
        self.data = list(filter(filters, self.data))
        if create_rand:
            rand_graph_list = []
            rand_choice = np.random.choice(len(self.data[:-5000]), size, replace=False)
            for num in rand_choice:
                rand_graph_list.append(self.data[:-5000][num])
            self.data = rand_graph_list
        if not create_rand and size is not None:
            self.data = self.data[-size:]

        self.log(
            "Extracted {} out of {} molecules {}adding Hydrogen!".format(
                len(self.data),
                len(Chem.SDMolSupplier(filename)),
                "" if add_h else "not ",
            )
        )

        self._generate_encoders_decoders()
        self._generate_AX()

        self.data = np.array(self.data)
        self.data_A = np.stack(self.data_A)
        self.data_X = np.stack(self.data_X)

        if adapted:
            self.data_A = self.data_A.clip(0, 1)

        for idx in range(len(self.data_A)):
            graph = _data_helper()
            graph.x = torch.tensor(self.data_X[idx])
            graph.A = torch.tensor(self.data_A[idx])
            if k_ is not None:

                Cycle_C = kcl()
                kcycles = Cycle_C.k_cycles(graph.A)
                if k_ == 4:
                    if (
                        kcycles[0] == 0
                        and kcycles[1] > 0
                        and kcycles[2] == 0
                        and kcycles[3] == 0
                    ):
                        self.Py_data.append(graph)
                if k_ == 5:
                    if (
                        kcycles[0] == 0
                        and kcycles[1] == 0
                        and kcycles[2] > 0
                        and kcycles[3] == 0
                    ):
                        self.Py_data.append(graph)
                if k_ == 6:
                    if (
                        kcycles[0] == 0
                        and kcycles[1] == 0
                        and kcycles[2] == 0
                        and kcycles[3] > 0
                    ):
                        self.Py_data.append(graph)
            else:
                self.Py_data.append(graph)

    def _generate_encoders_decoders(self):
        self.log("Creating atoms encoder and decoder..")
        atom_labels = sorted(
            set(
                [atom.GetAtomicNum() for mol in self.data for atom in mol.GetAtoms()]
                + [0]
            )
        )
        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        self.atom_num_types = len(atom_labels)
        self.log(
            "Created atoms encoder and decoder with {} atom types and 1 PAD symbol!".format(
                self.atom_num_types - 1
            )
        )

        self.log("Creating bonds encoder and decoder..")
        bond_labels = [Chem.rdchem.BondType.ZERO] + list(
            sorted(
                set(bond.GetBondType() for mol in self.data for bond in mol.GetBonds())
            )
        )

        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        self.bond_num_types = len(bond_labels)
        self.log(
            "Created bonds encoder and decoder with {} bond types and 1 PAD symbol!".format(
                self.bond_num_types - 1
            )
        )

    def _generate_AX(self):
        self.log("Creating features and adjacency matrices..")

        data = []
        data_A = []
        data_X = []

        max_length = max(mol.GetNumAtoms() for mol in self.data)
        max_length_s = max(len(Chem.MolToSmiles(mol)) for mol in self.data)

        for i, mol in enumerate(self.data):
            A = self._genA(mol, connected=True, max_length=max_length)
            D = np.count_nonzero(A, -1)
            if A is not None:
                data.append(mol)
                data_A.append(A)
                data_X.append(self._genX(mol, max_length=max_length))

        self.log(date=False)
        self.log(
            "Created {} features and adjacency matrices  out of {} molecules!".format(
                len(data), len(self.data)
            )
        )

        self.data = data
        self.data_A = data_A
        self.data_X = data_X
        self.length = len(self.data)

    def _genA(self, mol, connected=True, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)

        begin, end = (
            [b.GetBeginAtomIdx() for b in mol.GetBonds()],
            [b.GetEndAtomIdx() for b in mol.GetBonds()],
        )
        bond_type = [self.bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]

        A[begin, end] = bond_type
        A[end, begin] = bond_type

        degree = np.sum(A[: mol.GetNumAtoms(), : mol.GetNumAtoms()], axis=-1)

        return A if connected and (degree > 0).all() else None

    def _genX(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        return np.array(
            [self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
            + [0] * (max_length - mol.GetNumAtoms()),
            dtype=np.int32,
        )

    @staticmethod
    def log(msg="", date=True):
        print(
            str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + " " + str(msg)
            if date
            else str(msg)
        )

    def __len__(self):
        return len(self.Py_data)

    def __getitem__(self, item):
        g = self.Py_data[0]
        return g.x, g.A
