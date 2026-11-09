import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PropertyNormalizer:
    """归一化QED, SA, Vina属性"""

    def __init__(self):

        self.qed_range = (0.0, 1.0)  # QED: [0, 1]
        self.sa_range = (0.0, 1.0)  # SA
        self.vina_range = (-15.0, 0.0)  # Vina: [-15, 0]

    def normalize_qed(self, qed):
        """QED已经在[0,1]范围内"""
        return np.clip(qed, self.qed_range[0], self.qed_range[1])

    def normalize_sa(self, sa):
        """SA已经归一化"""
        return np.clip(sa, self.sa_range[0], self.sa_range[1])

    def normalize_vina(self, vina):
        """归一化Vina到[0, 1]"""
        vina_clipped = np.clip(vina, self.vina_range[0], self.vina_range[1])
        return (vina_clipped - self.vina_range[0]) / (self.vina_range[1] - self.vina_range[0])

    def normalize_properties(self, qed, sa, vina):
        """归一化所有属性"""
        return np.array([
            self.normalize_qed(qed),
            self.normalize_sa(sa),
            self.normalize_vina(vina)
        ], dtype=np.float32)



class MoleculeDatasetAdapted(Dataset):
    """

    info.pkl: {pair_key: {pid, lid, smiles, QED, SA, L_length, P_length, affinity_vina}}
    """

    def __init__(self, data_path, info_pkl_path, split='train', normalize_properties=True):


        with open(data_path, 'rb') as f:
            self.data_list = pickle.load(f)
        with open(info_pkl_path, 'rb') as f:
            self.info_data = pickle.load(f)

        self.split = split
        self.normalize_properties = normalize_properties

        if normalize_properties:
            self.normalizer = PropertyNormalizer()


        self._build_key_mapping()

        print(f"[INFO] Loaded {split} dataset with {len(self.data_list)} samples")
        print(f"[INFO] info.pkl contains {len(self.info_data)} protein-ligand pairs")
        print(f"[INFO] Successfully mapped {self.mapped_count}/{len(self.data_list)} samples")

    def _build_key_mapping(self):

        self.key_to_info = {}
        self.mapped_count = 0

        for pair_key, info in self.info_data.items():
            pid = info['pid']
            lid = info['lid']

            self.key_to_info[(pid, lid)] = info
            self.key_to_info[pair_key] = info

            simple_key = f"{pid}-{lid}"
            self.key_to_info[simple_key] = info

    def _get_info_by_keys(self, data):


        if 'pair_key' in data:
            pair_key = data['pair_key']
            if pair_key in self.key_to_info:
                return self.key_to_info[pair_key]


        if 'pid' in data and 'lid' in data:
            key = (data['pid'], data['lid'])
            if key in self.key_to_info:
                return self.key_to_info[key]


            simple_key = f"{data['pid']}-{data['lid']}"
            if simple_key in self.key_to_info:
                return self.key_to_info[simple_key]


        if 'protein_id' in data and 'ligand_id' in data:
            key = (data['protein_id'], data['ligand_id'])
            if key in self.key_to_info:
                return self.key_to_info[key]

        return None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx].copy()

        info = self._get_info_by_keys(data)

        if info is None:

            print(f"[WARNING] Cannot find info for sample {idx}, using defaults")
            smiles = 'C'
            qed = 0.5
            sa = 0.5
            vina = -5.0
        else:
  
            smiles = info.get('smiles', 'C')
            qed = info.get('QED', 0.5)
            sa = info.get('SA', 0.5)
            vina = info.get('affinity_vina', -5.0)
            self.mapped_count += 1 if not hasattr(self, '_counted') else 0


        if self.normalize_properties:
            properties = self.normalizer.normalize_properties(qed, sa, vina)
        else:
            properties = np.array([qed, sa, vina], dtype=np.float32)


        data['smiles'] = smiles
        data['properties'] = properties
        data['raw_properties'] = {
            'QED': qed,
            'SA': sa,
            'Vina': vina
        }

        return data



def collate_fn(batch):

   
    protein_pos_list = [item['protein_pos'] for item in batch]
    protein_v_list = [item['protein_v'] for item in batch]
    ligand_pos_list = [item['ligand_pos'] for item in batch]
    ligand_v_list = [item['ligand_v'] for item in batch]
    smiles_list = [item['smiles'] for item in batch]
    properties_list = [item['properties'] for item in batch]

  
    batch_protein = []
    batch_ligand = []

    for i, (protein_pos, ligand_pos) in enumerate(zip(protein_pos_list, ligand_pos_list)):
        batch_protein.extend([i] * len(protein_pos))
        batch_ligand.extend([i] * len(ligand_pos))

    protein_pos = torch.cat([torch.FloatTensor(pos) for pos in protein_pos_list], dim=0)
    protein_v = torch.cat([torch.FloatTensor(v) for v in protein_v_list], dim=0)
    ligand_pos = torch.cat([torch.FloatTensor(pos) for pos in ligand_pos_list], dim=0)
    ligand_v = torch.cat([torch.LongTensor(v) for v in ligand_v_list], dim=0)
    batch_protein = torch.LongTensor(batch_protein)
    batch_ligand = torch.LongTensor(batch_ligand)
    properties = torch.FloatTensor(properties_list)

    return {
        'protein_pos': protein_pos,
        'protein_v': protein_v,
        'batch_protein': batch_protein,
        'ligand_pos': ligand_pos,
        'ligand_v': ligand_v,
        'batch_ligand': batch_ligand,
        'smiles': smiles_list,
        'properties': properties
    }



def create_dataloaders(train_data_path, valid_data_path, info_pkl_path,
                       batch_size=8, num_workers=4):


    train_dataset = MoleculeDatasetAdapted(
        train_data_path, info_pkl_path, split='train'
    )
    valid_dataset = MoleculeDatasetAdapted(
        valid_data_path, info_pkl_path, split='valid'
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader



def convert_data_format(old_data_path, info_pkl_path, output_path):


    # 加载数据
    with open(old_data_path, 'rb') as f:
        data_list = pickle.load(f)

    with open(info_pkl_path, 'rb') as f:
        info_data = pickle.load(f)


    updated_data = []
    for data in data_list:

        if 'pid' in data and 'lid' in data:

            for pair_key, info in info_data.items():
                if info['pid'] == data['pid'] and info['lid'] == data['lid']:
                    data['pair_key'] = pair_key
                    break
        updated_data.append(data)


    with open(output_path, 'wb') as f:
        pickle.dump(updated_data, f)

    print(f"[INFO] Updated data saved to {output_path}")



def validate_data_info_mapping(data_path, info_pkl_path):




    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)

    with open(info_pkl_path, 'rb') as f:
        info_data = pickle.load(f)

    print(f"[INFO] Validating {len(data_list)} samples...")


    matched = 0
    unmatched = []

    for idx, data in enumerate(data_list):

        found = False


        if 'pair_key' in data and data['pair_key'] in info_data:
            found = True


        if not found and 'pid' in data and 'lid' in data:
            for pair_key, info in info_data.items():
                if info['pid'] == data['pid'] and info['lid'] == data['lid']:
                    found = True
                    break

        if found:
            matched += 1
        else:
            unmatched.append(idx)


    print(f"\n[VALIDATION RESULTS]")
    print(f"Total samples: {len(data_list)}")
    print(f"Matched: {matched} ({matched / len(data_list) * 100:.2f}%)")
    print(f"Unmatched: {len(unmatched)} ({len(unmatched) / len(data_list) * 100:.2f}%)")

    if unmatched:
        print(f"\nFirst 10 unmatched sample indices: {unmatched[:10]}")
        print("\nExample unmatched sample:")
        print(data_list[unmatched[0]])


    print(f"\n[INFO.PKL FORMAT]")
    print(f"Total pairs: {len(info_data)}")
    print("\nFirst 3 keys:")
    for i, key in enumerate(list(info_data.keys())[:3]):
        print(f"  {i + 1}. {key}")

    print("\nFirst pair structure:")
    first_key = list(info_data.keys())[0]
    print(f"Key: {first_key}")
    print(f"Value: {info_data[first_key]}")



if __name__ == '__main__':

    print("=" * 60)
    print("STEP 1: Validating data-info mapping")
    print("=" * 60)
    validate_data_info_mapping(
        data_path='./data/train_data.pkl',
        info_pkl_path='./data/info.pkl'
    )


    print("\n" + "=" * 60)
    print("STEP 2: Creating dataloaders")
    print("=" * 60)
    train_loader, valid_loader = create_dataloaders(
        train_data_path='./data/train_data.pkl',
        valid_data_path='./data/valid_data.pkl',
        info_pkl_path='./data/info.pkl',
        batch_size=8
    )


    print("\n" + "=" * 60)
    print("STEP 3: Testing data loading")
    print("=" * 60)
    batch = next(iter(train_loader))

    print(f"Batch keys: {batch.keys()}")
    print(f"Protein pos shape: {batch['protein_pos'].shape}")
    print(f"Ligand pos shape: {batch['ligand_pos'].shape}")
    print(f"SMILES list length: {len(batch['smiles'])}")
    print(f"Properties shape: {batch['properties'].shape}")
    print(f"\nFirst SMILES: {batch['smiles'][0]}")
    print(f"First properties: {batch['properties'][0]}")

    print("\n[SUCCESS] Data loading works correctly!")



def analyze_info_pkl(info_pkl_path):

    with open(info_pkl_path, 'rb') as f:
        info_data = pickle.load(f)


    qed_values = [info['QED'] for info in info_data.values() if 'QED' in info]
    sa_values = [info['SA'] for info in info_data.values() if 'SA' in info]
    vina_values = [info['affinity_vina'] for info in info_data.values() if 'affinity_vina' in info]
    l_length_values = [info['L_length'] for info in info_data.values() if 'L_length' in info]
    p_length_values = [info['P_length'] for info in info_data.values() if 'P_length' in info]

    print(f"\n[INFO.PKL STATISTICS]")
    print(f"Total pairs: {len(info_data)}")
    print(f"\nQED:")
    print(f"  Min: {min(qed_values):.4f}")
    print(f"  Max: {max(qed_values):.4f}")
    print(f"  Mean: {np.mean(qed_values):.4f}")
    print(f"  Std: {np.std(qed_values):.4f}")

    print(f"\nSA:")
    print(f"  Min: {min(sa_values):.4f}")
    print(f"  Max: {max(sa_values):.4f}")
    print(f"  Mean: {np.mean(sa_values):.4f}")
    print(f"  Std: {np.std(sa_values):.4f}")

    print(f"\nVina (affinity):")
    print(f"  Min: {min(vina_values):.4f}")
    print(f"  Max: {max(vina_values):.4f}")
    print(f"  Mean: {np.mean(vina_values):.4f}")
    print(f"  Std: {np.std(vina_values):.4f}")

    print(f"\nLigand length:")
    print(f"  Min: {min(l_length_values)}")
    print(f"  Max: {max(l_length_values)}")
    print(f"  Mean: {np.mean(l_length_values):.1f}")

    print(f"\nProtein length:")
    print(f"  Min: {min(p_length_values)}")
    print(f"  Max: {max(p_length_values)}")
    print(f"  Mean: {np.mean(p_length_values):.1f}")