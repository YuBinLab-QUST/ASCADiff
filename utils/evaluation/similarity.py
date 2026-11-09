import numpy as np
from rdkit import Chem, DataStructs
from tqdm import tqdm

from utils.evaluation.scoring_func import tanimoto_dis_N_to_1


def tanimoto_sim(mol, ref):
    fp1 = Chem.RDKFingerprint(ref)
    fp2 = Chem.RDKFingerprint(mol)
    return DataStructs.TanimotoSimilarity(fp1,fp2)
    

def tanimoto_sim_N_to_1(mols, ref):
    sim = [tanimoto_sim(m, ref) for m in mols]
    return sim


def tanimoto_distance(mol, ref):
    similarity = tanimoto_sim(mol, ref)
    return 1 - similarity

def batched_number_of_rings(mols):
    n = []
    for m in mols:
        n.append(Chem.rdMolDescriptors.CalcNumRings(m))
    return np.array(n)


#计算多样性
def compute_pocket_diversity(mols):
    """
    计算一个 pocket 内所有分子的多样性（两两Tanimoto距离平均值）
    mols: List[RDKit Mol]
    Return: float (mean diversity)
    """
    if len(mols) < 2:
        return 0.0

    dists = []
    for i in range(len(mols)):
        for j in range(i + 1, len(mols)):
            d = tanimoto_distance(mols[i], mols[j])
            dists.append(d)

    return float(np.mean(dists))


