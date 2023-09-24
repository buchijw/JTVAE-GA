import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
import sascorer
import networkx as nx


def penalized_logp_standard(mol):
    """
    Calculates the penalized logP standard score for a given molecule.

    Args:
    - mol: A molecule object.

    Returns:
    - float: The penalized logP standard score.

    Notes:
    These values were calculated based on the MOSES training dataset.
    If using another dataset, adjust these numbers.
    """

    logP_mean = 2.4399606244103639873799239
    logP_std = 0.9293197802518905481505840
    SA_mean = -2.4485512208785431553792478
    SA_std = 0.4603110476923852334429910
    cycle_mean = -0.0307270378623088931402396
    cycle_std = 0.2163675785228087178335699

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    standardized_log_p = (log_p - logP_mean) / logP_std
    standardized_SA = (SA - SA_mean) / SA_std
    standardized_cycle = (cycle_score - cycle_mean) / cycle_std
    return standardized_log_p + standardized_SA + standardized_cycle