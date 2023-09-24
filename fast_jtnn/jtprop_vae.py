import torch
import torch.nn as nn
import torch.nn.functional as F
from mol_tree import Vocab, MolTree
from nnutils import create_var, flatten_tensor, avg_pool
from jtnn_enc import JTNNEncoder
from jtnn_dec import JTNNDecoder
from mpn import MPN
from jtmpn import JTMPN
from datautils import tensorize

from chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols
import rdkit
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import copy, math
from tqdm.auto import tqdm

class JTPropVAE(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, depthT, depthG):
        super(JTPropVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size = int(latent_size / 2) #Tree and Mol has two vectors

        self.jtnn = JTNNEncoder(hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size))
        self.decoder = JTNNDecoder(vocab, hidden_size, latent_size, nn.Embedding(vocab.size(), hidden_size))

        self.jtmpn = JTMPN(hidden_size, depthG)
        self.mpn = MPN(hidden_size, depthG)

        self.A_assm = nn.Linear(latent_size, hidden_size, bias=False)
        # self.assm_loss = nn.CrossEntropyLoss(size_average=False)
        self.assm_loss = nn.CrossEntropyLoss(reduction='sum')

        self.T_mean = nn.Linear(hidden_size, latent_size)
        self.T_var = nn.Linear(hidden_size, latent_size)
        self.G_mean = nn.Linear(hidden_size, latent_size)
        self.G_var = nn.Linear(hidden_size, latent_size)
        
        # Prop
        self.propNN = nn.Sequential(
                nn.Linear(self.latent_size*2, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 1)
        )
        self.prop_loss = nn.MSELoss()

    def encode(self, jtenc_holder, mpn_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        return tree_vecs, tree_mess, mol_vecs
    
    def encode_from_smiles(self, smiles_list):
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        return torch.cat([tree_vecs, mol_vecs], dim=-1)
    
    def encode_from_smiles_to_mean(self, smiles_list):
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        return torch.cat([tree_mean, mol_mean], dim=1)
    
    def encode_from_smiles_to_latent(self, smiles_list):
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        tree_var = -torch.abs(self.T_var(tree_vecs))
        mol_var = -torch.abs(self.G_var(mol_vecs))
        return torch.cat([tree_mean, mol_mean], dim=1), torch.cat([tree_var, mol_var], dim=1)
    
    def encode_and_samples_from_smiles(self, smiles_list):
        tree_batch=[]
        for s in smiles_list:
            mol_tree = MolTree(s)
            mol_tree.recover()
            mol_tree.assemble()
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)
            del mol_tree.mol
            for node in mol_tree.nodes:
                del node.mol
            tree_batch.append(mol_tree)
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        z_tree_vecs,_ = self.rsample(tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs,_ = self.rsample(mol_vecs, self.G_mean, self.G_var)
        return torch.cat([z_tree_vecs, z_mol_vecs], dim=1)
    
    def recon_eval(self, smiles_list,encode_times,decode_times):
        tree_batch=[]
        batch_size=len(smiles_list)
        for s in smiles_list:
            mol_tree = MolTree(s)
            mol_tree.recover()
            mol_tree.assemble()
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)
            del mol_tree.mol
            for node in mol_tree.nodes:
                del node.mol
            tree_batch.append(mol_tree)
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        
        tree_means = self.T_mean(tree_vecs)
        mol_means = self.G_mean(mol_vecs)
        tree_vars = -torch.abs(self.T_var(tree_vecs))
        mol_vars = -torch.abs(self.G_var(mol_vecs))
        
        chunk_result = [[] for _ in range(batch_size)]
        for _encode in range(encode_times):
            epsilon = create_var(torch.randn_like(tree_means), False)
            tree_vecs = tree_means + torch.exp(tree_vars / 2) * epsilon
            epsilon = create_var(torch.randn_like(mol_means), False)
            mol_vecs = mol_means + torch.exp(mol_vars / 2) * epsilon
            encode_id, encode_total = _encode + 1, encode_times
            for _decode in tqdm(list(range(decode_times)),
                'encode %d/%d decode' % (encode_id, encode_total)
                    ):
                _result=[]
                for i in range(batch_size):
                    tree_vec = tree_vecs[i].detach().reshape(1,-1)
                    mol_vec = mol_vecs[i].detach().reshape(1,-1)
                    assert tree_vec.shape[1] == mol_vec.shape[1] == self.latent_size
                    s = self.decode(tree_vec, mol_vec, prob_decode=False)
                    if Chem.MolFromSmiles(s) is None:
                        s = 'JUNK'+s
                    _result.append(s)
                for index, s in enumerate(_result):
                    chunk_result[index].append(s)
        return chunk_result

    def encode_latent(self, jtenc_holder, mpn_holder):
        tree_vecs, _ = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        tree_var = -torch.abs(self.T_var(tree_vecs))
        mol_var = -torch.abs(self.G_var(mol_vecs))
        return torch.cat([tree_mean, mol_mean], dim=1), torch.cat([tree_var, mol_var], dim=1)

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def sample_prior(self, prob_decode=False):
        z_tree = torch.randn(1, self.latent_size).cuda()
        z_mol = torch.randn(1, self.latent_size).cuda()
        return self.decode(z_tree, z_mol, prob_decode)

    def forward(self, x_batch, beta):
        """
        Performs the forward pass of the model.

        Args:
            x_batch (tuple): A tuple containing the input batch data.
            beta (float): The weight for the KL divergence term.

        Returns:
            tuple: A tuple containing the following elements:
                - float: The total loss of the function.
                - float: The KL divergence.
                - float: The word accuracy.
                - float: The topo accuracy.
                - float: The assm accuracy.
                - float: The prop loss.
        """
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder, prop_batch = x_batch
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
        z_tree_vecs,tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs,mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)

        kl_div = tree_kl + mol_kl
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_batch, z_tree_vecs)
        assm_loss, assm_acc = self.assm(x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess)
        
        all_vec = torch.cat([z_tree_vecs, z_mol_vecs], dim=1)
        # prop_label = create_var(torch.Tensor(prop_batch))
        prop_label = create_var(torch.Tensor(prop_batch))
        prop_loss = self.prop_loss(self.propNN(all_vec).squeeze(), prop_label)

        return word_loss + topo_loss + assm_loss + beta * kl_div + prop_loss, kl_div.item(), word_acc, topo_acc, assm_acc, prop_loss.item()

    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, x_tree_mess):
        jtmpn_holder,batch_idx = jtmpn_holder
        fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
        batch_idx = create_var(batch_idx)

        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)

        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)
        x_mol_vecs = self.A_assm(x_mol_vecs) #bilinear
        scores = torch.bmm(
                x_mol_vecs.unsqueeze(1),
                cand_vecs.unsqueeze(-1)
        ).squeeze()
        
        cnt,tot,acc = 0,0,0
        all_loss = []
        for i,mol_tree in enumerate(mol_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]))
                all_loss.append( self.assm_loss(cur_score.view(1,-1), label) )
        
        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode):
        #currently do not support batch decoding
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        pred_root,pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode)
        if len(pred_nodes) == 0: return None
        elif len(pred_nodes) == 1: return pred_root.smiles

        #Mark nid & is_leaf & atommap
        for i,node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder,mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _,tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict) #Important: tree_mess is a matrix, mess_dict is a python dict

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze() #bilinear

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol,_ = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=True)
        if cur_mol is None: 
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}
            cur_mol,pre_mol = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=False)
            if cur_mol is None: cur_mol = pre_mol

        if cur_mol is None: 
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None
        
    def dfs_assemble(self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node, prob_decode, check_aroma):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid,a2,a1) for nid,a1,a2 in fa_amap if nid == cur_node.nid]
        cands,aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles,cand_amap = zip(*cands)
        if torch.cuda.is_available():
            aroma_score = torch.Tensor(aroma_score).cuda()
        else:
            aroma_score = torch.Tensor(aroma_score)
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
            cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = torch.Tensor([1.0])

        if prob_decode:
            probs = F.softmax(scores.view(1,-1), dim=1).squeeze() + 1e-7 #prevent prob = 0
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _,cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id,ctr_atom,nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap) #father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue
            
            has_error = False
            for nei_node in children:
                if nei_node.is_leaf: continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(y_tree_mess, x_mol_vecs, all_nodes, cur_mol, new_global_amap, pred_amap, nei_node, cur_node, prob_decode, check_aroma)
                if tmp_mol is None: 
                    has_error = True
                    if i == 0: pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error: return cur_mol, cur_mol

        return None, pre_mol
    
    # def optimize(self, smiles, sim_cutoff, lr=2.0, num_iter=20):
    #     # mol_tree = MolTree(smiles)
    #     # mol_tree.recover()
    #     mol_tree = MolTree(smiles)
    #     mol_tree.recover()
    #     mol_tree.assemble()
    #     for node in mol_tree.nodes:
    #         if node.label not in node.cands:
    #             node.cands.append(node.label)
    #     del mol_tree.mol
    #     for node in mol_tree.nodes:
    #         del node.mol
    #     tree_batch = [mol_tree]
    #     _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
    #     tree_vec, _, mol_vec = self.encode(jtenc_holder, mpn_holder)
        
    #     mol = Chem.MolFromSmiles(smiles)
    #     fp1 = AllChem.GetMorganFingerprint(mol, 2)
        
    #     tree_mean = self.T_mean(tree_vec)
    #     tree_log_var = -torch.abs(self.T_var(tree_vec)) #Following Mueller et al.
    #     mol_mean = self.G_mean(mol_vec)
    #     mol_log_var = -torch.abs(self.G_var(mol_vec)) #Following Mueller et al.
    #     mean = torch.cat([tree_mean, mol_mean], dim=1)
    #     log_var = torch.cat([tree_log_var, mol_log_var], dim=1)
    #     cur_vec = create_var(mean.data, True)

    #     visited = []
    #     for step in range(num_iter):
    #         prop_val = self.propNN(cur_vec).squeeze()
    #         grad = torch.autograd.grad(prop_val, cur_vec)[0]
    #         cur_vec = cur_vec.data + lr * grad.data
    #         cur_vec = create_var(cur_vec, True)
    #         visited.append(cur_vec)
        
    #     l,r = 0, num_iter - 1
    #     while l < r - 1:
    #         mid = (l + r) // 2
    #         new_vec = visited[mid]
    #         tree_vec,mol_vec = torch.chunk(new_vec, 2, dim=1)
    #         new_smiles = self.decode(tree_vec, mol_vec, prob_decode=False)
    #         if new_smiles is None:
    #             r = mid - 1
    #             continue

    #         new_mol = Chem.MolFromSmiles(new_smiles)
    #         fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
    #         sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
    #         if sim < sim_cutoff:
    #             r = mid - 1
    #         else:
    #             l = mid
    #     """
    #     best_vec = visited[0]
    #     for new_vec in visited:
    #         tree_vec,mol_vec = torch.chunk(new_vec, 2, dim=1)
    #         new_smiles = self.decode(tree_vec, mol_vec, prob_decode=False)
    #         if new_smiles is None: continue
    #         new_mol = Chem.MolFromSmiles(new_smiles)
    #         fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
    #         sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
    #         if sim >= sim_cutoff:
    #             best_vec = new_vec
    #     """
    #     tree_vec,mol_vec = torch.chunk(visited[l], 2, dim=1)
    #     #tree_vec,mol_vec = torch.chunk(best_vec, 2, dim=1)
    #     new_smiles = self.decode(tree_vec, mol_vec, prob_decode=False)
    #     if new_smiles is None:
    #         return None, None
    #     new_mol = Chem.MolFromSmiles(new_smiles)
    #     fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
    #     sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
    #     if sim >= sim_cutoff:
    #         return new_smiles, sim
    #     else:
    #         return None, None
        
    def optimize(self, smiles, sim_cutoff=0.0, lr=2.0, num_iter=20):
        # mol_tree = MolTree(smiles)
        # mol_tree.recover()
        tree_batch = [MolTree(smiles)]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vec, _, mol_vec = self.encode(jtenc_holder, mpn_holder)
        
        mol = Chem.MolFromSmiles(smiles)
        fp1 = AllChem.GetMorganFingerprint(mol, 2)
        
        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec)) #Following Mueller et al.
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec)) #Following Mueller et al.
        mean = torch.cat([tree_mean, mol_mean], dim=1)
        log_var = torch.cat([tree_log_var, mol_log_var], dim=1)
        cur_vec = create_var(mean.data, True)

        visited = []
        for step in range(num_iter):
            prop_val = self.propNN(cur_vec).squeeze()
            grad = torch.autograd.grad(prop_val, cur_vec)[0]
            cur_vec = cur_vec.data + lr * grad.data
            cur_vec = create_var(cur_vec, True)
            visited.append(cur_vec)
        
        result_list = []
        
        for step in range(num_iter):
            new_vec = visited[step]
            tree_vec,mol_vec = torch.chunk(new_vec, 2, dim=1)
            new_smiles = self.decode(tree_vec, mol_vec, prob_decode=False)
            if new_smiles is None:
                result_list.append('None')
                continue
            new_mol = Chem.MolFromSmiles(new_smiles)
            if new_mol is None:
                result_list.append('None')
                continue
            # sim_cutoff == 0.0
            if sim_cutoff==0.0:
                result_list.append(new_smiles)
            # sim_cutoff != 0.0
            else:
                fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
                sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
                if sim < sim_cutoff:
                    result_list.append('None')
                else:
                    result_list.append(new_smiles)
        return result_list
            
        
        
        """
        best_vec = visited[0]
        for new_vec in visited:
            tree_vec,mol_vec = torch.chunk(new_vec, 2, dim=1)
            new_smiles = self.decode(tree_vec, mol_vec, prob_decode=False)
            if new_smiles is None: continue
            new_mol = Chem.MolFromSmiles(new_smiles)
            fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
            if sim >= sim_cutoff:
                best_vec = new_vec
        """
        # tree_vec,mol_vec = torch.chunk(visited[l], 2, dim=1)
        # #tree_vec,mol_vec = torch.chunk(best_vec, 2, dim=1)
        # new_smiles = self.decode(tree_vec, mol_vec, prob_decode=False)
        # if new_smiles is None:
        #     return None, None
        # new_mol = Chem.MolFromSmiles(new_smiles)
        # fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
        # sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
        # if sim >= sim_cutoff:
        #     return new_smiles, sim
        # else:
        #     return None, None
    
    def reconstruct(self, smiles, prob_decode=False):
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)
        del mol_tree.mol
        for node in mol_tree.nodes:
            del node.mol
        tree_batch = [mol_tree]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        x_tree_vecs, _, x_mol_vecs = self.encode(jtenc_holder, mpn_holder)
        
        z_tree_vecs,_ = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs,_ = self.rsample(x_mol_vecs, self.G_mean, self.G_var)
        
        return self.decode(z_tree_vecs, z_mol_vecs, prob_decode)
    
    def grad_ascent(self, smiles, base_lr=2.0, num_iter=20, grid_size=3,sim_cutoff=0,must_find=False):
        # mol_tree = MolTree(smiles)
        # mol_tree.recover()
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)
        del mol_tree.mol
        for node in mol_tree.nodes:
            del node.mol
        tree_batch = [mol_tree]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vec, _, mol_vec = self.encode(jtenc_holder, mpn_holder)
        
        mol = Chem.MolFromSmiles(smiles)
        fp1 = AllChem.GetMorganFingerprint(mol, 2)
        
        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec)) #Following Mueller et al.
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec)) #Following Mueller et al.
        mean = torch.cat([tree_mean, mol_mean], dim=1)
        log_var = torch.cat([tree_log_var, mol_log_var], dim=1)
        cur_vec = create_var(mean.data, True)
        
        mid_grid = grid_size // 2
        z_list = []
        decoded=[]
        lr = 0
        for i in tqdm(range(grid_size)):
            
            # z_list.append(visited)
            if sim_cutoff==0:
                lr = (i - mid_grid) * base_lr
                if lr == 0:
                    decoded.append((smiles,1.0))
                    z_list.append(cur_vec.detach().cpu().numpy().reshape(1,-1))
                    continue
                visited = []
                for step in range(num_iter):
                    prop_val = self.propNN(cur_vec).squeeze()
                    grad = torch.autograd.grad(prop_val, cur_vec)[0]
                    cur_vec = cur_vec.data + lr * grad.data
                    cur_vec = create_var(cur_vec, True)
                    visited.append(cur_vec)
                max_sim=-100.0
                best_smiles = 'C'
                best_vec = visited[0]
                for new_vec in visited:
                    tree_vec,mol_vec = torch.chunk(new_vec, 2, dim=1)
                    new_smiles = self.decode(tree_vec, mol_vec, prob_decode=False)
                    if new_smiles is not None:
                        new_mol = Chem.MolFromSmiles(new_smiles)
                        fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
                        sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
                        if sim >= max_sim:
                            max_sim = sim
                            best_smiles = new_smiles
                            best_vec = new_vec
                decoded.append((best_smiles,max_sim))
                z_list.append(best_vec.detach().cpu().numpy().reshape(1,-1))
            elif must_find==False:
                lr = (i - mid_grid) * base_lr
                if lr == 0:
                    decoded.append((smiles,1.0))
                    z_list.append(cur_vec.detach().cpu().numpy().reshape(1,-1))
                    continue
                visited = []
                for step in range(num_iter):
                    prop_val = self.propNN(cur_vec).squeeze()
                    grad = torch.autograd.grad(prop_val, cur_vec)[0]
                    cur_vec = cur_vec.data + lr * grad.data
                    cur_vec = create_var(cur_vec, True)
                    visited.append(cur_vec)
                # l,r = 0, num_iter - 1
                # while l < r - 1:
                #     mid = (l + r) // 2
                #     new_vec = visited[mid]
                #     tree_vec,mol_vec = torch.chunk(new_vec, 2, dim=1)
                #     new_smiles = self.decode(tree_vec, mol_vec, prob_decode=False)
                #     if new_smiles is None:
                #         r = mid - 1
                #         continue

                #     new_mol = Chem.MolFromSmiles(new_smiles)
                #     fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
                #     sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
                #     if sim < sim_cutoff:
                #         r = mid - 1
                #     else:
                #         l = mid
                # tree_vec,mol_vec = torch.chunk(visited[l], 2, dim=1)
                # z_list.append(visited[l].detach().cpu().numpy().reshape(1,-1))
                best_vec = visited[0]
                max_sim=-100.0
                best_smiles = 'C'
                for new_vec in visited:
                    tree_vec,mol_vec = torch.chunk(new_vec, 2, dim=1)
                    new_smiles = self.decode(tree_vec, mol_vec, prob_decode=False)
                    if new_smiles is not None:
                        new_mol = Chem.MolFromSmiles(new_smiles)
                        fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
                        sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
                        if sim >= max_sim:
                            max_sim = sim
                            best_smiles = new_smiles
                            best_vec = new_vec
                tree_vec,mol_vec = torch.chunk(best_vec, 2, dim=1)
                new_smiles = self.decode(tree_vec, mol_vec, prob_decode=False)
                if new_smiles is None:
                    decoded.append((smiles,1.0))
                    z_list.append(cur_vec.detach().cpu().numpy().reshape(1,-1))
                new_mol = Chem.MolFromSmiles(new_smiles)
                fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
                sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
                if sim >= sim_cutoff:
                    decoded.append((new_smiles,sim))
                    z_list.append(best_vec.detach().cpu().numpy().reshape(1,-1))
                else:
                    decoded.append((smiles,1.0))
                    z_list.append(cur_vec.detach().cpu().numpy().reshape(1,-1))
            else:
                while len(decoded) == i:
                    lr+=base_lr
                    print('Trying %d * lr'%(int(lr/base_lr)))
                    visited = []
                    for step in range(num_iter):
                        prop_val = self.propNN(cur_vec).squeeze()
                        grad = torch.autograd.grad(prop_val, cur_vec)[0]
                        cur_vec = cur_vec.data + lr * grad.data
                        cur_vec = create_var(cur_vec, True)
                        visited.append(cur_vec)
                    best_vec = visited[0]
                    max_sim=-100.0
                    best_smiles = 'C'
                    for new_vec in visited:
                        tree_vec,mol_vec = torch.chunk(new_vec, 2, dim=1)
                        new_smiles = self.decode(tree_vec, mol_vec, prob_decode=False)
                        if new_smiles is not None:
                            new_mol = Chem.MolFromSmiles(new_smiles)
                            fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
                            sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
                            if sim >= max_sim:
                                max_sim = sim
                                best_smiles = new_smiles
                                best_vec = new_vec
                    if max_sim >= sim_cutoff:
                        decoded.append((best_smiles,max_sim))
                        z_list.append(best_vec.detach().cpu().numpy().reshape(1,-1))
        
        return decoded,z_list
        
        """
        best_vec = visited[0]
        for new_vec in visited:
            tree_vec,mol_vec = torch.chunk(new_vec, 2, dim=1)
            new_smiles = self.decode(tree_vec, mol_vec, prob_decode=False)
            if new_smiles is None: continue
            new_mol = Chem.MolFromSmiles(new_smiles)
            fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2) 
            if sim >= sim_cutoff:
                best_vec = new_vec
        """
