from pathlib import Path
from .segmentor_model_v2 import *
from .segmentor_utils import *
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import scipy.stats

idx_to_class = {0: 'Unassigned (Loop)',
                1: 'Orthogonal Bundle',
                2: 'Up-down Bundle',
                3: 'Alpha Horseshoe',
                4: 'Alpha/alpha barrel',
                5: 'Ribbon',
                6: 'Aligned Prism',
                7: '3-layer Sandwich',
                8: '4 Propeller',
                9: '5 Propeller',
                10: '6 Propeller',
                11: '7 Propeller',
                12: '2 Solenoid',
                13: '3 Solenoid',
                14: 'Beta Complex',
                15: 'Single Sheet',
                16: 'Roll_16',
                17: 'Beta Barrel',
                18: 'Clam',
                19: 'Sandwich',
                20: 'Distorted Sandwich',
                21: 'Trefoil',
                22: 'Orthogonal Prism',
                23: 'Roll_23',
                24: 'Ribosomal Protein L15; Chain: K; domain 2',
                25: 'Super Roll',
                26: 'Alpha-Beta Barrel',
                27: '2-Layer Sandwich',
                28: '3-Layer(aba) Sandwich',
                29: '3-Layer(bba) Sandwich',
                30: '3-Layer(bab) Sandwich',
                31: '4-Layer Sandwich',
                32: 'Alpha-beta prism',
                33: 'Box',
                34: '5-stranded Propeller',
                35: 'Alpha-Beta Horseshoe',
                36: 'Alpha-Beta Complex',
                37: 'Irregular',
                -1: 'NULL'}

class DomainSegmentor:
    def __init__(self, model_path='segmentor_epoch95_model_v2', class_dict=idx_to_class, try_gpu=True):
        self.class_dict = class_dict
        self.cuda_avail = torch.cuda.is_available() and try_gpu
        self.num_classes = len(self.class_dict)-1 
        self.model = SegmentorModel(1,8,16, self.num_classes)
        model_path = Path(__file__).parent / model_path
        self._init_model(self.model, model_path)
        
    def _init_model(self, net, model_path):
        if self.cuda_avail:
            net.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count())) 
            print("Model Initialized on GPU.")
        else:
            net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
            print("Model Initialized on CPU.")
        net.eval() # Set model to evaluation mode to remove stochastic regularization.
    
    def _get_input(self, pdb_name):
        seq_len, numbering = get_pdb_info(pdb_name)
        cm, numbering = makeContactMapTensor(pdb_name, seq_len, numbering, target_size=512, upper_tol=512)
        if self.cuda_avail:
            model_input = Variable(cm).cuda()
        else:
            model_input = Variable(cm)
        return numbering, model_input

    def predict(self, pdb_name, ignore_index=-9999, log=False, return_embeds=False):
        '''
        Input: pdb name as string.
        Output: 
            trunc_class_probs -- 38 x 512 matrix. Entry (i,j) is the probility of residue j being in class i.
            res_num -- the list of pdb_residue numberings corresponding to the columns of trunc_class_probs. For example, out_num[i] = 50, then the i-th column of trunc_class_probs corresponds to residue 50 in the actual PDB.
        '''
        numbering, model_input = self._get_input(pdb_name)
        embeds = self.model(model_input)
        if log:
            outputs = F.log_softmax(embeds, dim=1)
        else:
            outputs = F.softmax(embeds, dim=1)
        outputs = outputs.data[0,:,:,:]
        # Format for output.
        class_probs = outputs.cpu().numpy().squeeze() # 38 x 512 matrix. The columns define a probability distribution over the 38 classes for each residue.
        res_num = []
        trunc_class_probs = np.array([None])
        for i in range(len(numbering)): # Remove entries outside of the range of the PDB.
            if numbering[i] != ignore_index:
                res_num.append(numbering[i])
                #if not trunc_class_probs.any():
                #    trunc_class_probs = np.expand_dims(class_probs[:,i], axis=1)
                #else:
                #    trunc_class_probs = np.column_stack([trunc_class_probs, class_probs[:,i]])
        if return_embeds:
            return embeds.detach().cpu().numpy().squeeze(), numbering
        else:
            return trunc_class_probs, res_num

    def predictClass(self, pdb_name, ignore_index=-9999):
        '''
        Input: pdb name as string.
        Output:
            out_pred -- the predicted classes for each residue.
            res_num -- the pdb residue numberings corresponding to the entries in out_pred. For example, if res_num[i] = 10 and out_pred[i] = 15, then the model predicts class 15 for residue 10. 
        '''
        numbering, model_input = self._get_input(pdb_name)
        outputs = self.model(model_input)
        _, predicted = torch.max(outputs.data, 1)
        # Format for output.
        predicted = predicted[0,:,:].cpu().numpy().flatten()
        out_pred = []
        res_num = []
        for i in range(len(numbering)):
            if numbering[i] != ignore_index:
                out_pred.append(predicted[i])
                res_num.append(numbering[i])
        assert len(out_pred) == len(res_num)
        return out_pred, res_num

    def computeEntropy(self, pdb_name, ignore_index=-9999):
        trunc_class_probs, res_num = self.predict(pdb_name, log=False)
        entropy = scipy.stats.entropy(trunc_class_probs)
        assert len(entropy) == len(res_num)
        return entropy, res_num
