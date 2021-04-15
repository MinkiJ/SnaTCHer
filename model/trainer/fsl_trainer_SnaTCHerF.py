import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.SnaTCHer_helpers import (
    get_dataloader, prepare_model,
)
from model.utils import (
    count_acc,
)
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score

def calc_auroc(known_scores, unknown_scores):
    y_true = np.array([0] * len(known_scores) + [1] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    
    return auc_score

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        
        emb_dim = 640 if args.backbone_class == 'Res12' else 512
        self.emb_dim = emb_dim

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux
    
    def evaluate_test(self):
        # restore model args
        emb_dim = self.emb_dim
        args = self.args
        weights = torch.load(osp.join(self.args.save_path, self.args.weight_name))
        model_weights = weights['params']
        self.missing_keys, self.unexpected_keys = self.model.load_state_dict(model_weights, strict=False)
        self.model.eval()
       
        test_steps = 600
        
        self.record = np.zeros((test_steps, 2)) # loss and acc
        self.auroc_record = np.zeros((test_steps, 10))
        label = torch.arange(args.closed_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
                
        way = args.closed_way
        label = torch.arange(way).repeat(15).cuda()
        
        for i, batch in tqdm(enumerate(self.test_loader, 1)):
            if i > test_steps:
                break
            
            if torch.cuda.is_available():
                data, dlabel = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            
            self.probe_data = data
            self.probe_dlabel = dlabel
            
            with torch.no_grad():
                _ = self.para_model(data)
                instance_embs = self.para_model.probe_instance_embs
                support_idx = self.para_model.probe_support_idx
                query_idx = self.para_model.probe_query_idx
                
                support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
                query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))
                emb_dim = support.shape[-1]
                
                support = support[:, :, :way].contiguous()
                # get mean of the support
                bproto = support.mean(dim=1) # Ntask x NK x d
                proto = bproto
                
                kquery = query[:, :, :way].contiguous()
                uquery = query[:, :, way:].contiguous()

                # get mean of the support
                proto = self.para_model.slf_attn(proto, proto, proto)
                proto = proto[0]
                    
            klogits = -(kquery.reshape(-1, 1, emb_dim) - proto).pow(2).sum(2) / 64.0
            ulogits = -(uquery.reshape(-1, 1, emb_dim) - proto).pow(2).sum(2) / 64.0
            

            loss = F.cross_entropy(klogits, label)
            acc = count_acc(klogits, label)

            """ Probability """
            known_prob = F.softmax(klogits, 1).max(1)[0]
            unknown_prob = F.softmax(ulogits, 1).max(1)[0]

            known_scores = (known_prob).cpu().detach().numpy()
            unknown_scores = (unknown_prob).cpu().detach().numpy()
            known_scores = 1 - known_scores
            unknown_scores = 1 - unknown_scores
            
            auroc = calc_auroc(known_scores, unknown_scores)
            
            """ Distance """
            kdist = -(klogits.max(1)[0])
            udist = -(ulogits.max(1)[0])
            kdist = kdist.cpu().detach().numpy()
            udist = udist.cpu().detach().numpy()
            dist_auroc = calc_auroc(kdist, udist)
            
            """ Snatcher """
            with torch.no_grad():
              snatch_known = []
              for j in range(75):
                pproto = bproto.clone().detach()
                """ Algorithm 1 Line 1 """
                c = klogits.argmax(1)[j]
                """ Algorithm 1 Line 2 """
                pproto[0][c] = kquery.reshape(-1, emb_dim)[j]
                """ Algorithm 1 Line 3 """
                pproto = self.para_model.slf_attn(pproto, pproto, pproto)[0]
                pdiff = (pproto - proto).pow(2).sum(-1).sum() / 64.0
                """ pdiff: d_SnaTCHer in Algorithm 1 """
                snatch_known.append(pdiff)
                
              snatch_unknown = []
              for j in range(ulogits.shape[0]):
                pproto = bproto.clone().detach()
                """ Algorithm 1 Line 1 """
                c = ulogits.argmax(1)[j]
                """ Algorithm 1 Line 2 """
                pproto[0][c] = uquery.reshape(-1, emb_dim)[j]
                """ Algorithm 1 Line 3 """
                pproto = self.para_model.slf_attn(pproto, pproto, pproto)[0]
                pdiff = (pproto - proto).pow(2).sum(-1).sum() / 64.0
                """ pdiff: d_SnaTCHer in Algorithm 1 """
                snatch_unknown.append(pdiff)
                
              pkdiff = torch.stack(snatch_known)
              pudiff = torch.stack(snatch_unknown)
              pkdiff = pkdiff.cpu().detach().numpy()
              pudiff = pudiff.cpu().detach().numpy()
              
              snatch_auroc = calc_auroc(pkdiff, pudiff)
            

                        
            self.record[i-1, 0] = loss.item()
            self.record[i-1, 1] = acc
            self.auroc_record[i-1, 0] = auroc
            self.auroc_record[i-1, 1] = snatch_auroc
            self.auroc_record[i-1, 2] = dist_auroc

            if i % 100 == 0:                
                vdata = self.record[:, 1]
                vdata = 1.0 * np.array(vdata)
                vdata = vdata[:i]
                va = np.mean(vdata)
                std = np.std(vdata)
                vap = 1.96 * (std / np.sqrt(i))
                
                audata = self.auroc_record[:,0]
                audata = np.array(audata, np.float32)
                audata = audata[:i]
                aua = np.mean(audata)
                austd = np.std(audata)
                auap = 1.96 * (austd / np.sqrt(i))
                
                sdata = self.auroc_record[:,1]
                sdata = np.array(sdata, np.float32)
                sdata = sdata[:i]
                sa = np.mean(sdata)
                sstd = np.std(sdata)
                sap = 1.96 * (sstd / np.sqrt(i))
                
                ddata = self.auroc_record[:, 2]
                ddata = np.array(ddata, np.float32)[:i]
                da = np.mean(ddata)
                dstd = np.std(ddata)
                dap = 1.96 * (dstd / np.sqrt(i))
                
                print("acc: {:.4f} + {:.4f} Prob: {:.4f} + {:.4f} Dist: {:.4f} + {:.4f} SnaTCHer: {:.4f} + {:.4f}"\
                      .format(va, vap, aua, auap, da, dap, sa, sap))
              
        return
