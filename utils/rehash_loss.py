import torch.nn as nn
import torch
import sys


class ReHashLoss(nn.Module):
    def __init__(self, gamma, code_length, num_database, temp, alpha, margin, lam, weight_rh, weight_sql):
        super(ReHashLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha
        self.margin = margin
        self.lam = lam
        self.code_length = code_length
        self.num_database = num_database
        self.gamma = gamma
        self.weight_rh = weight_rh
        self.weight_sql = weight_sql

        self.alpha_uu = alpha
        self.margin_uu = margin

        self.eps = 1e-12


    def forward(self, u, V, S, S_omega_u, V_omega_u, batch_ind):

        V = torch.from_numpy(V).type(torch.FloatTensor).cuda()
        V_omega_u = torch.from_numpy(V_omega_u).type(torch.FloatTensor).cuda()

        S = S.cuda()
        S_omega_u = S_omega_u.cuda()

        n = u.size(0)

        ##======================================================================
        ## Compute pairwise distance in the batch (among U)
        # dist_uu = torch.pow(u, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist_uu = dist_uu + dist_uu.t()
        # dist_uu.addmm_(1, -2, u, u.t())
        # dist_uu = dist_uu.clamp(min=1e-12).sqrt() # for numerical stability
        ##======================================================================

        ## Compute pairwise distance between U and V
        dist_u = torch.pow(u, 2).sum(dim=1, keepdim=True).expand(n, self.num_database)
        dist_V = torch.pow(V, 2).sum(dim=1, keepdim=True).expand(self.num_database, n)
        dist_uV = dist_u + dist_V.t()
        dist_uV.addmm_(1, -2, u, V.t())
        dist_uV = dist_uV.clamp(min=1e-12).sqrt() # for numerical stability

        # mask_omega = S_omega_u > 0
        mask_uV_pos = (S > 0).to(torch.uint8)
        mask_uV_neg = S <= 0
        one_diag = torch.diag(torch.ones(n, dtype=torch.uint8)).cuda()
        mask_uV_pos[:,batch_ind] = mask_uV_pos[:,batch_ind] - one_diag
        mask_uV_pos = mask_uV_pos.to(torch.bool)

        Loss_rh = []

        for i in range(n):

            ##======================================================================================
            ##======================================================================================
            ## ReHashLoss inside batch
            # index_pos_uu = dist_uu[i][mask_omega[i]] > (self.alpha_uu - self.margin_uu)
            # index_neg_uu = dist_uu[i][mask_omega[i] == 0] < self.alpha
            #
            # Lp_uu = torch.max(dist_uu[i][mask_omega[i]] - (self.alpha_uu - self.margin_uu),
            #                   other=torch.zeros_like(dist_uu[i][mask_omega[i]]))
            # Lp_uu = torch.mean(Lp_uu)
            #
            # Ln_uu = self.alpha_uu - dist_uu[i][mask_omega[i] == 0][index_neg_uu]
            # if len(index_neg_uu.nonzero()) != 0:
            #     w_uu = torch.exp(self.temp * (self.alpha_uu - dist_uu[i][mask_omega[i] == 0][index_neg_uu]))
            #     Ln_uu = torch.sum(torch.mul(w_uu, Ln_uu)) / torch.sum(w_uu)
            # else:
            #     Ln_uu = 0.0
            #
            # Loss_rh_uu = Lp_uu + self.lam * Ln_uu
            # Loss_rh.append(Loss_rh_uu)

            ##======================================================================================
            ##======================================================================================
            ## ReHashLoss between u and V
            index_pos_uV = dist_uV[i][mask_uV_pos[i]] > (self.alpha - self.margin)
            index_neg_uV = dist_uV[i][mask_uV_neg[i]] < self.alpha

            Lp_uV = torch.max(dist_uV[i][mask_uV_pos[i]] - (self.alpha - self.margin),
                              other=torch.zeros_like(dist_uV[i][mask_uV_pos[i]]))
            Lp_uV = torch.mean(Lp_uV)

            if len(index_neg_uV.nonzero()) != 0:
                Ln_uV = self.alpha - dist_uV[i][mask_uV_neg[i]][index_neg_uV]
                w_uV = torch.exp(self.temp * (self.alpha - dist_uV[i][mask_uV_neg[i]][index_neg_uV]))
                Ln_uV = torch.sum(torch.mul(w_uV, Ln_uV)) / (torch.sum(w_uV) + self.eps)
            else:
                Ln_uV = 0.0

            Loss_rh_uV = Lp_uV + self.lam * Ln_uV
            Loss_rh.append(Loss_rh_uV)

        Loss_rh = torch.mean(torch.stack(Loss_rh))
        square_loss = torch.mean((u.mm(V.t())-self.code_length * S) ** 2)
        quantization_loss = torch.mean((V_omega_u - u) ** 2)

        loss = self.weight_rh * Loss_rh + self.weight_sql * square_loss + self.gamma * quantization_loss

        ##======================================================================================
        ##======================================================================================
        ## recording W matrix for laplacian
        dist_lt_alpha = (dist_uV < self.alpha)
        dist_gt_alpha = (dist_uV > (self.alpha - self.margin))
        neg_alpha = (mask_uV_neg) * (dist_lt_alpha)
        pos_alpha = (mask_uV_pos) * (dist_gt_alpha)

        w_neg = torch.exp(self.temp * (self.alpha - dist_uV)) * neg_alpha.to(torch.float32)
        w_neg = w_neg / (w_neg.sum(dim=1, keepdim=True) + self.eps)

        W = -1 * w_neg + pos_alpha.to(torch.float32)

        return loss, W
