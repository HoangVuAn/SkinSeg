import torch
from torch.nn import functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from itertools import permutations



def corr_loss_nor(feat_w, feat_s):
    """
    Compute correlation loss between weak and strong features
    Args:
        feat_w: weak features
        feat_s: strong features
    Returns:
        correlation loss
    """
    def orthogonal_landmarks(q, q_s, num_landmarks=64, subsample_fraction=1.0):
        """
        Construct set of landmarks by recursively selecting new landmarks 
        that are maximally orthogonal to the existing set.
        Returns near orthogonal landmarks with shape (B, M, D).
        """
        B, D, H, W = q.shape
        N = H * W
        q = q.permute(0, 2, 3, 1).reshape(B, -1, D)
        q_s = q_s.permute(0, 2, 3, 1).reshape(B, -1, D)

        # ignore_mask = F.interpolate(ignore_mask.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
        # nonignore_mask = nonignore_mask.reshape(B, -1)

        if subsample_fraction < 1.0:
            # Need at least M/2 samples of queries and keys
            num_samples = max(int(subsample_fraction * q.size(-2)), num_landmarks)
            q_unnormalised = q[:, torch.randint(q.size(-2), (num_samples,), device=q.device), :] # (B, N, D)
        else:
            # (B, N, D)
            q_unnormalised = q

        # may need to change default eps to eps=1e-8 for mixed precision compatibility
        qk = F.normalize(q_unnormalised, p=2, dim=-1)
        # B, N, D = qk.shape

        selected_mask = torch.zeros((B, N, 1), device=qk.device)
        landmark_mask = torch.ones((B, 1, 1), dtype=selected_mask.dtype, device=qk.device)

        # Get initial random landmark
        random_idx = torch.randint(qk.size(-2), (B, 1, 1), device=qk.device)
        # random_idx = torch.empty((B, 1, 1), dtype=torch.long, device=qk.device)
        # for i in range(B):
        #     nonignore_indices = torch.nonzero(nonignore_mask[i])
        #     selected_index = nonignore_indices[torch.randint(0, nonignore_indices.size(0), (1,))]
        #     random_idx[i, 0, 0] = selected_index.item()

        selected_landmark = qk[torch.arange(qk.size(0)), random_idx.view(-1), :].view(B, D)
        selected_mask.scatter_(-2, random_idx, landmark_mask)

        # Selected landmarks
        selected_landmarks = torch.empty((B, num_landmarks, D), device=qk.device, dtype=qk.dtype)
        selected_landmarks[:, 0, :] = selected_landmark

        # Store computed cosine similarities
        cos_sims = torch.empty((B, N, num_landmarks), device=qk.device, dtype=qk.dtype)

        for M in range(1, num_landmarks):
            # Calculate absolute cosine similarity between selected and unselected landmarks
            # (B, N, D) * (B, D) -> (B, N)
            cos_sim = torch.einsum('b n d, b d -> b n', qk, selected_landmark).abs()
            # # set cosine similarity for ignore mask to > 1
            # cos_sim.view(-1)[nonignore_mask.flatten() == False] = 10
            cos_sims[:, :, M - 1] = cos_sim
            # (B, N, M) cosine similarities of current set of landmarks wrt all queries and keys
            cos_sim_set = cos_sims[:, :, :M]

            # Get orthogonal landmark: landmark with smallest absolute cosine similarity:
            # set cosine similarity for already selected landmarks to > 1
            cos_sim_set.view(-1, M)[selected_mask.flatten().bool(), :] = 10
            # (B,) - want max for non
            selected_landmark_idx = cos_sim_set.amax(-1).argmin(-1)
            selected_landmark = qk[torch.arange(qk.size(0)), selected_landmark_idx, :].view(B, D)

            # Add most orthogonal landmark to selected landmarks: 
            selected_landmarks[:, M, :] = selected_landmark

            # Removed selected indices from non-selected mask: 
            selected_mask.scatter_(-2, selected_landmark_idx.unsqueeze(-1).unsqueeze(-1), landmark_mask)

        landmarks = torch.masked_select(q_unnormalised, selected_mask.bool()).reshape(B, -1, D) # (B, M, D)
        landmarks_s = torch.masked_select(q_s, selected_mask.bool()).reshape(B, -1, D)

        return landmarks, landmarks_s # (B, M, D)
    
    def prob2rank(prob, prob_s, k=4):
        """
        input: prob(probability) [b, h, w, n]
        return: rank [b, h, w, k!]
        To save the computing resources, use top-k ranther than n
        """
        full_permutation = [c for c in permutations(range(k))]
        full_permutation = torch.from_numpy(np.stack(full_permutation)) # [k!, k]

        _, prob_topk_index = prob.topk(k, dim=-1) # [b, h, w, k]
        A = prob_topk_index[:, :, :, full_permutation] # [b, h, w, k!, k]
        B = prob.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1) # [b, h, w, k!, n]
        B_s = prob_s.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1)
        C = torch.gather(input=B, dim=-1, index=A) # [b, h, w, k!, k]
        C_s = torch.gather(input=B_s, dim=-1, index=A)

        rank = C[:, :, :, :, 0] / (C[:, :, :, :, 0:].sum(dim=-1) + 1e-10) # [b, h, w, k!]
        rank_s = C_s[:, :, :, :, 0] / (C_s[:, :, :, :, 0:].sum(dim=-1) + 1e-10)

        for i in range(1, k):
            rank *= C[:, :, :, :, i] / (C[:, :, :, :, i:].sum(dim=-1) + 1e-10)
            rank_s *= C_s[:, :, :, :, i] / (C_s[:, :, :, :, i:].sum(dim=-1) + 1e-10)

        return rank, rank_s


    criterion_c = torch.nn.KLDivLoss(reduction='batchmean') # pixel-reference correlation criterion

    num_landmarks = 64

    refers_w, refers_s = orthogonal_landmarks(feat_w, feat_s, num_landmarks)

    p2r_w = torch.einsum('b c h w, b n c -> b h w n', feat_w, refers_w).softmax(dim=-1)
    p2r_s = torch.einsum('b c h w, b n c -> b h w n', feat_s, refers_s).softmax(dim=-1)

    p2r_w_rank, p2r_s_rank = prob2rank(p2r_w, p2r_s)

    loss = criterion_c((p2r_s_rank + 1e-10).log(), p2r_w_rank)
    loss = 1 - torch.exp(-loss)

    return loss

def corr_loss(feat_w, feat_s):
    """
    Compute correlation loss between weak and strong features
    Args:
        feat_w: weak features
        feat_s: strong features
    Returns:
        correlation loss
    """
    def orthogonal_landmarks(q, q_s, num_landmarks=64, subsample_fraction=1.0):
        """
        Construct set of landmarks by recursively selecting new landmarks 
        that are maximally orthogonal to the existing set.
        Returns near orthogonal landmarks with shape (B, M, D).
        """
        B, D, H, W = q.shape
        N = H * W
        q = q.permute(0, 2, 3, 1).reshape(B, -1, D)
        q_s = q_s.permute(0, 2, 3, 1).reshape(B, -1, D)

        # ignore_mask = F.interpolate(ignore_mask.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
        # nonignore_mask = nonignore_mask.reshape(B, -1)

        if subsample_fraction < 1.0:
            # Need at least M/2 samples of queries and keys
            num_samples = max(int(subsample_fraction * q.size(-2)), num_landmarks)
            q_unnormalised = q[:, torch.randint(q.size(-2), (num_samples,), device=q.device), :] # (B, N, D)
        else:
            # (B, N, D)
            q_unnormalised = q

        # may need to change default eps to eps=1e-8 for mixed precision compatibility
        qk = F.normalize(q_unnormalised, p=2, dim=-1)
        # B, N, D = qk.shape

        selected_mask = torch.zeros((B, N, 1), device=qk.device)
        landmark_mask = torch.ones((B, 1, 1), dtype=selected_mask.dtype, device=qk.device)

        # Get initial random landmark
        random_idx = torch.randint(qk.size(-2), (B, 1, 1), device=qk.device)
        # random_idx = torch.empty((B, 1, 1), dtype=torch.long, device=qk.device)
        # for i in range(B):
        #     nonignore_indices = torch.nonzero(nonignore_mask[i])
        #     selected_index = nonignore_indices[torch.randint(0, nonignore_indices.size(0), (1,))]
        #     random_idx[i, 0, 0] = selected_index.item()

        selected_landmark = qk[torch.arange(qk.size(0)), random_idx.view(-1), :].view(B, D)
        selected_mask.scatter_(-2, random_idx, landmark_mask)

        # Selected landmarks
        selected_landmarks = torch.empty((B, num_landmarks, D), device=qk.device, dtype=qk.dtype)
        selected_landmarks[:, 0, :] = selected_landmark

        # Store computed cosine similarities
        cos_sims = torch.empty((B, N, num_landmarks), device=qk.device, dtype=qk.dtype)

        for M in range(1, num_landmarks):
            # Calculate absolute cosine similarity between selected and unselected landmarks
            # (B, N, D) * (B, D) -> (B, N)
            cos_sim = torch.einsum('b n d, b d -> b n', qk, selected_landmark).abs()
            # # set cosine similarity for ignore mask to > 1
            # cos_sim.view(-1)[nonignore_mask.flatten() == False] = 10
            cos_sims[:, :, M - 1] = cos_sim
            # (B, N, M) cosine similarities of current set of landmarks wrt all queries and keys
            cos_sim_set = cos_sims[:, :, :M]

            # Get orthogonal landmark: landmark with smallest absolute cosine similarity:
            # set cosine similarity for already selected landmarks to > 1
            cos_sim_set.view(-1, M)[selected_mask.flatten().bool(), :] = 10
            # (B,) - want max for non
            selected_landmark_idx = cos_sim_set.amax(-1).argmin(-1)
            selected_landmark = qk[torch.arange(qk.size(0)), selected_landmark_idx, :].view(B, D)

            # Add most orthogonal landmark to selected landmarks: 
            selected_landmarks[:, M, :] = selected_landmark

            # Removed selected indices from non-selected mask: 
            selected_mask.scatter_(-2, selected_landmark_idx.unsqueeze(-1).unsqueeze(-1), landmark_mask)

        landmarks = torch.masked_select(q_unnormalised, selected_mask.bool()).reshape(B, -1, D) # (B, M, D)
        landmarks_s = torch.masked_select(q_s, selected_mask.bool()).reshape(B, -1, D)

        return landmarks, landmarks_s # (B, M, D)
    
    def prob2rank(prob, prob_s, k=4):
        """
        input: prob(probability) [b, h, w, n]
        return: rank [b, h, w, k!]
        To save the computing resources, use top-k ranther than n
        """
        full_permutation = [c for c in permutations(range(k))]
        full_permutation = torch.from_numpy(np.stack(full_permutation)) # [k!, k]

        _, prob_topk_index = prob.topk(k, dim=-1) # [b, h, w, k]
        A = prob_topk_index[:, :, :, full_permutation] # [b, h, w, k!, k]
        B = prob.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1) # [b, h, w, k!, n]
        B_s = prob_s.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1)
        C = torch.gather(input=B, dim=-1, index=A) # [b, h, w, k!, k]
        C_s = torch.gather(input=B_s, dim=-1, index=A)

        rank = C[:, :, :, :, 0] / (C[:, :, :, :, 0:].sum(dim=-1) + 1e-10) # [b, h, w, k!]
        rank_s = C_s[:, :, :, :, 0] / (C_s[:, :, :, :, 0:].sum(dim=-1) + 1e-10)

        for i in range(1, k):
            rank *= C[:, :, :, :, i] / (C[:, :, :, :, i:].sum(dim=-1) + 1e-10)
            rank_s *= C_s[:, :, :, :, i] / (C_s[:, :, :, :, i:].sum(dim=-1) + 1e-10)

        return rank, rank_s


    criterion_c = torch.nn.KLDivLoss(reduction='batchmean') # pixel-reference correlation criterion

    num_landmarks = 8

    refers_w, refers_s = orthogonal_landmarks(feat_w, feat_s, num_landmarks)

    p2r_w = torch.einsum('b c h w, b n c -> b h w n', feat_w, refers_w).softmax(dim=-1)
    p2r_s = torch.einsum('b c h w, b n c -> b h w n', feat_s, refers_s).softmax(dim=-1)

    p2r_w_rank, p2r_s_rank = prob2rank(p2r_w, p2r_s)

    loss = criterion_c((p2r_s_rank + 1e-10).log(), p2r_w_rank)

    return loss


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss



def dice_loss1(score, target):
    # non-square
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def iou_loss(score, target):
    target = target.float()
    smooth = 1e-5
    tp_sum = torch.sum(score * target)
    fp_sum = torch.sum(score * (1 - target))
    fn_sum = torch.sum((1 - score) * target)
    loss = (tp_sum + smooth) / (tp_sum + fp_sum + fn_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    ## p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / torch.tensor(
        np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                         keepdim=True) / torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


def compute_sdf01(segmentation):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)

    if len(segmentation.shape) == 4:  # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]):  # batch size
        for c in range(dis_id, segmentation.shape[1]):  # class_num
            # ignore background
            posmask = segmentation[b][c]
            if np.max(posmask) == 0:
                continue
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(
                posmask, mode='inner').astype(np.uint8)
            sdf = negdis / np.max(negdis) / 2 - posdis / np.max(
                posdis) / 2 + 0.5
            sdf[boundary > 0] = 0.5
            normalized_sdf[b][c] = sdf
    return normalized_sdf


def compute_sdf1_1(segmentation):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)
    if len(segmentation.shape) == 4:  # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]):  # batch size
        for c in range(dis_id, segmentation.shape[1]):  # class_num
            # ignore background
            posmask = segmentation[b][c]
            if np.max(posmask) == 0:
                continue
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(
                posmask, mode='inner').astype(np.uint8)
            sdf = negdis / np.max(negdis) - posdis / np.max(posdis)
            sdf[boundary > 0] = 0
            normalized_sdf[b][c] = sdf
    return normalized_sdf


def compute_fore_dist(segmentation):
    """
    compute the foreground of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)
    if len(segmentation.shape) == 4:  # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]):  # batch size
        for c in range(dis_id, segmentation.shape[1]):  # class_num
            # ignore background
            posmask = segmentation[b][c]
            posdis = distance(posmask)
            normalized_sdf[b][c] = posdis / np.max(posdis)
    return normalized_sdf


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def AAAI_sdf_loss(net_output, gt):
    """
    net_output: net logits; shape=(batch_size, class, x, y, z)
    gt: ground truth; (shape (batch_size, 1, x, y, z) OR (batch_size, x, y, z))
    """
    smooth = 1e-5
    axes = tuple(range(2, len(net_output.size())))
    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
        gt_sdm_npy = compute_sdf1_1(y_onehot.cpu().numpy())
        if net_output.device.type == "cuda":
            gt_sdm = torch.from_numpy(gt_sdm_npy).float().cuda(
                net_output.device.index)
        else:
            gt_sdm = torch.from_numpy(gt_sdm_npy).float()
    intersect = sum_tensor(net_output * gt_sdm, axes, keepdim=False)
    pd_sum = sum_tensor(net_output**2, axes, keepdim=False)
    gt_sum = sum_tensor(gt_sdm**2, axes, keepdim=False)
    L_product = (intersect + smooth) / (intersect + pd_sum + gt_sum)
    # print('L_product.shape', L_product.shape) (4,2)
    L_SDF_AAAI = -L_product.mean() + torch.norm(net_output - gt_sdm,
                                                1) / torch.numel(net_output)

    return L_SDF_AAAI


def sdf_kl_loss(net_output, gt):
    """
    net_output: net logits; shape=(batch_size, class, x, y, z)
    gt: ground truth; (shape (batch_size, 1, x, y, z) OR (batch_size, x, y, z))
    """
    smooth = 1e-5
    axes = tuple(range(2, len(net_output.size())))
    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
        # print('y_onehot.shape', y_onehot.shape)
        gt_sdf_npy = compute_sdf(y_onehot.cpu().numpy())
        gt_sdf = torch.from_numpy(gt_sdf_npy + smooth).float().cuda(
            net_output.device.index)
    # print('net_output, gt_sdf', net_output.shape, gt_sdf.shape)
    # exit()
    sdf_kl_loss = F.kl_div(net_output,
                           gt_sdf[:, 1:2, ...],
                           reduction='batchmean')

    return sdf_kl_loss


# don't put the sample itself into the Positive set
class Supervised_Contrastive_Loss(torch.nn.Module):
    '''
    from https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
    https://blog.csdn.net/wf19971210/article/details/116715880
    Treat samples in the same labels as the positive samples, others as negative samples
    '''
    def __init__(self, temperature=0.1, device='cpu'):
        super(Supervised_Contrastive_Loss, self).__init__()
        self.temperature = temperature
        self.device = device
    
    def forward(self, projections, targets, attribute=None):
        # projections (bs, dim), targets (bs)
        # similarity matrix/T
        # dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        dot_product_tempered = F.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0),dim=2)/self.temperature
        # print(dot_product_tempered)
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        # exp_dot_tempered = (
        #     torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        # )
        exp_dot_tempered = torch.exp(dot_product_tempered- torch.max(dot_product_tempered, dim=1, keepdim=True)[0])+ 1e-5
        # a matrix, same labels are true, others are false
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(self.device)
        # a matrix, diagonal are zeros, others are ones
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(self.device)
        mask_nonsimilar_class = ~mask_similar_class
        # mask_nonsimilar_attr = ~mask_similar_attr
        # a matrix, same labels are 1, others are 0, and diagonal are zeros
        mask_combined = mask_similar_class * mask_anchor_out
        # num of similar samples for sample
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        # print(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr)
        # print(torch.sum(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered)
        if attribute != None:
            mask_similar_attr = (attribute.unsqueeze(1).repeat(1, attribute.shape[0]) == attribute).to(self.device)
            log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class * mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered+1e-5))
       
        else:
            log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class, dim=1, keepdim=True)+exp_dot_tempered+1e-5))
        supervised_contrastive_loss = torch.sum(log_prob * mask_combined)/(torch.sum(cardinality_per_samples)+1e-5)

        
        return supervised_contrastive_loss


# class Supervised_Contrastive_Loss(torch.nn.Module):
#     '''
#     from https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
#     https://blog.csdn.net/wf19971210/article/details/116715880
#     Treat samples in the same labels as the positive samples (including itself), others as negative samples
#     '''
#     def __init__(self, temperature=0.1, device='cpu'):
#         super(Supervised_Contrastive_Loss, self).__init__()
#         self.temperature = temperature
#         self.device = device
    
#     def forward(self, projections, targets, attribute=None):
#         # projections (bs, dim), targets (bs)
#         # similarity matrix/T
#         # dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
#         dot_product_tempered = F.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0),dim=2)/self.temperature
#         # print(dot_product_tempered)
#         # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
#         # exp_dot_tempered = (
#         #     torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
#         # )
#         exp_dot_tempered = torch.exp(dot_product_tempered- torch.max(dot_product_tempered, dim=1, keepdim=True)[0])+ 1e-6
#         # a matrix, same labels are true, others are false
#         mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(self.device)
#         # a matrix, diagonal are zeros, others are ones
#         mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(self.device)
#         mask_nonsimilar_class = ~mask_similar_class
#         # mask_nonsimilar_attr = ~mask_similar_attr
#         # a matrix, same labels are 1, others are 0, and diagonal are zeros
#         mask_combined = mask_similar_class * mask_anchor_out
#         # num of similar samples for sample
#         cardinality_per_samples = torch.sum(mask_similar_class, dim=1)
#         # print(exp_dot_tempered * mask_nonsimilar_class)
#         # print(torch.sum(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered)
#         if attribute != None:
#             mask_similar_attr = (attribute.unsqueeze(1).repeat(1, attribute.shape[0]) == attribute).to(self.device)
#             log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class * mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered+1e-6))
       
#         else:
#             log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class, dim=1, keepdim=True)+exp_dot_tempered+1e-6))
#         supervised_contrastive_loss = torch.sum(log_prob * mask_similar_class)/(torch.sum(cardinality_per_samples)+1e-6)

        
#         return supervised_contrastive_loss




if __name__ == '__main__':

    # # check supervised contrastive loss
    # loss_func = Supervised_Contrastive_Loss()
    # # a,b = torch.tensor([[0.,0,0,0,1,1,1,1,1,1]]), torch.tensor([[1.,1,1,1,0,0,0,0,0,0]])
    # a,b  = torch.ones((3,7)), torch.ones(3,7)
    # # a,b = a.repeat((3,1)), b.repeat((3,1))
    # # a = torch.tensor([[0.,0,1,1]])
    # # a= a.repeat((6,1))
    # # a = torch.randn(3,10)
    # # b = torch.tensor([[1.,1,1,1,0,0,0,0,0,0]])
    # # x = torch.cat((a,b),dim=0)
    # x = torch.randn(6,10)

    # y = torch.tensor([1,2,3,4,5,6])
    # # z = torch.tensor([2,3,3,2,3,3])
    # loss = loss_func(x, y)
    # print(loss)

    a = torch.tensor([0.0,1.0,0.0,1.0])
    b = torch.tensor([0.0,0.0,0.0,1.0])
    # print(a)
    # print(b)
    dice = dice_per_img(a,b)
    dice_all = dice_loss(a,b)
    print(dice.shape)
    print(dice)
    print(dice_all)