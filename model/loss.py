import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision.models as models
import random

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
class EdgeLoss(nn.Module):
    def __init__(self,device):
        super(EdgeLoss, self).__init__()
        # Sobel operator kernels, initialized for edge detection
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).cuda(device=device)
        self.sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).cuda(device=device)
        self.sobel_kernel_x = self.sobel_kernel_x.view((1, 1, 3, 3))
        self.sobel_kernel_y = self.sobel_kernel_y.view((1, 1, 3, 3))
        
    def forward(self, gen_imgs, real_imgs):
        # Assumes gen_imgs and real_imgs are torch tensors of shape (batch_size, channels, height, width)
        gen_imgs_edges_x = F.conv2d(gen_imgs, self.sobel_kernel_x, padding=1)
        gen_imgs_edges_y = F.conv2d(gen_imgs, self.sobel_kernel_y, padding=1)
        real_imgs_edges_x = F.conv2d(real_imgs, self.sobel_kernel_x, padding=1)
        real_imgs_edges_y = F.conv2d(real_imgs, self.sobel_kernel_y, padding=1)
          
        gen_imgs_edges = torch.sqrt(torch.pow(gen_imgs_edges_x, 2) + torch.pow(gen_imgs_edges_y, 2) + 1e-8)
        real_imgs_edges = torch.sqrt(torch.pow(real_imgs_edges_x, 2) + torch.pow(real_imgs_edges_y, 2) + 1e-8)
        
        edge_loss = F.l1_loss(gen_imgs_edges, real_imgs_edges)  # or use F.mse_loss
        return edge_loss
def gram_matrix(input_features):
    """
    Compute the Gram Matrix of the input features.
    'input_features' is a tensor of shape (batch_size, number_of_feature_maps, height, width).
    """
    a, b, c, d = input_features.size()  # a=batch size(=1), b=number of feature maps, (c,d)=dimensions of a feature map (N=c*d)
    features = input_features.view(a * b, c * d)  # resize F_XL into \hat F_XL
    
    G = torch.mm(features, features.t())  # compute the gram product
    # Normalize the values of the gram matrix by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
def style_loss(style_features, generated_features):
    """
    Compute the style loss as the mean squared error between the Gram matrices of
    the style image and the generated image.
    """
    G = gram_matrix(generated_features)
    A = gram_matrix(style_features)
    return F.mse_loss(G, A)  
class StyleLoss(nn.Module):
    def __init__(self, device="cuda"):
        super(StyleLoss, self).__init__()
        self.device = device
        self.vgg = models.vgg16(pretrained=True).features.to(device)
        self.style_feature_layers = [1, 6, 11, 20, 29]  # 注意VGG的layer numbering starts at 0
        
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_style_features(self, x):
        features = []
        for name, layer in enumerate(self.vgg):
            x = layer(x)
            if name in self.style_feature_layers:
                features.append(x)
            if len(features) == len(self.style_feature_layers):
                break
        return features

    def forward(self, style_image, generated_image):
        style_image = style_image.repeat(1, 3, 1, 1).to(self.device)
        generated_image = generated_image.repeat(1, 3, 1, 1).to(self.device)

        s_loss = 0.0
        style_features = self.get_style_features(style_image)
        generated_features = self.get_style_features(generated_image)
        for sf, gf in zip(style_features, generated_features):
            s_loss += style_loss(sf, gf)

        return s_loss / len(self.style_feature_layers)

class NCELoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super(NCELoss, self).__init__()
        self.margin = margin
        self.mlp = nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 256))
    
    def loss(self, anchor, positive, negatives):
        # 计算anchor和positive之间的相似度
        pos_sim = F.cosine_similarity(anchor, positive.unsqueeze(0))
        # 计算anchor和negatives之间的相似度
        neg_sim = F.cosine_similarity(anchor, negatives, dim=1)
        # 计算NCE损失
        losses = F.relu(self.margin - pos_sim + neg_sim.mean())
        return losses

    def forward(self, input, label):
        cumulative_loss = 0.0
        input = self.mlp(input)
        for i in range(len(input)):
            item = input[i]
            item_label = label[i]
            # 对于item中随机抽取一张图片的特征
            idx = random.randint(0, len(item)-1)
            anchor = item[idx]
            # 获取正样本
            positive = [v[idx] for k, v in enumerate(input) if label[k] == item_label and k != i]
            if positive:
                positive = torch.stack(positive).to(input.device)
            else:
                continue

            # 获取负样本
            negative = [v[idx] for k, v in enumerate(input) if label[k] != item_label]
            # 将负样本转换为tensor
            negative = torch.stack(negative).to(input.device)

            cumulative_loss += self.loss(anchor, positive, negative)
        return cumulative_loss