import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Tnet(nn.Module):
    '''Learns a Transformation matrix for the specified dimension'''
    def __init__(self, num_channels):
        super(Tnet, self).__init__()

        self.num_channels = num_channels 

        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_channels**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)        

    def forward(self, x):

        bs = x.shape[0]

        # shared MLP(64, 128, 1024)
        x = F.relu(self.bn1(self.conv1(x))) # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x))) # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x))) # (B, 1024, N)

        # max pooling across points
        x = torch.max(x, dim=2, keepdim=False)[0] # (B, 1024)

        # fully connected layers with output sizes 512, 256
        x = F.relu(self.bn4(self.linear1(x))) # (B, 512)
        x = F.relu(self.bn5(self.linear2(x))) # (B, 256)

        # Resize to build transformation matrix, hence **2
        x = self.linear3(x) # (B, num_channels^2)

        # initialize identity matrix
        iden = torch.eye(self.num_channels) # identity matrix for 1 channel
        iden = iden.repeat(bs, 1, 1).to(x.device) # repeat for each batch
        
        # reshape to get transformation matrix
        x = x.view(-1, self.num_channels, self.num_channels) 
        # add identity matrix to the transformation matrix for regularization
        x = x + iden

        return x



class PointNetBackbone(nn.Module):
    '''The entire backbone before the classification or segmentation heads''' 
    def __init__(self,num_channels, append_local_feat=True, return_critical_indexes=False):
        super(PointNetBackbone, self).__init__()

        self.append_local_feat = append_local_feat
        self.num_channels = num_channels
        self.return_critical_indexes = return_critical_indexes

        # Spatial Transformer Networks (T-nets)
        self.tnet1 = Tnet(3) # For the xyz coordinates
        self.tnet2 = Tnet(64) # For the 64 dimensional features

        # shared MLP 1
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # shared MLP 2
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1)
        
        # batch norms for both shared MLPs
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    
    def forward(self, x):
        
        # get batch shape
        B, C, N = x.shape
        
        # Get First Transform matrix
        Transf1 = self.tnet1(x) # (B, 3, 3)
        if C < 3:
            raise ValueError("Input must have at least 3 channels for xyz coordinates.")
        

        # Apply transform to position channels only
        xyz = x[:, :3, :]
        transformed_xyz = torch.bmm(xyz.transpose(2, 1), Transf1).transpose(2, 1)
        if C > 3:
            features = x[:, 3:, :]
            x = torch.cat((transformed_xyz, features), dim=1)
        else:
            x = transformed_xyz

        # shared MLP(64, 64)
        x = F.relu(self.bn1(self.conv1(x))) # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x))) # (B, 64, N)
        
        # Get and apply Second Transform matrix
        Transf2 = self.tnet2(x) # (B, 64, 64)
        x = torch.bmm(x.transpose(2, 1), Transf2).transpose(2, 1) # (B, 64, N)
        local_features = x.clone() # (B, 64, N)

        # shared MLP(64, 128, 1024)
        x = F.relu(self.bn3(self.conv3(x))) # (B, 64, N)
        x = F.relu(self.bn4(self.conv4(x))) # (B, 128, N)
        x = F.relu(self.bn5(self.conv5(x))) # (B, 1024, N)

        # Max pooling to get global features
        global_features, critical_indexes = torch.max(x, dim=2, keepdim=True)  # (B, 1024, 1), (B, 1024)
        global_features = global_features.view(B, -1) # (B, 1024)

        # Output
        if self.append_local_feat: # for Segmentation
            global_expanded = global_features.unsqueeze(-1) # (B, 1024, 1) # extra dim for broadcasting
            global_expanded = global_expanded.repeat(1, 1, N) # (B, 1024, N)
            locnglob_features = torch.cat((local_features, global_expanded), dim=1) # (B, 1088, N)

            if self.return_critical_indexes:
                return locnglob_features, critical_indexes
            else:
                return locnglob_features
            
        else: # for Classification
            if self.return_critical_indexes:
                return global_features, critical_indexes
            else:
                return global_features

# ============================================================================
# Classification Head
class PointNetClassHead(nn.Module):
    '''' Classification Head '''
    def __init__(self, num_points, num_global_feats=1024, k=2):
        super(PointNetClassHead, self).__init__()

        # get the backbone (only need global features for classification)
        self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=False)

        # MLP for classification
        self.linear1 = nn.Linear(num_global_feats, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, k)

        # batchnorm for the first 2 linear layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        # The paper states that batch norm was only added to the layer 
        # before the classication layer, but another version adds dropout  
        # to the first 2 layers
        self.dropout = nn.Dropout(p=0.3)
        

    def forward(self, x):
        # get global features
        x, crit_idxs, A_feat = self.backbone(x) 

        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dropout(x)
        x = self.linear3(x)

        # return logits
        return x, crit_idxs, A_feat

# ============================================================================
# Segmentation Head
class PointNetSegHead(nn.Module):
    ''' Segmentation Head '''
    def __init__(self, num_points, num_global_feats=1024, m=2):
        super(PointNetSegHead, self).__init__()

        self.num_points = num_points
        self.m = m

        # get the backbone 
        self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=True)

        # shared MLP
        num_features = num_global_feats + 64 # local and global features
        self.conv1 = nn.Conv1d(num_features, 512, kernel_size=1)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=1)
        self.conv4 = nn.Conv1d(128, m, kernel_size=1)

        # batch norms for shared MLP
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)


    def forward(self, x):
        
        # get combined features
        x, crit_idxs, A_feat = self.backbone(x) 

        # pass through shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose(2, 1)
        
        return x, crit_idxs, A_feat


class PointNetSegLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0, size_average=True, dice=False):
        super(PointNetSegLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.dice = dice

        # sanitize inputs
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,(list, np.ndarray)): self.alpha = torch.Tensor(alpha)

        # get Balanced Cross Entropy Loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha)
        

    def forward(self, predictions, targets, pred_choice=None):

        # get Balanced Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions.transpose(2, 1), targets)

        # reformat predictions (b, n, c) -> (b*n, c)
        predictions = predictions.contiguous().view(-1, predictions.size(2)) 
        # get predicted class probabilities for the true class
        pn = F.softmax(predictions)
        pn = pn.gather(1, targets.view(-1, 1)).view(-1)

        # compute loss (negative sign is included in ce_loss)
        loss = ((1 - pn)**self.gamma * ce_loss)
        if self.size_average: loss = loss.mean() 
        else: loss = loss.sum()

        # add dice coefficient if necessary
        if self.dice: return loss + self.dice_loss(targets, pred_choice, eps=1)
        else: return loss


    @staticmethod
    def dice_loss(predictions, targets, eps=1):
        ''' Compute Dice loss, directly compare predictions with truth '''

        targets = targets.reshape(-1)
        predictions = predictions.reshape(-1)

        cats = torch.unique(targets)

        top = 0
        bot = 0
        for c in cats:
            locs = targets == c

            # get truth and predictions for each class
            y_tru = targets[locs]
            y_hat = predictions[locs]

            top += torch.sum(y_hat == y_tru)
            bot += len(y_tru) + len(y_hat)


        return 1 - 2*((top + eps)/(bot + eps)) 


def compute_iou(targets, predictions):

    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)

    intersection = torch.sum(predictions == targets) # true positives
    union = len(predictions) + len(targets) - intersection

    return intersection / union


class PointNetLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0, reg_weight=0, size_average=True):
        super(PointNetLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.size_average = size_average

        # sanitize inputs
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,(list, np.ndarray)): self.alpha = torch.Tensor(alpha)

        # get Balanced Cross Entropy Loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha)

    def forward(self, predictions, targets, A):

        # get batch size
        bs = predictions.size(0)

        # get Balanced Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions, targets)

        # reformat predictions and targets (segmentation only)
        if len(predictions.shape) > 2:
            predictions = predictions.transpose(1, 2) # (b, c, n) -> (b, n, c)
            predictions = predictions.contiguous() \
                                     .view(-1, predictions.size(2)) # (b, n, c) -> (b*n, c)

        # get predicted class probabilities for the true class
        pn = F.softmax(predictions)
        pn = pn.gather(1, targets.view(-1, 1)).view(-1)

        # get regularization term
        if self.reg_weight > 0:
            I = torch.eye(64).unsqueeze(0).repeat(A.shape[0], 1, 1) # .to(device)
            if A.is_cuda: I = I.cuda()
            reg = torch.linalg.norm(I - torch.bmm(A, A.transpose(2, 1)))
            reg = self.reg_weight*reg/bs
        else:
            reg = 0

        # compute loss (negative sign is included in ce_loss)
        loss = ((1 - pn)**self.gamma * ce_loss)
        if self.size_average: return loss.mean() + reg
        else: return loss.sum() + reg

