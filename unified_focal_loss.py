from turtle import forward
import warnings
from typing import Optional, Sequence, Union

import torch
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction


class AsymmetricFocalTverskyLoss(_Loss):
    """
    AsymmetricFocalTverskyLoss is a variant of FocalTverskyLoss, which attentions to the foreground class.
    
    Reimplementation of the Asymmetric Focal Tversky Loss (with a build-in sigmoid activation) described in:
    
    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    """
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        delta: float = 0.7,
        gamma: float = 0.75,
        epsilon:float = 1e-7,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            include_background (bool, optional): If False channel index 0 (background category) is excluded from the calculation. Defaults to True.
            to_onehot_y (bool, optional): whether to convert `y` into the one-hot format. Defaults to False.
            delta (float, optional): weight of the backgrand. Defaults to 0.7.
            gamma (float, optional): value of the exponent gamma in the definition of the Focal loss  . Defaults to 0.75.
            epsilon (float, optional): it's define a very small number each time. simmily smooth value. Defaults to 1e-7.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n_pred_ch = y_pred.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y_true = one_hot(y_true, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                y_true = y_true[:, 1:]
                y_pred = y_pred[:, 1:]

        if y_true.shape != y_pred.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")
        
        # clip the prediction to avoid NaN
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = list(range(2, len(y_pred.shape)))
        
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        # Calculate losses separately for each class, enhancing both classes
        back_dice = (1-dice_class[:,0])
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma)

        # Average class scores
        loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))
        return loss

class AsymmetricFocalLoss(_Loss):
    """
        AsymmetricFocalLoss is a variant of FocalTverskyLoss, which attentions to the foreground class.
        
        Reimplementation of the Asymmetric Focal Loss (with a build-in sigmoid activation) described in:
        
        - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
        Michael Yeung, Computerized Medical Imaging and Graphics
    """
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        delta: float = 0.7,
        gamma: float = 2,
        epsilon:float = 1e-7,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ):
        """
        Args:
            include_background (bool, optional): If False channel index 0 (background category) is excluded from the calculation. Defaults to True.
            to_onehot_y (bool, optional): whether to convert `y` into the one-hot format. Defaults to False.
            delta (float, optional): weight of the backgrand. Defaults to 0.7.
            gamma (float, optional): value of the exponent gamma in the definition of the Focal loss  . Defaults to 0.75.
            epsilon (float, optional): it's define a very small number each time. simmily smooth value. Defaults to 1e-7.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n_pred_ch = y_pred.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y_true = one_hot(y_true, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                y_true = y_true[:, 1:]
                y_pred = y_pred[:, 1:]

        if y_true.shape != y_pred.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")
        
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)
        
        back_ce = torch.pow(1 - y_pred[:,0], self.gamma) * cross_entropy[:,0]
        back_ce = (1 - self.delta) * back_ce
        
        fore_ce = cross_entropy[:,1]
        fore_ce = self.delta * fore_ce
        print(back_ce.shape, fore_ce.shape)
        
        loss = torch.mean(torch.sum(torch.stack([back_ce,fore_ce], axis=1), axis=1))
        return loss

class SymmetricFocalTverskyLoss(_Loss):
    """
        SymmetricFocalTverskyLoss is a variant of FocalTverskyLoss.
        
        Reimplementation of the Symmetric Focal Tversky Loss (with a build-in sigmoid activation) described in:
        
        - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
        Michael Yeung, Computerized Medical Imaging and Graphics
    """
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        delta: float = 0.7,
        gamma: float = 2,
        epsilon:float = 1e-7,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ):
        """
        Args:
            include_background (bool, optional): If False channel index 0 (background category) is excluded from the calculation. Defaults to True.
            to_onehot_y (bool, optional): whether to convert `y` into the one-hot format. Defaults to False.
            delta (float, optional): weight of the backgrand. Defaults to 0.7.
            gamma (float, optional): value of the exponent gamma in the definition of the Focal loss  . Defaults to 0.75.
            epsilon (float, optional): it's define a very small number each time. simmily smooth value. Defaults to 1e-7.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n_pred_ch = y_pred.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y_true = one_hot(y_true, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                y_true = y_true[:, 1:]
                y_pred = y_pred[:, 1:]

        if y_true.shape != y_pred.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")
        
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        # axis = identify_axis(y_true.size())
        axis = list(range(2, len(y_pred.shape)))
        
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)
        
        back_dice = (1-dice_class[:,0]) * torch.pow(1-dice_class[:,0], -self.gamma)
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma)
        loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))
        
        return loss
        


class SymmetricFocalLoss(_Loss):
    """
        SymmetricFocalLoss is a variant of FocalTverskyLoss.
        
        Reimplementation of the Symmetric Focal Loss (with a build-in sigmoid activation) described in:
        
        - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
        Michael Yeung, Computerized Medical Imaging and Graphics
    """
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        delta: float = 0.7,
        gamma: float = 2,
        epsilon:float = 1e-7,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n_pred_ch = y_pred.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y_true = one_hot(y_true, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                y_true = y_true[:, 1:]
                y_pred = y_pred[:, 1:]

        if y_true.shape != y_pred.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")
        
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)
        
        back_ce = torch.pow(1 - y_pred[:,0], self.gamma) * cross_entropy[:,0]
        back_ce = (1 - self.delta) * back_ce
        
        fore_ce = torch.pow(1 - y_pred[:,1], self.gamma) * cross_entropy[:,1]
        fore_ce = self.delta * fore_ce
        
        loss = torch.mean(torch.sum(torch.stack([back_ce,fore_ce], axis=1), axis=1))
        return loss

class AsymmetricUnifiedFocalLoss(_Loss):
    """
        AsymmetricUnifiedFocalLoss is a variant of Focal Loss.
        
        Reimplementation of the Asymmetric Unified Focal Tversky Loss (with a build-in sigmoid activation) described in:
        
        - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
        Michael Yeung, Computerized Medical Imaging and Graphics
    """
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
        gamma: float = 2.0,
        delta: float = 0.7,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ):
        """
        Args:
            include_background (bool, optional): If False channel index 0 (background category) is excluded from the calculation. Defaults to True.
            to_onehot_y (bool, optional): whether to convert `y` into the one-hot format. Defaults to False.
            delta (float, optional): weight of the backgrand. Defaults to 0.7.
            gamma (float, optional): value of the exponent gamma in the definition of the Focal loss  . Defaults to 0.75.
            epsilon (float, optional): it's define a very small number each time. simmily smooth value. Defaults to 1e-7.
            weight (Optional[Union[Sequence[float], float, int, torch.Tensor]]): weight for each loss function, if it's none it's 0.5. Defaults to None.
            
        Example:
            >>> import torch
            >>> from monai.losses import AsymmetricUnifiedFocalLoss
            >>> pred = torch.ones((1,1,32,32), dtype=torch.float32)
            >>> grnd = torch.ones((1,1,32,32), dtype=torch.int64)
            >>> fl = AsymmetricUnifiedFocalLoss(to_onehot_y=True)
            >>> fl(pred, grnd)
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.gamma = gamma
        self.delta = delta
        self.weight: Optional[Union[Sequence[float],float, int, torch.Tensor]] = weight
        self.asy_focal_loss = AsymmetricFocalLoss(gamma=self.gamma, delta=self.delta)
        self.asy_focal_tversky_loss = AsymmetricFocalTverskyLoss(gamma=self.gamma, delta=self.delta)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n_pred_ch = y_pred.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y_true = one_hot(y_true, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                y_true = y_true[:, 1:]
                y_pred = y_pred[:, 1:]

        if y_true.shape != y_pred.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")
    
        asy_focal_loss = self.asy_focal_loss(y_pred, y_true)
        asy_focal_tversky_loss = self.asy_focal_tversky_loss(y_pred, y_true)
        if self.weight is None:
            loss = (asy_focal_loss + asy_focal_tversky_loss) / 2
        else:
            loss = self.weight * asy_focal_loss + (1 - self.weight) * asy_focal_tversky_loss
        
        return loss
        
class SymmetricUnifiedFocalLoss(_Loss):
    """
        SymmetricUnifiedFocalLoss is a variant of Focal Loss.
        
        Reimplementation of the Symmetric Unified Focal Tversky Loss (with a build-in sigmoid activation) described in:
        
        - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
        Michael Yeung, Computerized Medical Imaging and Graphics
    """
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
        gamma: float = 2.0,
        delta: float = 0.7,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ):
        """
        Args:
            include_background (bool, optional): If False channel index 0 (background category) is excluded from the calculation. Defaults to True.
            to_onehot_y (bool, optional): whether to convert `y` into the one-hot format. Defaults to False.
            delta (float, optional): weight of the backgrand. Defaults to 0.7.
            gamma (float, optional): value of the exponent gamma in the definition of the Focal loss  . Defaults to 0.75.
            epsilon (float, optional): it's define a very small number each time. simmily smooth value. Defaults to 1e-7.
            weight (Optional[Union[Sequence[float], float, int, torch.Tensor]]): weight for each loss function, if it's none it's 0.5. Defaults to None.
            
        Example:
            >>> import torch
            >>> from monai.losses import SymmetricUnifiedFocalLoss
            >>> pred = torch.ones((1,1,32,32), dtype=torch.float32)
            >>> grnd = torch.ones((1,1,32,32), dtype=torch.int64)
            >>> fl = SymmetricUnifiedFocalLoss(to_onehot_y=True)
            >>> fl(pred, grnd)
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.gamma = gamma
        self.delta = delta
        self.weight: Optional[Union[Sequence[float],float, int, torch.Tensor]] = weight
        self.sym_focal_loss = SymmetricFocalLoss(gamma=self.gamma, delta=self.delta)
        self.sym_focal_tversky_loss = SymmetricFocalTverskyLoss(gamma=self.gamma, delta=self.delta)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n_pred_ch = y_pred.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y_true = one_hot(y_true, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                y_true = y_true[:, 1:]
                y_pred = y_pred[:, 1:]

        if y_true.shape != y_pred.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")
    
        sym_focal_loss = self.sym_focal_loss(y_pred, y_true)
        sym_focal_tversky_loss = self.sym_focal_tversky_loss(y_pred, y_true)
        if self.weight is None:
            loss = (sym_focal_loss + sym_focal_tversky_loss) / 2
        else:
            loss = self.weight * sym_focal_loss + (1 - self.weight) * sym_focal_tversky_loss
        
        return loss