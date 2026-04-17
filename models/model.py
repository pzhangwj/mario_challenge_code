import torch.nn as nn
import timm
import torch

#input [B, 3, H, W] x2
class MarioModelT1(nn.Module):
    """
    A convolutional neural network model for image classification tasks, built upon a ResNet-50 backbone. 
    This model leverages the TIMM library to create the backbone, allowing for easy modification and experimentation 
    with different pre-trained models. The model is designed for binary or multi-class classification tasks.

    Note:
    The model by default does not load pre-trained weights due to the absence of internet access on the cluster. 
    To use a pre-trained model, download the weights manually and adjust the code to load them as needed.

    Attributes:
        backbone (timm.models.resnet.ResNet): The ResNet-50 model used as the backbone for feature extraction.
                                               The model is modified to output the desired number of classes based on 
                                               the `num_classes` parameter.

    Args:
        num_classes (int): The number of classes for the classification task. Default is 2 for binary classification.
    model_type : input_fusion [1,3,200,768] x2
                 channel_fusion [1,2,200,768] x1
                 
    """
    def __init__(self, backbone, pretrained, num_classes, model_type):
        
        super(MarioModelT1, self).__init__()
        # Initialize the backbone ResNet-50 model without pre-trained weights
          # Set num_classes to 0 to get the feature extractor
            
        self.model_type = model_type
        
        if self.model_type == "features_fusion" : #[1, 2048] x2
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3)
            self.fc = nn.Linear(self.backbone.num_features * 2, num_classes)  # Adjusted for two inputs
        
        if self.model_type == "channels_fusion":
            
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=2)
            self.fc = nn.Linear(self.backbone.num_features, num_classes) 
            
    def forward(self, x1, x2):
        """
        Defines the forward pass of the model with two inputs.

        Args:
            x1 (torch.Tensor): The first input tensor containing a batch of images.
                               Tensor dimensions should be `(batch_size, channels, height, width)`.
            x2 (torch.Tensor): The second input tensor containing a batch of images.
                               Tensor dimensions should be `(batch_size, channels, height, width)`.

        Returns:
            torch.Tensor: The output tensor containing the model predictions. The tensor's dimensions will be 
                          `(batch_size, num_classes)`, where `num_classes` is the number provided during model initialization.
        """
        
        if self.model_type == "features_fusion":
            # Pass both inputs through the backbone to get their representations
            x1 = self.backbone(x1) #[1, 2048][1, 2048]
    #         print(x1.shape)
            x2 = self.backbone(x2) #[1, 2048]
    #         print(x2.shape)

            # Concatenate the representations
            x = torch.cat((x1, x2), dim=1)

            # Pass the concatenated representation through the fully connected layer
            x = self.fc(x)
            
        if self.model_type == "channels_fusion":
            
            # Concatenate the inputs
            x = torch.cat((x1, x2), dim=1) #[2, 2048]
            
            x = self.backbone(x)
            
            x = self.fc(x)
        
        return x
