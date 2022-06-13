import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm



class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "embeds" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)
        # TODO
        
        self.embed_features = embed_features
        self.num_features = num_features
        self.gamma_ = nn.Linear(embed_features, num_features)
        self.bias_ = nn.Linear(embed_features, num_features)

    def forward(self, inputs, embeds):
        #gamma = ... # TODO 
        #bias = ... # TODO
        
        gamma = self.gamma_(embeds)
        bias = self.bias_(embeds)

        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]
        assert bias.shape[0] == inputs.shape[0] and bias.shape[1] == inputs.shape[1]

        #outputs = ... # TODO: apply batchnorm
        outputs = super().forward(inputs)

        return outputs * gamma[..., None, None] + bias[..., None, None]


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()
        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers
        # Don't forget that activation after residual sum cannot be inplace!
        
        self.batchnorm = batchnorm
        self.upsample = upsample
        self.downsample = downsample
        
        
        if self.batchnorm:
            self.abn_1 = AdaptiveBatchNorm(in_channels, embed_channels)
            self.abn_2 = AdaptiveBatchNorm(out_channels, embed_channels)
            
            self.conv_in = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            self.conv_out = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            
            self.relu_1 = nn.ReLU()
            self.relu_2 = nn.ReLU()
            self.activation = nn.Identity()
        else:
            self.relu_1 = nn.ReLU()
            self.relu_2 = nn.ReLU()
            self.conv_in = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            self.conv_out = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            self.activation = nn.ReLU()

        self.skip_ = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        if self.downsample:
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        

    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm
        #pass # TODO
        if self.upsample:
            inputs = F.interpolate(inputs, scale_factor=2)
        

        if self.batchnorm:

            x = self.abn_1(inputs, embeds)
            x = self.relu_1(x)
            x = self.conv_in(x)
            x = self.abn_2(x, embeds)
            x = self.relu_2(x)
            x = self.conv_out(x)
        else:

            x = self.relu_1(inputs)
            x = self.conv_in(x)
            x = self.relu_2(x)
            x = self.conv_out(x)
        inputs = self.skip_(inputs)
        outputs = self.activation(x + inputs)

        if self.downsample:
            outputs = self.avgpool(outputs)
        return outputs


class Generator(nn.Module):
    """
    Generator network (8 points)
    
    TODO:

      - Implement an option to condition the synthesis on trainable class embeddings
        (use nn.Embedding module with noise_channels as the size of each embed)

      - Concatenate input noise with class embeddings (if use_class_condition = True) to obtain input embeddings

      - Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4

      - Forward an input tensor through a convolutional part, 
        which consists of num_blocks PreActResBlocks and performs upsampling by a factor of 2 in each block

      - Each PreActResBlock is additionally conditioned on the input embeddings (via adaptive batch normalization)

      - At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head

      - Apply spectral norm to all conv and linear layers (not the embedding layer)

      - Use Sigmoid at the end to map the outputs into an image

    Notes:

      - The last convolutional layer should map min_channels to 3. With each upsampling you should decrease
        the number of channels by a factor of 2

      - Class embeddings are only used and trained if use_class_condition = True
    """    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_class_condition: bool):
        super(Generator, self).__init__()
        self.output_size = 4 * 2**num_blocks
        # TODO

        self.use_class_condition = use_class_condition
        self.max_channels = max_channels
        
        if use_class_condition:
            embedding_channels = 2 * noise_channels
        else:
            embedding_channels = noise_channels

        self.embedding_ = nn.Embedding(num_classes, noise_channels)
        self.embedding_mapping = spectral_norm(nn.Linear(embedding_channels, 16 * max_channels))
        
        self.conv = nn.ModuleList()
        for i in range(num_blocks-1):
            in_ = max_channels // 2 ** i
            out_ = max_channels // 2 ** (i + 1)
            self.conv.add_module('PreActResBlock'+str(i+1), PreActResBlock(in_, out_,embed_channels=embedding_channels,batchnorm=True,upsample=True))

        self.out_ = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(min_channels),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(min_channels, 3, kernel_size=3, padding=1, bias=True)),
            nn.Sigmoid()
        )
        
    def forward(self, noise, labels):
        # TODO
        
        if self.use_class_condition:
            noise = torch.cat([noise, self.embedding_(labels)], dim=1)

        input_ = self.embedding_mapping(noise).reshape(-1, self.max_channels, 4, 4)

        for block in self.conv:
            input_ = block(input_, noise)



        outputs = self.out_(input_)


        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        return outputs


class Discriminator(nn.Module):
    """
    Discriminator network (8 points)

    TODO:
    
      - Define a convolutional part of the discriminator similarly to
        the generator blocks, but in the inverse order, with downsampling, and
        without batch normalization
    
      - At the end of the convolutional part apply ReLU and sum pooling
    
    TODO: implement projection discriminator head (https://arxiv.org/abs/1802.05637)
    
    Scheme: materials/prgan.png
    
    Notation:
    
      - phi is a convolutional part of the discriminator
    
      - psi is a vector
    
      - y is a class embedding
    
    Class embeddings matrix is similar to the generator, shape: num_classes x max_channels

    Discriminator outputs a B x 1 matrix of realism scores

    Apply spectral norm for all layers (conv, linear, embedding)
    """

    def __init__(self,
                 min_channels: int,
                 max_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_projection_head: bool):
        super(Discriminator, self).__init__()
        # TODO
        
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.use_projection_head = use_projection_head

        self.init_ = nn.Sequential(spectral_norm(nn.Conv2d(3, min_channels, kernel_size=3, padding=1)),
                                   nn.AvgPool2d(kernel_size=2, stride=2))

        self.embedding_ = spectral_norm(nn.Embedding(num_classes, max_channels))
        self.out_ = spectral_norm(nn.Linear(max_channels, 1))


        list_of_modules = [PreActResBlock(min_channels * 2 ** i, 2 * min_channels * 2 ** i, batchnorm=False, downsample=True) for i in range(num_blocks - 1)]

        self.conv = nn.ModuleList(list_of_modules)

    def forward(self, inputs, labels):
        # TODO

        r_cnv = self.init_(inputs)
        for module_ in self.conv:
            r_cnv = module_(r_cnv)
        res = torch.sum(F.relu(r_cnv, inplace=True), dim=(-1, -2))
        scores = self.out_(res).squeeze(dim=1)
        if self.use_projection_head:
            scores += torch.diag(torch.inner(res, self.embedding_(labels)))

        assert scores.shape == (inputs.shape[0],)
        return scores
