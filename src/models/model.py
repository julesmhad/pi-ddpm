"""
U-Net Architectures for Denoising Diffusion Probabilistic Models (DDPM)

This module implements U-Net based architectures specifically designed for DDPMs, featuring:
- Noise-conditional U-Net with skip connections
- Instance normalization for stable training
- Attention mechanisms for improved feature learning
- Support for both standard and physics-informed diffusion models

Key Components:
1. tile_gamma: Utility for broadcasting noise levels
2. tile_embedding: Utility for broadcasting embeddings
3. UNet_ddpm: Main U-Net architecture for DDPM with noise conditioning
4. UNet: Standard U-Net architecture for comparison/ablation
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    UpSampling3D,
    Concatenate,
    Add,
    Lambda,
    Conv3DTranspose,
    Attention,
)
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def tile_gamma(x):
    """
    Broadcasts gamma (noise level) to match spatial dimensions of input tensor.
    
    This function is used to condition the network on the noise level at each
    diffusion timestep by spatially tiling the noise level parameter.
    
    Mathematical Formulation:
        output = 1_{D×H×W×1} ⊗ γ
    where:
        1_{D×H×W×1}: Tensor of ones with shape (D, H, W, 1)
        γ: Scalar noise level
        ⊗: Element-wise multiplication with broadcasting
    
    Args:
        x: List containing [gamma, input_tensor] where:
           - gamma: 1D tensor of shape (batch_size,), noise levels
           - input_tensor: 5D tensor of shape (batch_size, D, H, W, C)
           
    Returns:
        5D tensor of shape (batch_size, D, H, W, 1) where gamma is tiled spatially
    """
    gamma = x[0]
    input_tensor = x[1]
    return tf.ones_like(input_tensor) * tf.expand_dims(tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1), axis=-1)


def tile_embedding(x):
    """
    Broadcasts embedding vectors to match spatial dimensions of input tensor.
    
    This function is used to condition the network on additional information
    (e.g., class labels, style vectors) by spatially tiling the embedding.
    
    Mathematical Formulation:
        output = 1_{D×H×W×1} ⊗ e
    where:
        1_{D×H×W×1}: Tensor of ones with shape (D, H, W, 1)
        e: Embedding vector of shape (embedding_dim,)
        ⊗: Element-wise multiplication with broadcasting
    
    Args:
        x: List containing [embedding, input_tensor] where:
           - embedding: 2D tensor of shape (batch_size, embedding_dim)
           - input_tensor: 5D tensor of shape (batch_size, D, H, W, C)
           
    Returns:
        5D tensor of shape (batch_size, D, H, W, embedding_dim) where embedding is tiled spatially
    """
    emb = x[0]
    input_tensor = x[1]
    return tf.ones_like(input_tensor) * tf.expand_dims(tf.expand_dims(tf.expand_dims(emb, axis=-1), axis=-1), axis=-1)


def UNet_ddpm(
        shape,
        previous_tensor=None,
        inputs=None,
        gamma_inp=None,
        full_model=False,
        noise_input=False,
        z_enc=False,
        out_filters=2):
    """
    U-Net architecture for Denoising Diffusion Probabilistic Models (DDPM).
    
    This U-Net variant is specifically designed for DDPM with the following features:
    - Noise conditioning at multiple scales
    - Skip connections between encoder and decoder
    - Optional attention mechanisms
    - Support for residual connections
    
    Architecture:
    1. Encoder path (contracting):
       - Repeated blocks of Conv2D -> InstanceNorm -> Softplus
       - MaxPooling for downsampling
       - Feature dimensions: 64 -> 128 -> 256 -> 512 -> 1024
    
    2. Bottleneck:
       - Additional convolutional layers at the lowest resolution
       - Optional attention mechanism
    
    3. Decoder path (expanding):
       - Transposed convolutions for upsampling
       - Skip connections from encoder
       - Feature dimensions: 1024 -> 512 -> 256 -> 128 -> 64
    
    4. Output:
       - Final convolution to produce output with 'out_filters' channels
    
    Args:
        shape: Tuple, input shape (H, W, C)
        previous_tensor: Optional tensor from previous timestep for temporal modeling
        inputs: Optional input tensor, if None creates a new Input layer
        gamma_inp: Optional noise level input tensor
        full_model: Boolean, if True returns a complete Keras Model
        noise_input: Boolean, if True adds noise input channel
        z_enc: Boolean or tensor, if not False adds latent variable conditioning
        out_filters: Integer, number of output channels
        
    Returns:
        If full_model is True, returns a Keras Model
        Otherwise, returns tuple of (output_tensor, bottleneck_tensor)
    """
    if inputs is None:
        inputs = Input(shape)
    if gamma_inp is None:
        gamma_inp = Input((1,))

    # Noise input branch (optional)
    if noise_input:
        noise_in = Input((128,))
        # c_noise = Concatenate()([noise_in, noise_in])
        # c_inpt = Concatenate()([inputs, noise_in])
        conv1 = Conv3D(64, 3, activation='softplus', padding='same')(inputs)
    else:
        conv1 = Conv3D(64, 3, activation='softplus', padding='same')(inputs)

    # Noise conditioning
    tiled_gamma = Lambda(tile_gamma)([gamma_inp, inputs])
    conv_gamma = Conv3D(64, 7, activation='softplus', padding='same')(tiled_gamma)
    conv1 = Add()([conv_gamma, conv1])
    conv1 = GroupNormalization(groups=-1)(conv1)
    conv1 = Conv3D(64, 3, activation='softplus', padding='same')(conv1)
    conv1 = GroupNormalization(groups=-1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = (Conv3D(128, 3, activation='softplus', padding='same'))(pool1)
    conv2 = GroupNormalization(groups=-1)(conv2)
    conv2 = Conv3D(128, 3, activation='softplus', padding='same')(conv2)
    conv2 = GroupNormalization(groups=-1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(256, 3, activation='softplus', padding='same')(pool2)
    conv3 = GroupNormalization(groups=-1)(conv3)
    conv3 = Conv3D(256, 3, activation='softplus', padding='same')(conv3)
    conv3 = GroupNormalization(groups=-1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(512, 3, activation='softplus', padding='same')(pool3)
    conv4 = GroupNormalization(groups=-1)(conv4)
    conv4 = Conv3D(512, 3, activation='softplus', padding='same')(conv4)
    conv4 = GroupNormalization(groups=-1)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    # pool4 = Attention(512)(pool4)

    conv5 = Conv3D(512, 3, activation='softplus', padding='same')(pool4)
    conv5 = GroupNormalization(groups=-1)(conv5)
    conv5 = Conv3D(1024, 3, activation='softplus', padding='same')(conv5)
    conv5 = GroupNormalization(groups=-1)(conv5)

    print(K.int_shape(conv5))
    if previous_tensor is not None:
        previous_tensor = Conv3DTranspose(1024, 3, strides=2, activation='softplus', padding='same')(previous_tensor)
        previous_tensor = Attention(1024)(previous_tensor)
        conv5 = Add()([conv5, previous_tensor])

    if z_enc is not False:
        gamma_vec = Concatenate()([z_enc, conv5])
    else:
        gamma_vec = conv5

    up6 = Conv3D(
        512,
        2,
        activation='softplus',
        padding='same')(
        Conv3DTranspose(512, 3, activation='softplus', padding='same', strides=2)(
            gamma_vec))

    merge6 = Concatenate()([conv4, up6])
    conv6 = Conv3D(512, 3, activation='softplus', padding='same')(merge6)
    conv6 = GroupNormalization(groups=-1)(conv6)
    conv6 = Conv3D(512, 3, activation='softplus', padding='same')(conv6)
    conv6 = GroupNormalization(groups=-1)(conv6)
    # conv6 = Attention(512)(conv6)
    up7 = Conv3D(
        256,
        2,
        activation='softplus',
        padding='same')(
        Conv3DTranspose(256, 3, activation='softplus', padding='same', strides=2)(
            conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv3D(256, 3, activation='softplus', padding='same')(merge7)
    conv7 = GroupNormalization(groups=-1)(conv7)
    conv7 = Conv3D(256, 3, activation='softplus', padding='same')(conv7)
    conv7 = GroupNormalization(groups=-1)(conv7)
    # conv7 = Attention(256)(conv7)
    up8 = Conv3D(
        128,
        2,
        activation='softplus',
        padding='same')(
        Conv3DTranspose(128, 3, activation='softplus', padding='same', strides=2)(
            conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv3D(128, 3, activation='softplus', padding='same')(merge8)
    conv8 = GroupNormalization(groups=-1)(conv8)
    conv8 = Conv3D(128, 3, activation='softplus', padding='same')(conv8)
    conv8 = GroupNormalization(groups=-1)(conv8)
    up9 = Conv3D(
        64,
        2,
        activation='softplus',
        padding='same')(
        Conv3DTranspose(64, 3, activation='softplus', padding='same', strides=2)(conv8))

    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv3D(64, 3, activation='softplus', padding='same')(merge9)
    conv9 = GroupNormalization(groups=-1)(conv9)
    conv9 = Conv3D(64, 3, activation='softplus', padding='same')(conv9)
    conv9 = GroupNormalization(groups=-1)(conv9)
    conv9 = Conv3D(32, 3, activation='softplus', padding='same')(conv9)
    conv9 = GroupNormalization(groups=-1)(conv9)
    conv10 = Conv3D(out_filters, 1, activation='linear', padding='same')(conv9)
    if full_model and noise_input:
        out = Add()([inputs, conv10])
        model = Model([noise_in, inputs], out)
        model.summary()
        return model
    elif full_model:
        model = Model([inputs, gamma_inp], conv10)
        return model
    return conv10, conv5


def UNet(shape, out_channels=3, inputs=None):
    """
    Standard U-Net architecture for image-to-image translation.
    
    A more traditional U-Net implementation with the following features:
    - ReLU activations
    - Instance normalization
    - Skip connections
    - No noise conditioning
    
    Architecture:
    1. Encoder: 4 downsampling blocks
       - Each block: Conv -> InstanceNorm -> ReLU -> Conv -> InstanceNorm -> ReLU -> MaxPool
    2. Bottleneck: Two convolutional layers at lowest resolution
    3. Decoder: 4 upsampling blocks with skip connections
       - Each block: UpSampling -> Conv -> Concat(skip) -> Conv -> InstanceNorm -> ReLU -> Conv -> InstanceNorm -> ReLU
    
    Args:
        shape: Tuple, input shape (H, W, C)
        out_channels: Integer, number of output channels
        inputs: Optional input tensor, if None creates a new Input layer
        
    Returns:
        Keras Model with U-Net architecture
    """
    if inputs is None:
        inputs = Input(shape)
    
    # conv1 = Conv2D(64, 3, activation='softplus', padding='same')(inputs)
    conv1 = Conv3D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = GroupNormalization(groups=-1)(conv1)
    conv1 = Conv3D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = GroupNormalization(groups=-1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = GroupNormalization(groups=-1)(conv2)
    conv2 = Conv3D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = GroupNormalization(groups=-1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = GroupNormalization(groups=-1)(conv3)
    conv3 = Conv3D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = GroupNormalization(groups=-1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = GroupNormalization(groups=-1)(conv4)
    conv4 = Conv3D(1024, 3, activation='relu', padding='same')(conv4)
    conv4 = GroupNormalization(groups=-1)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    # pool4 = Attention(1024)(pool4)
    conv5 = Conv3D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = GroupNormalization(groups=-1)(conv5)
    conv5 = Conv3D(1024, 3, activation='relu', padding='same')(conv5)

    # print('conv5:', conv5.shape)
    cur_hidden = GroupNormalization(groups=-1)(conv5)

    up6 = Conv3D(512, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(cur_hidden))

    # print('u6:', conv4.shape, up6.shape)

    merge6 = Concatenate()([conv4, up6])
    conv6 = Conv3D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = GroupNormalization(groups=-1)(conv6)
    conv6 = Conv3D(512, 3, activation='relu', padding='same')(conv6)
    conv6 = GroupNormalization(groups=-1)(conv6)

    # conv6 = Attention(512)(conv6)
    up7 = Conv3D(256, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv3D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = GroupNormalization(groups=-1)(conv7)
    conv7 = Conv3D(256, 3, activation='relu', padding='same')(conv7)
    conv7 = GroupNormalization(groups=-1)(conv7)

    # conv7 = Attention(256)(conv7)
    up8 = Conv3D(128, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv3D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = GroupNormalization(groups=-1)(conv8)
    conv8 = Conv3D(128, 3, activation='relu', padding='same')(conv8)
    conv8 = GroupNormalization(groups=-1)(conv8)

    up9 = Conv3D(64, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv3D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = GroupNormalization(groups=-1)(conv9)
    conv9 = Conv3D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = GroupNormalization(groups=-1)(conv9)
    conv9 = Conv3D(32, 3, activation='relu', padding='same')(conv9)
    conv9 = GroupNormalization(groups=-1)(conv9)

    # last layer
    conv10 = Conv3D(out_channels, 1, activation='linear', padding='same')(conv9)
    model = Model(inputs, conv10)

    return model
