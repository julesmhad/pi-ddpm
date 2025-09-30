import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Lambda, Conv2D, UpSampling3D, UpSampling3D
from tensorflow.keras.models import Model
from src.models.model import UNet_ddpm
from src.training_utils.forward_models import forward_model_conv
import tensorflow.keras.backend as K



def obtain_noisy_sample(x):
    """
    Applies forward diffusion process by adding noise to the input sample.
    
    Args:
        x: List containing [x_0, gamma] where:
           - x_0: Original clean sample
           - gamma: Noise level parameter
           
    Returns:
        List containing [noisy_sample, noise_sample] where:
        - noisy_sample: The noised version of x_0
        - noise_sample: The random noise that was added
    """
    x_0 = x[0]
    gamma = x[1]
    gamma_vec = tf.expand_dims(tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1), axis=-1)
    noise_sample = tf.random.normal(tf.shape(x_0))
    return [tf.sqrt(gamma_vec) * x_0 + tf.sqrt(1 - gamma_vec) * noise_sample, noise_sample]


def obtain_clean_sample(x):
    """
    Reconstructs the clean sample from a noisy sample using the predicted noise.
    This is part of the reverse diffusion process.
    
    Args:
        x: List containing [pred_noise, gamma, x_t] where:
           - pred_noise: Predicted noise from the model
           - gamma: Noise level parameter
           - x_t: Noisy sample at timestep t
           
    Returns:
        x_0: The reconstructed clean sample
    """
    pred_noise = x[0]
    gamma = x[1]
    x_t = x[2]
    gamma_vec = tf.expand_dims(tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1), axis=-1)
    x_0 = (x_t - tf.sqrt(1 - gamma_vec) * pred_noise) / tf.clip_by_value(tf.sqrt(gamma_vec), 1e-5, 1)
    return x_0


def obtain_sr_t(x):
    """
    Applies super-resolution transformation to the input by combining
    a high-resolution noisy sample with a low-resolution version.
    
    Args:
        x: List containing [x_t, x_lr, gamma] where:
           - x_t: High-resolution noisy sample
           - x_lr: Low-resolution version of the sample
           - gamma: Noise level parameter
           
    Returns:
        y_t: The transformed sample with super-resolution information
    """
    x_t = x[0]
    x_lr = x[1]
    gamma = x[2]
    gamma_vec = tf.expand_dims(tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1), axis=-1)
    y_t = x_t + tf.sqrt(gamma_vec) * x_lr
    return y_t


def norm_01(x):
    """
    Normalizes the input tensor to the range [0, 1] using min-max scaling.
    
    Args:
        x: Input tensor to be normalized
        
    Returns:
        Normalized tensor with values in [0, 1] range
    """
    return tf.math.divide_no_nan(x - tf.stop_gradient(tf.math.reduce_min(x, axis=(1, 2, 3, 4), keepdims=True)),
                                 (tf.stop_gradient(tf.math.reduce_max(x, axis=(1, 2, 3, 4), keepdims=True)) -
                                  tf.stop_gradient(tf.math.reduce_min(x, axis=(1, 2, 3, 4), keepdims=True))))


def norm_2(x):
    """
    Computes the L2 norm (Euclidean norm) of the input tensor across spatial dimensions.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor containing the L2 norm values for each sample in the batch
    """
    return tf.reduce_sum(x ** 2, axis=(1, 2, 3, 4), keepdims=True)


def forward_model_gradient(x):
    """
    Computes the gradient of the forward model used in the physics-informed loss.
    This is part of the physics-informed training process that guides the diffusion.
    
    Args:
        x: List containing [x0_pred, dirty_img, x_t, gamma, noise] where:
           - x0_pred: Predicted clean sample
           - dirty_img: Degraded/observed image
           - x_t: Current noisy sample
           - gamma: Noise level parameter
           - noise: Random noise tensor
           
    Returns:
        grad_pi: Gradient of the physics-informed loss with respect to x_t
    """
    # lam = 1

    dirty_img = x[1]
    pred_dirty = norm_01(upsample_3d(x[0], K.int_shape(dirty_img))) * 2 - 1
    x_t = x[2]
    gamma = x[3]
    noise = x[4]
    gamma_vec = tf.expand_dims(tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1), axis=-1)
    pi_l2 = tf.reduce_sum(tf.square(pred_dirty - dirty_img), axis=(1, 2, 3, 4), keepdims=True)
    grad_pi = tf.gradients(pi_l2, x_t, unconnected_gradients='zero')[0]
    # lam = tf.math.reduce_max(grad_pi, axis=(1, 2, 3), keepdims=True)
    return grad_pi


def rescale(x):
    """
    Downsamples the input tensor by a factor of 2 and scales the values.
    
    Args:
        x: Input tensor
        
    Returns:
        Downsampled and scaled tensor
    """
    # Resizing 3D volumes using pooling, as tf.image.resize is for 2D
    return tf.nn.pool(x, window_shape=(2, 2, 2), pooling_type='AVG', strides=(2, 2, 2), padding='SAME') * 0.5 + 0.5


def upsample_3d(tensor, target_shape):
    """Upsamples a 5D tensor to a target shape using UpSampling3D layer."""
    input_shape = K.int_shape(tensor)
    
    # Calculate scaling factors
    depth_factor = target_shape[1] / input_shape[1]
    height_factor = target_shape[2] / input_shape[2]
    width_factor = target_shape[3] / input_shape[3]

    # Ensure factors are integers, as UpSampling3D expects integer factors
    size = (int(depth_factor), int(height_factor), int(width_factor))

    # Apply UpSampling3D
    upsampled_tensor = UpSampling3D(size=size)(tensor)
    
    return upsampled_tensor


def pi_model(input_shape_noisy, volume_shape, out_channels):
    """
    Defines the physics-informed neural network model.
    
    Args:
        input_shape_noisy: Shape of the noisy input data
        volume_shape: Shape of the volume data
        out_channels: Number of output channels
        
    Returns:
        Model: Physics-informed neural network model
    """
    noisy_input = Input(volume_shape)
    ref_frame = Input(input_shape_noisy)
    
    # Upsample ref_frame to match the shape of noisy_input before concatenation
    ref_frame_upsampled = UpSampling3D(size=(1, 2, 2))(ref_frame)

    c_inpt = Concatenate()([noisy_input, ref_frame_upsampled])
    gamma_inp = Input((1,))
    noise_out, _ = UNet_ddpm(volume_shape, inputs=c_inpt, gamma_inp=gamma_inp,
                             out_filters=out_channels, z_enc=False)
    # m_dfcan = dfcan_ddpm(K.int_shape(c_inpt)[1:], out_filters=out_channels)
    # noise_out = m_dfcan([c_inpt, gamma_inp])

    model_out = Model([noisy_input, ref_frame, gamma_inp], noise_out)

    return model_out


def train_model(input_shape_condition, volume_shape, out_channels, noise_est=True):
    """
    Defines the training model for the physics-informed neural network.
    
    Args:
        input_shape_condition: Shape of the conditioning input data
        volume_shape: Shape of the volume data
        out_channels: Number of output channels (default: 1)
        noise_est: Flag to indicate if noise estimation is enabled (default: False)
        physics_params: Optional dictionary containing physics parameters for the model
                        (default: None, which uses hardcoded default values)
        
    Returns:
        n_model: Physics-informed neural network model
        train_model: Training model for the physics-informed neural network
    """


    ground_truth = Input(volume_shape)
    dirty_img = Input(input_shape_condition)
    kernel_conv = Input((volume_shape[0] // 2, volume_shape[1] // 2, volume_shape[2] // 2, volume_shape[3]))
    gamma_inp = Input((1,))
    # ct_inp_shape = input_shape_noisy[:-1] + (input_shape_noisy[-1] + input_shape_noisy[-1],)
    # t_tiled = Lambda(tile_gamma)([t_inp, ground_truth])
    if noise_est:
        n_model = pi_model(input_shape_condition, volume_shape,
                           out_channels)
    else:
        n_model = pi_model(input_shape_condition, volume_shape,
                           out_channels + 1)  # adding the gradient as separate output to control nu outside of training

    n_sample = Lambda(obtain_noisy_sample)([ground_truth, gamma_inp])
    if not noise_est:
        noiseless_img = Input(input_shape_condition)
        noise_pred = n_model([n_sample[0], dirty_img, gamma_inp])
        delta_noise = n_sample[1] - noise_pred[..., 0:1]
        y_t = Lambda(obtain_sr_t)([n_sample[0], dirty_img, gamma_inp])
        pred_dirty = tf.abs(Lambda(forward_model_conv)([kernel_conv, Lambda(rescale)(y_t)]))
        grad = Lambda(forward_model_gradient)([pred_dirty, noiseless_img, y_t, gamma_inp, n_sample[1]])
        delta_grad = tf.stop_gradient(grad) - noise_pred[..., 1:]
        train_model = Model([ground_truth, dirty_img, noiseless_img, kernel_conv, gamma_inp], [delta_noise, delta_grad])
        train_model.summary()
    else:
        noise_pred = n_model([n_sample[0], dirty_img, gamma_inp])
        delta_noise = noise_pred - n_sample[1]
        train_model = Model([ground_truth, dirty_img, gamma_inp], delta_noise)
        train_model.summary()

    return n_model, train_model
