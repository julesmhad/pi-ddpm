import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.pi_ddpm import train_model
from src.training_utils.ddpm_utils import variance_schedule
from scripts.data_scripts.sim_loader import load_sim_stacks


def norm_2(x):
    return np.sqrt(np.sum(x ** 2, axis=(1, 2, 3), keepdims=True) + 1e-8)


def ddpm_reconstruct(fx, timesteps_test, p_model, smooth_factor=0.01, reg_type='l1', nu_lr=0.01):
    pred_sr = np.random.normal(0, 1, fx.shape)
    gamma_vec_test, alpha_vec_test = variance_schedule(timesteps_test, schedule_type='linear', min_beta=1e-4,
                                                       max_beta=3e-2)
    grad_h = np.zeros_like(pred_sr)
    for t in tqdm(range(timesteps_test, 0, -1)):
        z = np.random.normal(0, 1, fx.shape)
        if t == 1:
            z = 0
        alpha_t = alpha_vec_test[t - 1]
        gamma_t = gamma_vec_test[t - 1]
        gamma_tm1 = gamma_vec_test[t - 2]
        beta_t = 1 - alpha_t

        gamma_t_inp = np.ones((fx.shape[0], 1)) * np.reshape(gamma_t, (1, 1))
        beta_factor = (1 - gamma_tm1) * beta_t / (1 - gamma_t)
        sigma = beta_factor
        pred_params = p_model.predict([pred_sr, fx, gamma_t_inp], verbose=False)
        if pred_params.shape[-1] == 2:
            pred_noise = pred_params[..., 0:1]
            pred_grad = pred_params[..., 1:]
            pred_grad = pred_grad * nu_lr * np.sqrt(gamma_t) / norm_2(pred_grad)
        else:
            pred_noise = pred_params
            pred_grad = np.zeros_like(pred_params)

        alpha_factor = beta_t / np.sqrt(1 - gamma_t)

        if reg_type == 'l1':
            pred_sr = 1 / np.sqrt(alpha_t) * (pred_sr - alpha_factor * pred_noise) + np.sqrt(
                sigma) * z - smooth_factor * np.sign(pred_sr) * np.sqrt(sigma) - pred_grad * np.sqrt(sigma)
        else:  # l2
            pred_sr = 1 / np.sqrt(alpha_t) * (pred_sr - alpha_factor * pred_noise) + np.sqrt(
                sigma) * z - smooth_factor * np.sqrt(sigma) * (2 * pred_sr) - pred_grad * np.sqrt(sigma)
        if t == timesteps_test // 2:
            grad_h = pred_grad
    return pred_sr, grad_h


def main():
    wf_path = r"C:/Users/jules/Downloads/pi-ddpm-main/Recon/OMX_LSEC_Actin_525nm.tif"
    gt_path = r"C:/Users/jules/Downloads/pi-ddpm-main/Recon/OMX_LSEC_Actin_525nmRT.tif"

    os.makedirs('./imgs_output/sim_test', exist_ok=True)

    patch_size = 256
    splits = load_sim_stacks(wf_path, gt_path, patch_size=patch_size, stride=patch_size, psf_size=64, psf_sigma_px=6.0)

    # Build model
    img_channels = 1
    p_model, _ = train_model((patch_size, patch_size, 1), (patch_size, patch_size, img_channels), img_channels,
                              noise_est=False)
    # Load weights if available
    # p_model.load_weights('./models_weights/sim/model_10000.h5')

    # Take a small batch from test split
    x = splits['test']['x'][:2]
    fx = x

    pred_sr, grad_h = ddpm_reconstruct(fx, timesteps_test=100, p_model=p_model, smooth_factor=0.015, reg_type='l1')

    # Save montage
    for i in range(pred_sr.shape[0]):
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow((fx[i, ..., 0] + 1) / 2, cmap='gray'); axs[0].set_title('Input (widefield)'); axs[0].axis('off')
        axs[1].imshow((pred_sr[i, ..., 0] + 1) / 2, cmap='gray'); axs[1].set_title('Predicted delta'); axs[1].axis('off')
        axs[2].imshow(((pred_sr[i, ..., 0] + fx[i, ..., 0]) + 1) / 2, cmap='gray'); axs[2].set_title('Reconstruction'); axs[2].axis('off')
        plt.tight_layout()
        plt.savefig(f'./imgs_output/sim_test/recon_{i}.png', dpi=150)
        plt.close()


if __name__ == '__main__':
    main()



