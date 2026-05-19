import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def create_heatmap(original, adversarial):

    perturbation = np.abs(adversarial - original)

    heatmap = perturbation.mean(axis=2)

    return heatmap


def compute_metrics(original, adversarial):

    l2 = np.linalg.norm(adversarial - original)

    linf = np.max(np.abs(adversarial - original))

    similarity = ssim(
        original,
        adversarial,
        channel_axis=2,
        data_range=1.0,
    )

    psnr_score = psnr(
        original,
        adversarial,
        data_range=1.0,
    )

    return {
        "L2": l2,
        "Linf": linf,
        "SSIM": similarity,
        "PSNR": psnr_score,
    }
