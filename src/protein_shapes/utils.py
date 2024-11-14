import gzip
from pathlib import Path
import pickle

import numpy as np
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def load_embeddings(embed_dir: Path):
    pdbs, all_embeds = [], []
    for fp in embed_dir.iterdir():
        if fp.suffix == ".gz":  # * ProtDomainSegmentor embeddings
            with gzip.open(fp, "rb") as f:
                embed = np.load(f)
            pdbs.append(fp.stem)
            all_embeds.append(embed)
        elif fp.suffix == ".npy":  # * ProteinMPNN embeddings (3, 1, L, D)
            embed = np.load(fp)
            pdbs.append(fp.stem)
            all_embeds.append(embed[:, 0].mean(-2))
        elif fp.suffix == ".pkl":  # * ESM3 embeddings
            with open(fp, "rb") as f:
                embed_dict = pickle.load(f)
            for k, v in embed_dict.items():
                pdbs.append(k)
                all_embeds.append(v[0][0].mean(-2).numpy())
    return pdbs, np.array(all_embeds)


def fpd(
    samp_embeds: np.ndarray,
    gt_embeds: np.ndarray = None,
    gt_mu: np.ndarray = None,
    gt_sigma: np.ndarray = None,
):
    """Entrypoint for computing FPD

    Parameters
    ----------
    samp_embeds
        Array of embeddings of sampled structures, shape (N, D) for N samples with D dimensions each
        If using per-residue embeddings, can be mean-pooled along the sequence dimension to achieve (N, D)
            or treat each residue as a separate embedding, merged into axis=0
    gt_embeds
        Array of embeddings of reference structures
    """
    samp_mu = np.mean(samp_embeds, axis=0)
    samp_sigma = np.cov(samp_embeds, rowvar=False)

    if gt_embeds is not None:
        gt_mu = np.mean(gt_embeds, axis=0)
        gt_sigma = np.cov(gt_embeds, rowvar=False)

    return calculate_frechet_distance(samp_mu, samp_sigma, gt_mu, gt_sigma)
