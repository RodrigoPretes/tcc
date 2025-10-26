import numpy as np
import cv2
import math
import torch

# ============================================================
# --- PATCH-WISE SEGMENTATION HELPERS ---
# ============================================================

def segmentar_por_patches(image, n_nodes=1000, segmentador="DISF", disf_module=None, slic_fn=None, grid=(2, 2)):
    """
    Segmenta uma imagem em múltiplos patches para evitar sobrecarga de memória.

    Args:
        image (np.ndarray): Imagem RGB.
        n_nodes (int): Número total desejado de superpixels na imagem.
        segmentador (str): 'DISF' ou 'SLIC'.
        disf_module: módulo DISF compilado (necessário se segmentador == 'DISF').
        slic_fn: função SLIC (ex: skimage.segmentation.slic) se segmentador == 'SLIC'.
        grid (tuple): (n_linhas, n_colunas) - número de divisões da imagem.

    Returns:
        np.ndarray: Mapa de rótulos combinado (H, W) com índices únicos.
    """

    if segmentador not in ["DISF", "SLIC"]:
        raise ValueError("segmentador deve ser 'DISF' ou 'SLIC'.")

    if segmentador == "DISF" and disf_module is None:
        raise ValueError("disf_module é obrigatório para segmentador='DISF'.")

    H, W = image.shape[:2]
    n_linhas, n_colunas = grid
    patch_h = math.ceil(H / n_linhas)
    patch_w = math.ceil(W / n_colunas)

    total_patches = n_linhas * n_colunas
    n_nodes_patch = max(50, n_nodes // total_patches)

    label_img_global = np.zeros((H, W), dtype=np.int32)
    offset = 0

    for i in range(n_linhas):
        for j in range(n_colunas):
            y0, y1 = i * patch_h, min((i + 1) * patch_h, H)
            x0, x1 = j * patch_w, min((j + 1) * patch_w, W)
            patch = image[y0:y1, x0:x1, :]

            if segmentador == "SLIC":
                labels_patch = slic_fn(patch, n_segments=n_nodes_patch, slic_zero=True, channel_axis=2)
            else:  # DISF
                patch_int32 = patch.astype(np.int32)
                superpixel_patch, _ = disf_module.DISF_Superpixels(
                    patch_int32,
                    max(n_nodes_patch * 2, 200),
                    n_nodes_patch
                )
                labels_patch = np.array(superpixel_patch, dtype=np.int32)

            labels_patch += offset  # garante IDs únicos globais
            label_img_global[y0:y1, x0:x1] = labels_patch
            offset = labels_patch.max() + 1

    return label_img_global


def reconstruir_segmentacao(label_img, image_shape):
    """
    Redimensiona a segmentação reconstruída para combinar com a imagem original.

    Args:
        label_img (np.ndarray): Mapa de rótulos (H, W).
        image_shape (tuple): shape (H, W, C) da imagem original.

    Returns:
        np.ndarray: Mapa de rótulos ajustado ao tamanho original.
    """
    return cv2.resize(label_img.astype(np.int32), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)


def construir_arestas_patchwise(label_img):
    """
    Constrói arestas locais (rala) para um mapa de rótulos segmentado em patches.

    Args:
        label_img (np.ndarray): Mapa de rótulos.

    Returns:
        np.ndarray: lista de pares de nós (E, 2)
    """
    from skimage.segmentation import find_boundaries

    boundaries = find_boundaries(label_img, mode="thick")
    H, W = label_img.shape
    neigh = set()

    ys, xs = np.nonzero(boundaries)
    for y, x in zip(ys, xs):
        a = label_img[y, x]
        if y + 1 < H:
            b = label_img[y + 1, x]
            if a != b:
                u, v = (a, b) if a < b else (b, a)
                neigh.add((u, v))
        if x + 1 < W:
            b = label_img[y, x + 1]
            if a != b:
                u, v = (a, b) if a < b else (b, a)
                neigh.add((u, v))

    edges = np.array([(u, v) for u, v in neigh] + [(v, u) for u, v in neigh], dtype=np.int64)
    return edges
