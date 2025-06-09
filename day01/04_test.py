import numpy as np
import rasterio
import matplotlib.pyplot as plt

def stretch_band(band, low=2, high=98):
    """直方图拉伸增强对比度，避免整体过亮或过暗"""
    band = band.astype(float)
    p_low, p_high = np.percentile(band, (low, high))
    stretched = np.clip(band, p_low, p_high)
    return ((stretched - p_low) / (p_high - p_low)) * 255

def shuchu():
    # 路径写死，按你要求自动读取该文件
    tif_file = r"D:\huaqing_it\2019_1101_nofire_B2348_B12_10m_roi.tif"

    with rasterio.open(tif_file) as src:
        bands = src.read()

    # 波段定义（假设顺序为 B02, B03, B04, B08, B12）
    blue = bands[0].astype(float)
    green = bands[1].astype(float)
    red = bands[2].astype(float)
    nir = bands[3].astype(float)
    swir = bands[4].astype(float)

    # 亮度自然的 RGB 图（用于人眼参考）
    r_img = stretch_band(red)
    g_img = stretch_band(green)
    b_img = stretch_band(blue)
    rgb_image = np.dstack((r_img, g_img, b_img)).astype(np.uint8)

    # 🔥 红外火点分析图（火点呈现黑色，其它为红色）
    # 原理：SWIR 强 → 非火点；SWIR 弱 → 潜在火点（比如烧焦地）
    swir_stretched = stretch_band(swir)

    # 反转处理：火点呈黑色，背景红色
    fire_map = 255 - swir_stretched
    fire_visual = np.zeros((*fire_map.shape, 3), dtype=np.uint8)
    fire_visual[:, :, 0] = fire_map  # 仅红通道有值，其它通道为0

    # 显示
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image)
    plt.title("RGB Image (Natural Brightness)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(fire_visual)
    plt.title("Fire Analysis (Red = Normal, Black = Fire)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
