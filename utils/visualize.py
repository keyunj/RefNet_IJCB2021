import torch
import numpy as np
import matplotlib.pyplot as plt
from pylab import subplots_adjust


def tensor_to_numpy(tensor):
    if torch.is_tensor(tensor):
        try:
            tensor = tensor.squeeze(0).numpy()
        except:
            tensor = tensor.squeeze(0).cpu().numpy()
    return tensor


def matplotlib_imshow(img, alpha=1.0, one_channel=False):
    npimg = tensor_to_numpy(img)
    npimg = (npimg * 255 * alpha).astype(np.uint8)

    if one_channel:
        plt.imshow(npimg, cmap="gray")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def draw_orientation(ori, mask, factor=8, stride=32):
    ori = tensor_to_numpy(ori)
    mask = tensor_to_numpy(mask)

    ori = ori * np.pi / 180
    for ii in range(stride // factor // 2, ori.shape[0], stride // factor):
        for jj in range(stride // factor // 2, ori.shape[1], stride // factor):
            if mask[ii, jj] == 0:
                continue
            x, y, o, r = jj, ii, ori[ii, jj], stride * 0.8
            plt.plot(
                [x * factor - 0.5 * r * np.cos(o), x * factor + 0.5 * r * np.cos(o)],
                [y * factor - 0.5 * r * np.sin(o), y * factor + 0.5 * r * np.sin(o)],
                "-",
                color="red",
                linewidth=1.5,
            )


def draw_pose(trans, theta, length=200, color="green"):
    trans = tensor_to_numpy(trans)
    theta = tensor_to_numpy(theta)

    theta = theta * np.pi / 180
    start = trans
    end = (start[0] - length * np.sin(theta), start[1] - length * np.cos(theta))

    plt.plot(start[0], start[1], "o", color=color)
    plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], width=2, fc=color, ec=color)


def draw_on_image(
    img, ori_pred, seg_pred, ori_true, seg_true, anchor, angdiff, segiou, name, type_pred=None, type_true=None,
):
    segiou = tensor_to_numpy(segiou)
    angdiff = tensor_to_numpy(angdiff)
    type_pred = tensor_to_numpy(type_pred[0]) if type_pred is not None else None
    type_true = tensor_to_numpy(type_true[0]) if type_true is not None else None

    fig = plt.figure(figsize=(8, 6.3))
    suptitle = (
        f"{name[0]}, type: {type_true:d}, segiou: {segiou:.3f} angdiff: {angdiff:.3f}"
        if type_true is not None
        else f"{name[0]}, segiou: {segiou:.3f} angdiff: {angdiff:.3f}"
    )
    plt.suptitle(suptitle)
    subplots_adjust(left=0.0, bottom=0.0, top=0.9, right=1, hspace=0, wspace=0)

    # row 1
    # original image
    ax = fig.add_subplot(2, 3, 1, xticks=[], yticks=[])
    matplotlib_imshow(img[0], one_channel=True)
    ax.set_title("image")
    # orientaton prediction
    ax = fig.add_subplot(2, 3, 2, xticks=[], yticks=[])
    matplotlib_imshow(img[0], alpha=0.8, one_channel=True)
    draw_orientation(ori_pred[0], seg_true[0])
    ax.axis("off")
    ax.set_title("pred ori")
    # segmentation prediction
    ax = fig.add_subplot(2, 3, 3, xticks=[], yticks=[])
    matplotlib_imshow(seg_pred[0], one_channel=True)
    ax.set_title("pred seg")

    # row 2
    # anchor
    ax = fig.add_subplot(2, 3, 4, xticks=[], yticks=[])
    matplotlib_imshow(img[0], alpha=0.8, one_channel=True)
    draw_orientation(anchor[0], seg_true[0])
    ax.axis("off")
    axtitle = f"anchor, type: {np.round(type_pred, 2)}" if type_pred is not None else "anchor"
    ax.set_title(axtitle)
    # orientation true
    ax = fig.add_subplot(2, 3, 5, xticks=[], yticks=[])
    matplotlib_imshow(img[0], alpha=0.8, one_channel=True)
    draw_orientation(ori_true[0], seg_true[0])
    ax.axis("off")
    ax.set_title("true ori")
    # segmentation true
    ax = fig.add_subplot(2, 3, 6, xticks=[], yticks=[])
    matplotlib_imshow(seg_true[0], one_channel=True)
    ax.set_title("true seg")

    return fig
