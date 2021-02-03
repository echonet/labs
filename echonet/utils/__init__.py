"""Utility functions for videos, plotting and computing performance metrics."""

import os
import typing

import cv2  # pytype: disable=attribute-error
import matplotlib
import numpy as np
import torch
import tqdm

from . import video
from . import video_explain

def loadvideo(filename: str, h: int = 0, w: int = 0, rand_frames: int = -1, period: int = 1, single_repeated: bool = False) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = w if w > 0 else original_width
    frame_height = h if h > 0 else original_height

    margin_size = int(abs(original_width - original_height) / 2)

    if rand_frames > 0:
        if cv2.CAP_PROP_FRAME_COUNT - (rand_frames - 1) * period > 0:
            s = np.random.choice(cv2.CAP_PROP_FRAME_COUNT - (rand_frames - 1) * period)
        else:
            s = 0

        frame_ids = s + period * np.arange(rand_frames)

    v = np.zeros((rand_frames if rand_frames > 0 else frame_count, frame_width, frame_height, 3), np.uint8)

    i_in_v = 0

    for count in range(frame_count):
        if count == 0 or single_repeated == False:
            ret, frame = capture.read()

        if rand_frames > 0 and count not in frame_ids:
            continue

        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if h > 0:
            assert w > 0
            if margin_size > 0:
                if frame.shape[1] > frame.shape[0]:
                    frame = frame[ :, margin_size : -margin_size, :]
                else:
                    frame = frame[margin_size : -margin_size, :, :]

            frame = cv2.resize(frame, dsize = (h, w), interpolation = cv2.INTER_CUBIC)

        v[i_in_v] = frame
        i_in_v += 1

    capture.release()

    v = v.transpose((3, 0, 1, 2))

    return v


def savevideo(filename: str, array: np.ndarray, fps: typing.Union[float, int] = 1):
    """Saves a video to a file.

    Args:
        filename (str): filename of video
        array (np.ndarray): video of uint8's with shape (channels=3, frames, height, width)
        fps (float or int): frames per second

    Returns:
        None
    """

    c, f, height, width = array.shape

    if c != 3:
        raise ValueError("savevideo expects array of shape (channels=3, frames, height, width), got shape ({})".format(", ".join(map(str, array.shape))))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(f):
        out.write(array[:, i, :, :].transpose((1, 2, 0)))


def get_mean_and_std(dataset: torch.utils.data.Dataset,
                     samples: int = 128,
                     batch_size: int = 8,
                     num_workers: int = 4,
                     all_features: bool = False,
                     segments: bool = False):
    """Computes mean and std from samples from a Pytorch dataset.

    Args:
        dataset (torch.utils.data.Dataset): A Pytorch dataset.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int or None, optional): Number of samples to take from dataset. If ``None'', mean and
            standard deviation are computed over all elements.
            Defaults to 128.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 8.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.

    Returns:
       A tuple of the mean and standard deviation. Both are represented as np.array's of dimension (channels,).
    """

    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    n = 0  # number of elements taken (should be equal to samples by end of for loop)
    s1 = 0.  # sum of elements along channels (ends up as np.array of dimension (channels,))
    s2 = 0.  # sum of squares of elements along channels (ends up as np.array of dimension (channels,))

    st = None
    nt = None

    if segments:

        for (x, t) in dataloader:
        #for (x, t) in tqdm.tqdm(dataloader):

            if isinstance(t, list):
                t = np.array([targ.numpy() for targ in t])

            else:
                t = t.numpy()
                t = np.expand_dims(t, 1)

            st_batch = np.nan_to_num(t).sum(axis=0)
            nt_batch = (1-np.isnan(t)).sum(axis=0)

            st = st_batch if st is None else st + st_batch
            nt = nt_batch if nt is None else nt + nt_batch

        target_mean = st / nt

        mean = 0.
        std = 1.
        target_mean = target_mean.astype(np.float32)

    else:

        for (x, t) in tqdm.tqdm(dataloader):
            x = x.transpose(0, 1).contiguous().view(3, -1)
            n += x.shape[1]
            s1 += torch.sum(x, dim=1).numpy()
            s2 += torch.sum(x ** 2, dim=1).numpy()

            if isinstance(t, list):
                t = np.array([targ.numpy() for targ in t])

            else:
                t = t.numpy()
                t = np.expand_dims(t, 1)

            st_batch = np.nan_to_num(t).sum(axis=0)
            nt_batch = (1-np.isnan(t)).sum(axis=0)

            st = st_batch if st is None else st + st_batch
            nt = nt_batch if nt is None else nt + nt_batch

        mean = s1 / n  # type: np.ndarray
        std = np.sqrt(s2 / n - mean ** 2)  # type: np.ndarray
        target_mean = st / nt

        mean = mean.astype(np.float32)
        std = std.astype(np.float32)
        target_mean = target_mean.astype(np.float32)

    if all_features:
        return mean, std, target_mean

    return mean, std


def bootstrap(a, b, func, samples=10000):
    """Computes a bootstrapped confidence intervals for ``func(a, b)''.

    Args:
        a (array_like): first argument to `func`.
        b (array_like): second argument to `func`.
        func (callable): Function to compute confidence intervals for.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int, optional): Number of samples to compute.
            Defaults to 10000.

    Returns:
       A tuple of (`func(a, b)`, estimated 5-th percentile, estimated 95-th percentile).
    """
    a = np.array(a)
    b = np.array(b)

    bootstraps = []
    for _ in range(samples):
        ind = np.random.choice(len(a), len(a))
        bootstraps.append(func(a[ind], b[ind]))
    bootstraps = sorted(bootstraps)

    return func(a, b), bootstraps[round(0.05 * len(bootstraps))], bootstraps[round(0.95 * len(bootstraps))]


def latexify():
    """Sets matplotlib params to appear more like LaTeX.

    Based on https://nipunbatra.github.io/blog/2014/latexify.html
    """
    params = {'backend': 'pdf',
              'axes.titlesize': 8,
              'axes.labelsize': 8,
              'font.size': 8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'font.family': 'DejaVu Serif',
              'font.serif': 'Computer Modern',
              }
    matplotlib.rcParams.update(params)


def dice_similarity_coefficient(inter, union):
    """Computes the dice similarity coefficient.

    Args:
        inter (iterable): iterable of the intersections
        union (iterable): iterable of the unions
    """
    return 2 * sum(inter) / (sum(union) + sum(inter))


__all__ = ["video", "video_explain", "segmentation", "loadvideo", "savevideo", "get_mean_and_std", "bootstrap", "latexify", "dice_similarity_coefficient"]
