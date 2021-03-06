"""EchoNet-Labs Dataset."""

import pathlib
import os
import collections

import numpy as np
import skimage.draw
import torch.utils.data
import echonet


class Echo(torch.utils.data.Dataset):
    """EchoNet-Labs Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {"train", "val", "test", "external_test"}
        target_type (string or list, optional): Type of target to use.
            A lab name or one of the following:
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
        video_dir (string): directory containing videos within root. Defaults to ``Videos".
        segmentation_dir (string): directory containing segmentations generated by echonet-dynamic. Only used if segmentation_mode
            is set. Defaults to ``segmentation". 
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
        remove_nans (None or boolean, optional): If set, removes entries where target_type is nan, otherwise throws exception. Defaults to None.
        cutoff (None or float, optonal): Cutoff to binarized labels to `label>=cutoff`. Defaults to None.
        filelist (str): File with filenames and labels in root. Defaults to ``FileList.csv". 
        mem_efficient (boolean): Loads videos in more memory-efficient manner. Doesn't support all features.
        files_to_include (list of strings, optional): If set, only loads files in list.
        segmentation_mode (string): If only, returns only the left ventricular segmentation as an image. If video, returns both returns video and
            segmentation. Otherwise returns only video.
        single_repeated (boolean): If True, returns a single frame repeated as a video. 
    """

    def __init__(self, root=None,
                 split="train", target_type="logBNP",
                 video_dir="Videos",
                 segmentation_dir="segmentation",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None,
                 remove_nans=False,
                 cutoff=None, shape=None,
                 filelist="FileList.csv",
                 mem_efficient=True,
                 files_to_include=[],
                 segmentation_mode="",
                 single_repeated=False):

        assert segmentation_mode in ["only", "both", ""]

        if root is None:
            root = echonet.config.DATA_DIR

        self.folder = pathlib.Path(root)
        self.split = split
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.cutoff = cutoff
        self.video_dir = video_dir
        self.shape = shape
        self.segmentation_mode = segmentation_mode
        self.segmentation_dir = segmentation_dir
        self.mem_efficient = mem_efficient and self.clips == 1 and not self.segmentation_mode
        self.single_repeated = single_repeated

        if self.single_repeated:
            assert self.mem_efficient

        self.fnames, self.outcome = [], []

        if split == "external_test":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            with open(self.folder / filelist) as f:
                self.header = f.readline().strip().split(",")
                self.header = [v.strip('"') for v in self.header]

                if "FileName" in self.header:
                    filenameIndex = self.header.index("FileName")
                elif "V1" in self.header:
                    filenameIndex = self.header.index("V1")

                splitIndex = self.header.index("Split")

                i = 0


                for line in f:
                    lineSplit = line.strip().split(',')
                    lineSplit = [v.strip('"') for v in lineSplit]

                    fileName = lineSplit[filenameIndex]
                    if '.avi' not in fileName:
                        fileName = fileName + '.avi'

                    if files_to_include and fileName not in files_to_include:
                        continue

                    fileMode = lineSplit[splitIndex].lower()

                    if remove_nans:
                        try:
                            for t in self.target_type:
                                if (t not in ["Filename", "LargeIndex", "SmallIndex",
                                        "LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"] and
                                        np.isnan(np.float32(lineSplit[self.header.index(t)]))):
                                    raise Exception() 
                        except:
                            continue

                    if split in ["all", fileMode]:
                        i += 1

                    if split in ["all", fileMode] and os.path.exists(self.folder / self.video_dir / fileName):
                        self.fnames.append(fileName)
                        self.outcome.append(lineSplit)

            print("{} files loaded".format(len(self.fnames)))
            print(i)

            if any([t in ["LargeIndex", "SmallIndex", "LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
                    for t in self.target_type]):

                self.frames = collections.defaultdict(list)
                self.trace = collections.defaultdict(_defaultdict_of_lists)

                with open(self.folder / "VolumeTracings.csv") as f:
                    header = f.readline().strip().split(",")
                    assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                    for line in f:
                        filename, x1, y1, x2, y2, frame = line.strip().split(',')
                        x1 = float(x1)
                        y1 = float(y1)
                        x2 = float(x2)
                        y2 = float(y2)
                        frame = int(frame)
                        if frame not in self.trace[filename]:
                            self.frames[filename].append(frame)
                        self.trace[filename][frame].append((x1, y1, x2, y2))
                for filename in self.frames:
                    for frame in self.frames[filename]:
                        self.trace[filename][frame] = np.array(self.trace[filename][frame])

                keep = [len(self.frames[os.path.splitext(f)[0]]) >= 2 for f in self.fnames]
                self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
                self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "external_test":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "clinical_test":
            video = os.path.join(self.folder, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.folder, self.video_dir, self.fnames[index])

        segmentation_path = os.path.join(self.folder, self.segmentation_dir, "{}.npy".format(self.fnames[index]))

        # Load video into np.array

        args = {}
        if self.shape is not None:
            args = {"h": self.shape[0], "w": self.shape[1]}

        if self.mem_efficient:
            assert self.clips == 1
            args["rand_frames"] = self.length
            args["period"] = self.period
            args["single_repeated"] = self.single_repeated

        if self.segmentation_mode == "only":
            video = (np.load(segmentation_path) > 0).astype(np.float32)
            shape = video.shape
            video = np.expand_dims(video, 0)
            video = np.repeat(video, 3, 0)
        else:
            video = echonet.utils.loadvideo(video, **args).astype(np.float32)

        if self.segmentation_mode == "both":
            segmentation = (np.load(segmentation_path) > 0).astype(np.float32) 

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0


        c, f, h, w = video.shape

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(c, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(c, 1, 1, 1)

        if not self.mem_efficient:
            if self.length is None:
                # Take as many frames as possible
                length = f // self.period
            else:
                # Take specified number of frames
                length = self.length

            if self.max_length is not None:
                # Shorten videos to max_length
                length = min(length, self.max_length)

            if f < length * self.period:
                # Pad video with frames filled with zeros if too short
                # 0 represents the mean color (dark grey), since this is after normalization
                video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)

                if self.segmentation_mode == "both":
                    segmentation = np.concatenate((segmentation, np.zeros((length * self.period - f, h, w), segmentation.dtype)), axis=0)
                c, f, h, w = video.shape  # pylint: disable=E0633

            if self.clips == "all":
                # Take all possible clips of desired length
                start = np.arange(f - (length - 1) * self.period)
            else:
                # Take random clips from video
                start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets
        target = []
        for t in self.target_type:
            key = os.path.splitext(self.fnames[index])[0]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)
            else:
                if self.split == "clinical_test" or self.split == "external_test":
                    target.append(np.float32(0))
                else:
                    try:
                        s = self.outcome[index][self.header.index(t)]
                        val = np.float32(s)
                    except ValueError:
                        val = np.float32('nan')

                    if self.cutoff is None:
                        target.append(val)

                    else:
                        target.append((self.cutoff <= val).astype(np.float32))

        if target != []:
            target = tuple(target)
            if self.target_transform is not None:
                target = self.target_transform(target)

            target = np.array(target)

        # Select random clips
        if not self.mem_efficient:

            video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
            if self.clips == 1:
                video = video[0]
            else:
                video = np.stack(video)

            if self.segmentation_mode == "both":
                segmentation = tuple(segmentation[s + self.period * np.arange(length), :, :] for s in start)
                if self.clips == 1:
                    segmentation = segmentation[0]
                else:
                    segmentation = np.stack(segmentation)

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]

            if self.segmentation_mode == "both":
                l, h, w = segmentation.shape
                temp = np.zeros((l, h + 2 * self.pad, w + 2 * self.pad), dtype=segmentation.dtype)
                temp[:, self.pad:-self.pad, self.pad:-self.pad] = segmentation  # pylint: disable=E1130
                segmentation = temp[:, i:(i + h), j:(j + w)]

        if self.segmentation_mode == "both":
            return video, segmentation, target
        else:
            return video, target

    def __len__(self):
        return len(self.fnames)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
