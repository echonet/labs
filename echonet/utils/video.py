"""Functions for training and running lab prediction."""

import math
import os
import time

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm

import echonet

def run(num_epochs=45,
        modelname="r2plus1d_18",
        tasks="logBNP",
        frames=32,
        period=2,
        pretrained=True,
        output=None,
        device=None,
        n_train_patients=None,
        num_workers=5,
        batch_size=20,
        seed=0,
        lr_step_period=15,
        run_test=True,
        save_all_models=False,
        weight_by_std=True,
        remove_nans=False,
        weight_decay=1e-4,
        noise=None,
        classification=False,
        single_frame=False,
        video_dir="Videos",
        filelist="FileList.csv",
        valid_filelist=None,
        lr=None,
        root=None,
        score_by_auc=True,
        side_length=112,
        low_res=True,
        choose_by_score=False,
        test=False,
        initialize_from=None,
        embeddings=False,
        segmentation_mode="",
        segmentation_params={"rect": True, "mask": False, "mitral": False, "expand": 10},
        single_repeated=False):
    """Trains/tests lab prediction model.

    Args:
        num_epochs (int, optional): Number of epochs during training
            Defaults to 45.
        modelname (str, optional): Name of model. One of ``mc3_18'',
            ``r2plus1d_18'', or ``r3d_18''
            (options are torchvision.models.video.<modelname>)
            Defaults to ``r2plus1d_18''.
        tasks (str, optional): Name of task to predict. Options are the headers
            of FileList.csv.
            Defaults to ``logBNP''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to True.
        output (str or None, optional): Name of directory to place outputs
            Defaults to None (replaced by output/video/<modelname>_<pretrained/random>/).
        device (str or None, optional): Name of device to run on. See
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            for options. If ``None'', defaults to ``cuda'' if available, and ``cpu'' otherwise.
            Defaults to ``None''.
        n_train_patients (str or None, optional): Number of training patients. Used to ablations
            on number of training patients. If ``None'', all patients used.
            Defaults to ``None''.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 5.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 20.
        seed (int, optional): Seed for random number generator.
            Defaults to 0.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            If ``None'', learning rate is not decayed.
            Defaults to 15.
        run_test (bool, optional): Whether or not to run test loop.
            Defaults to False.
        save_all_models (bool, optional): Whether to save models for each epoch.
            Defaults to False. 
        weight_by_std (bool, optional): When doing multi-headed regression, whether
            to weight loss by standard deviation of each class. Defaults to False. 
        remove_nans (bool, optional): Whether to silently remove examples with nan labels.
            Defaults to False.
        weight_decay (float, optional): Weight decay coefficient. Defaults to 1e-4.
        noise (float or None, optional): Ammount of noise to add to input. Defaults to None.
        classification (bool, optional): Whether to cutoff labels and do classification, or to regress.
            Defaults to False.
        single_frame (bool, optional): Replaces video with repeated single frame. Defaults to False.
        root (str or None): root to dataset location. 
        video_dir (str, optional): Path to video directory within root. Defualts to "Videos".
        filelist (str, optional): Path to filelist with labels within root. Defualts to "FileList.csv".
        valid_filelist (str, optional): Path to filelist with labels within root, if differnt from filelist.
            Defualts to None.
        lr (float or None, optional): Learning rate. If None, defaults to 1e-3.
        score_by_auc (bool, optional): Score by AUC, otherwise by R2. Defaults to True.
        side_length (int, optional): Length of each side of video. Defaults to 112.
        low_res (bool, optional): Whether to use low_res video files (downsampled to 112*112), Defaults to True.
        choose_by_score (bool, optional): Choose best model based on score, rather than loss. Defaults to True.
        test (bool, optional): Sets split to test, otherwise uses train and val. Default False.
        initialize_from (str or None, optional): Checkpoint to initialize from. Default None.
        embeddings (bool, optional): Whether to save final layer embeddings. Defaults to False.
        segmentation_mode (str, optional): One of {"", "only", "both"}. 
            If "", trains normally.
            If "only", trains on ventricular segmentation.
            If "both", trains on combination of segmentation and video.
            Defaults to "".
        segmentation_params (dict, optional): Options for segmentation.
            Defaults to {"rect": True, "mask": False, "mitral": False, "expand": 10}.
        single_repeated (bool, optional): If True, returns a single frame repeated as a video. Defaults to False.
    """
    if valid_filelist is None:
        valid_filelist = filelist
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    stds = defaultdict(lambda: 1.)
    stds.update(EF=12.31, RAP=4.67, RVSP=14.39, TRVelocity=22.71,
        BNP=6964, A1c=2.08, Troponin=1.84, logTroponin=1.63, logBNP=1.83,
        ALT=104.71, AST=169.97, AlkPhos=70.34, BUN=12.54, Chloride=3.09,
        Cr=0.93, Creatinine=0.93, Hemoglobin=1.96, pH=0.05, Platelet=74.51, Sodium=2.72,
        WBC=4.91, logALT=0.59, logAST=0.64, CRP=21.53, logCRP=1.46)

    if not isinstance(tasks, list):
        tasks = [tasks]
    tasks = sorted(tasks)
    stds = np.array([stds[task] for task in tasks]) if weight_by_std else np.array([1.])

    classification_cutoffs = {'RVSP': 25, 'A1c': 6.5, 'BNP': 300, 'Troponin': .0001, 'logTroponin': np.log(.0001), 'random': 0.1, 'logBNP': np.log(300),
                        'RAP': 5, 'TRVelocity': 20, 'Cr': 1.5, 'Creatinine': 1.5, 'Sodium': 130, 'Potassium': 5, 'BUN': 30, 'WBC': 12,
                        'Hemoglobin': 10, 'Platelet': 150, 'AlkPhos': 200, 'Chloride': 105, 'pH': 7.30, 'AST': 200,
                        'ALT': 200, 'logALT': np.log(200), 'logAST': np.log(200), 'CRP': 2, 'logCRP': np.log(2)}
    if classification:
        assert len(tasks) == 1
        cutoff = classification_cutoffs[tasks[0]]
        binary = True
    else:
        cutoff = [classification_cutoffs[t] for t in tasks] if score_by_auc else None
        binary = False

    pad = 12
    if low_res and side_length==112:
        shape = None
    else:
        shape = (side_length, side_length)
        if low_res:
            pad = None

    if segmentation_mode != "both":
        segmentation_params = {}

    segmentation_params = defaultdict(lambda: False, segmentation_params)

    # Set default output director
    if output is None:
        if np.all([len(tasks) == 1, (not single_frame and modelname == "r2plus1d_18" and frames == 32) or (single_frame and modelname == "vgg16" and frames==1),
                   weight_decay == 1e-4, pretrained, remove_nans]):
            output = "{}{}{}_{}_{}_{}_{}".format('rev' if segmentation_params['reverse'] else '', 'mitral' if segmentation_params['mitral'] else ('rect' if segmentation_params['rect'] else segmentation_mode), tasks[0], period, lr, "low" if (low_res and side_length==112) else ("low{}".format(side_length) if low_res else side_length), filelist.split(".")[0])
            if single_repeated or single_frame:
                output += "_single_frame"
            if classification:
                output += "_classification"
            if n_train_patients:
                output += "_ablate" + str(n_train_patients)

            if initialize_from:
                output += "_from{}".format(initialize_from.replace("/", ""))
            elif not pretrained:
                output += "_nopretrain"
            if noise is not None:
                output += "_noise{}".format(noise)

            if choose_by_score:
                output = os.path.join("AUC_optimized", output) 
        else:
            # Legacy naming
            task_str = "_" + "_".join(sorted(tasks))
            output = "{}{}{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                "single_" if single_frame else "", "classification_" if classification else "",
                modelname, frames, period, weight_decay, noise, "pretrained" if pretrained else "random", task_str, "removenan" if remove_nans else "masknan", lr)
            if root is not None:
                output += "_{}".format(root.replace("/", "_"))
            if video_dir != "Videos":
                output += "_{}".format(video_dir.replace("/", "_"))
        output = os.path.join("/oak/stanford/groups/jamesz/jwhughes/output/video", output)
    os.makedirs(output, exist_ok=True)
    print(output)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    if single_frame:
        assert frames == 1
        model = torchvision.models.__dict__[modelname](pretrained=pretrained)

    else:
        model = torchvision.models.video.__dict__[modelname](pretrained=pretrained)


    # Compute mean and std
    ds = echonet.datasets.Echo(split="train", target_type=tasks, cutoff=cutoff if binary else None, video_dir=video_dir, shape=shape, root=root, filelist=filelist, segmentation_mode="only" if segmentation_mode == "only" else "")
    mean, std, target_mean = echonet.utils.get_mean_and_std(ds, all_features=True, segments=(segmentation_mode=="only"))

    if modelname in ['vgg16']:
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(tasks))
        last_layer = model.classifier[-1]
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, len(tasks))
        last_layer = model.fc

    if binary:
        last_layer.bias.data = torch.from_numpy(np.log(target_mean/(1-target_mean)))
    else:
        last_layer.bias.data = torch.from_numpy(target_mean)

    if initialize_from:
        last_layer.bias.data = last_layer.bias.data[0]

    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              "remove_nans": remove_nans,
              "cutoff": cutoff if binary else None,
              "video_dir": video_dir,
              "shape": shape,
              "root": root,
              "filelist": filelist,
              "segmentation_mode": segmentation_mode,
              "single_repeated": single_repeated,
              "noise": noise
              }

    valid_kwargs = kwargs.copy()
    valid_kwargs["filelist"] = valid_filelist

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=(1e-3 if weight_by_std else 1e-4) if lr is None else lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Set up datasets and dataloaders
    train_dataset = echonet.datasets.Echo(split="train", **kwargs, pad=pad)
    if n_train_patients is not None and len(train_dataset) > n_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(train_dataset), n_train_patients, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        echonet.datasets.Echo(split="val", **valid_kwargs), batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    if segmentation_mode == "both" and np.any([segmentation_params['expand'], segmentation_params['mitral'], segmentation_params['rect']]):
        if not segmentation_params["expand"]:
            segmentation_params["expand"] = 0
        expander = SegmentationExpander(segmentation_params["expand"], batch_size, device)
    else:
        expander = None

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        bestScore = -float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            if choose_by_score:
                bestScore = checkpoint["best_score"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            if initialize_from:
                try: 
                    checkpoint = torch.load(os.path.join(initialize_from, "best.pt"))
                    model.load_state_dict(checkpoint['state_dict'])
                except FileNotFoundError:
                    f.write("Starting run from scratch\n")
            else:
                f.write("Starting run from scratch\n")

        if embeddings:
            _, yhat, y = echonet.utils.video.run_epoch(model, dataloaders['test' if test else 'val'], False, optim, device, std=stds, binary=binary, single_frame=single_frame, embeddings=True, expander=expander, segmentation_params=segmentation_params)
            np.save(os.path.join(output, f"emb_{'test' if test else 'val'}.npy"), yhat)
            np.save(os.path.join(output, f"yemb_{'test' if test else 'val'}.npy"), y)
            return

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_max_memory_allocated(i)
                    torch.cuda.reset_max_memory_cached(i)
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloaders[phase], phase == "train", optim, device, std=stds, binary=binary, single_frame=single_frame, expander=expander, segmentation_params=segmentation_params)

                if binary:
                    score = sklearn.metrics.roc_auc_score(y.flatten(), yhat.flatten())

                elif score_by_auc:
                    if yhat.ndim == 1:
                        score = sklearn.metrics.roc_auc_score(y[~np.isnan(y)].flatten() >= cutoff[0], yhat[~np.isnan(y)].flatten())
                        r2 = sklearn.metrics.r2_score(y[~np.isnan(y)], yhat[~np.isnan(y)])
                    else:
                        score = [sklearn.metrics.roc_auc_score(y[:,i][~np.isnan(y[:,i])] >= cutoff[i], yhat[:,i][~np.isnan(y[:,i])]) for i in range(y.shape[1])]                
                        r2 = [sklearn.metrics.r2_score(y[:,i][~np.isnan(y[:,i])], yhat[:,i][~np.isnan(y[:,i])]) for i in range(y.shape[1])]

                    score = np.mean(score)
                    #score = str("AUC:{}, R2:{}".format(score, r2))
                else:
                    if yhat.ndim == 1:
                        score = sklearn.metrics.r2_score(y[~np.isnan(y)], yhat[~np.isnan(y)])
                    else:
                        score = [sklearn.metrics.r2_score(y[:,i][~np.isnan(y[:,i])], yhat[:,i][~np.isnan(y[:,i])]) for i in range(y.shape[1])]

                f.write("{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                 phase,
                                                                 loss,
                                                                 score,
                                                                 time.time() - start_time,
                                                                 y.size,
                                                                 sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                 sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count())),
                                                                 batch_size,
                                                                 np.sum(np.isnan(y))))

                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'best_score': bestScore,
                'loss': loss,
                'score': score,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }

            if choose_by_score and bestScore < score:
                bestScore = score
                save['best_score'] = bestScore
                torch.save(save, os.path.join(output, "best.pt"))

            elif not choose_by_score and loss < bestLoss:
                bestLoss = loss
                save['best_loss'] = bestLoss
                torch.save(save, os.path.join(output, "best.pt"))

            if save_all_models:
                torch.save(save, os.path.join(output, "checkpoint_{:02d}.pt".format(epoch)))

            torch.save(save, os.path.join(output, "checkpoint.pt"))

        # Load best weights
        checkpoint = torch.load(os.path.join(output, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
        f.flush()

        if run_test:
            splits = ["test"] if test else ["train", "val"]
            for j, split in enumerate(splits):
                # Performance without test-time augmentation
                dataloader = torch.utils.data.DataLoader(
                    echonet.datasets.Echo(split=split, **(kwargs if split=='train' else valid_kwargs)),
                    batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device, std=stds, binary=binary, single_frame=single_frame, expander=expander, segmentation_params=segmentation_params, save_first_path=None if j else output)
                f.write("train list: {}\n".format(filelist))
                f.write("valid list: {}\n".format(valid_filelist))

                for i, task in enumerate(tasks):
                    mask = ~np.isnan(y[:, i])
                    curr_y = y[:, i][mask]
                    curr_yhat = yhat[:, i][mask]
                    if binary:
                        f.write("{} (one clip) {} AUC:  {:.3f} ({:.3f} - {:.3f})\n".format(split, task, *echonet.utils.bootstrap(curr_y, curr_yhat, sklearn.metrics.roc_auc_score)))
                        f.write("{} (one clip) {} AP:   {:.3f} ({:.3f} - {:.3f})\n".format(split, task, *echonet.utils.bootstrap(curr_y, curr_yhat, sklearn.metrics.average_precision_score)))
                        f.write("{} (one clip) {} revAP:{:.3f} ({:.3f} - {:.3f})\n".format(split, task, *echonet.utils.bootstrap(1-curr_y, 1-curr_yhat, sklearn.metrics.average_precision_score)))

                    else:     
                        f.write("{} (one clip) {} R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, task, *echonet.utils.bootstrap(curr_y, curr_yhat, sklearn.metrics.r2_score)))
                        f.write("{} (one clip) {} MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, task, *echonet.utils.bootstrap(curr_y, curr_yhat, sklearn.metrics.mean_absolute_error)))
                        f.write("{} (one clip) {} RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, task, *tuple(map(math.sqrt, echonet.utils.bootstrap(curr_y, curr_yhat, sklearn.metrics.mean_squared_error)))))
                        if score_by_auc:
                            f.write("{} (one clip) {} AUC:  {:.2f} ({:.2f} - {:.2f})\n".format(split, task, *echonet.utils.bootstrap(curr_y >= cutoff[i], curr_yhat, sklearn.metrics.roc_auc_score)))
                    f.flush()

                np.save(os.path.join(output, f"yhat_{split}.npy"), yhat)
                np.save(os.path.join(output, f"y_{split}.npy"), y)

                if split == "train":
                    continue

                # Performance with test-time augmentation
                ds = echonet.datasets.Echo(split=split, **valid_kwargs, clips="all")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device, save_all=True, block_size=20, std=stds, binary=binary, single_frame=single_frame, expander=expander, segmentation_params=segmentation_params)
                for i, task in enumerate(tasks):
                    mask = ~np.isnan(y[:, i])
                    curr_y = y[:, i][mask]
                    curr_yhat = [yh[:, i] for j, yh in enumerate(yhat) if mask[j]]
                    f.write("{} (all clips) {} R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, task, *echonet.utils.bootstrap(curr_y, np.array(list(map(lambda x: x.mean(), curr_yhat))), sklearn.metrics.r2_score)))
                    f.write("{} (all clips) {} MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, task, *echonet.utils.bootstrap(curr_y, np.array(list(map(lambda x: x.mean(), curr_yhat))), sklearn.metrics.mean_absolute_error)))
                    f.write("{} (all clips) {} RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, task, *tuple(map(math.sqrt, echonet.utils.bootstrap(curr_y, np.array(list(map(lambda x: x.mean(), curr_yhat))), sklearn.metrics.mean_squared_error)))))
                    if score_by_auc:
                        f.write("{} (all clips) {} AUC:  {:.2f} ({:.2f} - {:.2f})\n".format(split, task, *echonet.utils.bootstrap(curr_y >= cutoff[i], np.array(list(map(lambda x: x.mean(), curr_yhat))), sklearn.metrics.roc_auc_score)))
                    f.flush()

                # Write full performance to file
                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    for (filename, pred) in zip(ds.fnames, yhat):
                        for (i, p) in enumerate(pred):
                            out_str = "{},{}"
                            args=[filename, i]
                            for task in p:
                                out_str = out_str + ",{:.4f}"
                                args.append(task)
                            out_str = out_str + "\n"
                            g.write(out_str.format(*args))

class SegmentationExpander():
    """Helper class for masking ablations. 
    """
    def __init__(self, radius, batch, device):
        conv = np.zeros((1, 1, 2*radius + 1, 2*radius + 1)).astype(np.float32)
        for i in range(len(conv[0,0])):
            for j in range(len(conv[0,0,0])):
                if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                    conv[0, :, i, j] = 1
        self.conv = torch.from_numpy(conv).to(device)
        self.radius = radius
        self.device = device

    def expand(self, mask):
        # Expands mask by self.radius
        shape = mask.shape
        mask = mask.reshape(-1, 1, *shape[2:])
        mask = torch.nn.functional.conv2d(mask, self.conv, padding=self.radius) > 0
        return mask.reshape(shape)

    def expand_rectangle(self, mask):
        # Finds rectangle around mask and then expands it
        # Union of masks for all frames of each video
        comp = (mask > 0).sum(1) > 0
        # Outer product of binary mask for h and binary mask for w
        comp_rect = torch.bmm((comp.sum(1).unsqueeze(2) > 0).float(),
                              (comp.sum(2).unsqueeze(1) > 0).float())
        mask = comp_rect.unsqueeze(1).expand(mask.shape[0], 32, 112, 112).transpose(3,2)
        mask = self.expand(mask)
        return mask

    def mitral_cover(self, mask):
        # Finds mask approximately covering mitral valve
        sums = mask.sum(3).reshape(-1, 112).cpu().numpy()
        nonz = [np.nonzero(s)[0] for s in sums]
        upper = [int(self.min(n) / 3 + 2 * self.max(n) / 3) for n in nonz]
        masks = [np.zeros((112, 112)) for _ in upper]
        for i in range(len(upper)):
            masks[i][upper[i]:] = 1
        masks = np.array(masks).reshape(mask.shape)
        masks = torch.logical_and(torch.from_numpy(masks).to(self.device), self.expand(mask))
        return masks

    def min(self, n):
        try:
            return min(n)
        except:
            return 111

    def max(self, n):
        try:
            return max(n)
        except:
            return 111

def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None, std=[1.], binary=False, single_frame=False, embeddings=False, expander=None, segmentation_params={}, save_first_path=None):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
        save_all (bool, optional): If True, return predictions for all
            test-time augmentations separately. If False, return only
            the mean prediction.
            Defaults to False.
        block_size (int or None, optional): Maximum number of augmentations
            to run on at the same time. Use to limit the amount of memory
            used. If None, always run on all augmentations simultaneously.
            Default is None.
    """
    both_segmentation = np.any(list(segmentation_params.values()))

    model.train(train)

    total = 0  # total training loss
    n = 0      # number of videos processed
    s1 = 0     # sum of ground truth
    s2 = 0     # Sum of ground truth squared

    yhat = []
    y = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for i, data in enumerate(dataloader):
                if both_segmentation:
                    X, segmentation_map, outcome = data
                    segmentation_map = segmentation_map.to(device)
                else:
                    X, outcome = data
                    segmentation_map = None
                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)
                mask = ~torch.isnan(outcome)

                if both_segmentation and expander and np.any([segmentation_params['expand'], segmentation_params['mitral'], segmentation_params['rect']]):
                    if segmentation_params['mask']:
                        segmentation_map = expander.expand(segmentation_map)
                        if segmentation_params['reverse']:
                            segmentation_map = ~segmentation_map
                        X = X * segmentation_map.unsqueeze(1)
                    elif segmentation_params['rect']:
                        segmentation_map = expander.expand_rectangle(segmentation_map)
                        if segmentation_params['reverse']:
                            segmentation_map = ~segmentation_map
                        X = X * segmentation_map.unsqueeze(1)
                    elif segmentation_params['mitral']:
                        mitral_mask = expander.mitral_cover(segmentation_map)
                        if segmentation_params['reverse']:
                            mitral_mask = ~mitral_mask
                        X = X * (~mitral_mask).unsqueeze(1)
                    
                average = (len(X.shape) == 6)
                if average:
                    batch, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                if single_frame:
                    X = X[:,:,0]

                if save_first_path and i==0:
                    path = os.path.join(save_first_path, 'X.npy')
                    np.save(path, X.cpu().numpy())

                if embeddings:
                    x = model.module.stem(X)

                    x = model.module.layer1(x)
                    x = model.module.layer2(x)
                    x = model.module.layer3(x)
                    x = model.module.layer4(x)

                    x = model.module.avgpool(x)
                    outputs = x.flatten(1)

                    yhat.append(outputs.to("cpu").detach().numpy())
                    pbar.update()
                    n += 1

                    continue

                elif block_size is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j + block_size), ...]) for j in range(0, X.shape[0], block_size)])

                if save_all:
                    yhat.append(outputs.to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)

                if not save_all:
                    yhat.append(outputs.to("cpu").detach().numpy())

                n += X.size(0)

                if binary:
                    assert np.sum(~mask.cpu().detach().numpy()) == 0
                    probs = torch.sigmoid(outputs)
                    loss = torch.nn.functional.binary_cross_entropy(probs, outcome)

                else:
                    s1 += outcome[mask].sum()
                    s2 += (outcome[mask] ** 2).sum()

                    if len(std) > 1:
                        loss =  sum([torch.nn.functional.mse_loss(outputs[:,i][mask[:,i]], outcome[:,i][mask[:,i]], reduction='sum') / (len(std) * s**2 * (sum(mask[:,i]) + 1e-5))
                                 for i, s in enumerate(std)])

                    else:
                        loss = torch.nn.functional.mse_loss(outputs[mask], outcome[mask], reduction='sum') / (sum(mask) * std[0]**2 + 1e-5)

                    
                    pbar.set_postfix_str("{:.2f} ({:.2f})".format(total / n, loss.item()))


                total += loss.item() * X.size(0)

                pbar.update()

                if train:
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
                    optim.step()

                

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return total / n, yhat, y
