"""Functions for training and running lab prediction."""

import math
import os
import time


from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import pandas as pd

from tqdm import tqdm

import echonet


def run(modelname="r2plus1d_18",
        tasks="logBNP",
        task_to_explain="logBNP",
        frames=32,
        period=2,
        pretrained=True,
        output=None,
        device=None,
        num_workers=0,
        batch_size=1,
        seed=0,
        grad_type='smooth',
        num_batches=10,
        remove_nans=False,
        weight_decay=1e-4,
        noise=None,
        split = "val",
        dataset="echo",
        root=None,
        flow_mode="interp_8_8",
        weight_loss=True,
        magnitude=True,
        distractors=0,
        save_dir=None,
        randomized_weights='none',
        prefix="",
        std=.1,
        n_samples=100,
        toy_classify=True,
        filelist="all.csv",
        video_dir="",
        examples=None,
        highlow=False):

    assert dataset in ["echo", "kinetics", "toy"]
    assert randomized_weights in ['none', 'logits', 'all']

    grad_dict = {'vanilla': VanillaGrad, 'smooth': SmoothGrad, 'flow': VanillaFlowGrad, 'smoothflow': SmoothFlowGrad}

    flow_types = ['flow', 'smoothflow']

    if grad_type in flow_types:
        mode = flow_mode.split("_")[0]

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    if dataset == "echo":

        if not isinstance(tasks, list):
            tasks = [tasks]

        tasks = sorted(tasks)
        index = tasks.index(task_to_explain)

        # Set default output directory
        if output is None:
            task_str = "_" + "_".join(sorted(tasks))
            output = os.path.join("output", "video", "{}_{}_{}_{}_{}_{}_{}_{}_".format(
                modelname, frames, period, weight_decay, noise, "pretrained" if pretrained else "random", task_str, "removenan" if remove_nans else "masknan"))
        os.makedirs(output, exist_ok=True)

        n_tasks = len(tasks)


    elif dataset == "kinetics":

        if output is None:
            output = "output/kinetics_explanations"

        output = os.path.join(output, task_to_explain)
        os.makedirs(output, exist_ok=True)

        n_tasks = 400

    elif dataset == "toy":
        if noise is None:
            noise = 0
        if output is None:
            output = "output/toy_model_noise:{}_weight:{}_distractors:{}".format(
                noise, 1 if weight_loss else 0, distractors)
            os.makedirs(output, exist_ok=True)

    print(output)
    if save_dir is None:
        save_dir = output

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = torchvision.models.video.__dict__[modelname](pretrained=True)
    model.eval()

    if grad_type in ["flow"]:
        lead = int(flow_mode.split("_")[2])
    else:
        lead = 0

    if dataset == "echo":
        # Compute mean and std

        kwargs = {"target_type": tasks,
                  "length": frames + lead,
                  "period": period,
                  "filelist": filelist,
                  "root": root,
                  "video_dir": video_dir
                  }

        model.fc = torch.nn.Linear(model.fc.in_features, n_tasks)
        mean, data_std, target_mean = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train", **kwargs), all_features=True)
        if tasks != ['EF']:
            model.fc.bias.data = torch.from_numpy(target_mean)

        if examples is None:
            df = pd.read_csv(os.path.join(output, '{}_predictions.csv'.format(split)),
                         names=['files', 'starts', 'preds'])
            files = df.drop('starts', 1).groupby('files')\
                         .mean().sort_values('preds')
            if highlow:
                files_to_include = list(files.index)[-num_batches//2:] + list(files.index)[:num_batches//2]
            else:
                files_to_include = list(files.index)[-num_batches:]

            files_to_include = [f + '.avi' if '.' not in f else f for f in files_to_include]

            print(files_to_include)

        else:
            files_to_include = examples
            print(files_to_include)

        kwargs['mean'] = mean
        kwargs['std'] = data_std
        kwargs['files_to_include'] = files_to_include

        ds = echonet.datasets.Echo(split=split, **kwargs)


    elif dataset == "kinetics":
        kwargs ={"mean": np.array([0.43216, 0.394666, 0.37645]),
                 "std": np.array([0.22803, 0.22145, 0.216989]),
                 "length": frames + lead,
                 "period": 1,
                 "single_class": task_to_explain,
                 "root": root}

        ds = echonet.datasets.Kinetics(split=split, **kwargs)

        index = None

    elif dataset == "toy":
        model.fc = torch.nn.Linear(model.fc.in_features, 9 if toy_classify else 2)
        if task_to_explain is None:
            index = 0
        else:
            index = ["angle", "sides"].index(task_to_explain)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if dataset in ["echo", "toy"]:

        if randomized_weights in ["none", "logits"]:

            try:
                # Attempt to load checkpoint
                checkpoint = torch.load(os.path.join(output, "best.pt"))
            except FileNotFoundError:
                  
                raise FileNotFoundError("Checkpoint required for explainability, but none found in {}".format(output))
            
            model.load_state_dict(checkpoint['state_dict'])
            
            bestLoss = checkpoint["best_loss"]
            print("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if randomized_weights in ["logits"]:
            model.module.fc.reset_parameters()


        model.train(False)

    kwargs = {'cuda': True,
              'magnitude': magnitude}

    if grad_type in flow_types:
        kwargs['flow_mode'] = flow_mode

    if 'smooth' in grad_type:
        kwargs['stdev_spread'] = std
        kwargs['n_samples'] = n_samples

    grad_fn = grad_dict[grad_type](model, **kwargs)
    



    if dataset in ["echo", "kinetics"]:
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))

    elif dataset == "toy":
        dataset_params={"noise": noise,
                        "num_distractors": distractors,
                        "moving_distractors": True,
                        "stop_length": 0,
                        "num_frames": frames + lead}
        dataloader = echonet.utils.video_toy.batcher(num_batches, batch_size, 'test', dataset_params)

    Xs, yhat, y, grads, labels = run_explanation(grad_fn, dataloader, device, num_batches, index=index, toy_classify=task_to_explain if toy_classify else "")

    if dataset == "toy":
        task_to_explain = task_to_explain + str(noise)

    if grad_type in flow_types:

        if mode == "decomp":
            errors = [yh[2] for yh in yhat]
            errors = np.concatenate(errors)
            error_grads = [g[2] for g in grads]
            error_grads = np.concatenate(error_grads)
            np.save(os.path.join(save_dir, f"{prefix}errors_{grad_type}_{task_to_explain}_{flow_mode}.npy"), errors)
            np.save(os.path.join(save_dir, f"{prefix}errorgrads_{grad_type}_{task_to_explain}_{flow_mode}.npy"), error_grads)

        flows = [yh[1] for yh in yhat]
        yhat = [yh[0] for yh in yhat]

        flows = np.concatenate(flows)
        yhat = np.concatenate(yhat)

        x_grads = [g[0] for g in grads]
        flow_grads = [g[1] for g in grads]
        x_grads = np.array(x_grads)
        flow_grads = np.array(flow_grads)
        Xs = np.array(Xs)

        np.save(os.path.join(save_dir, f"{prefix}Xs_{grad_type}_{task_to_explain}_{flow_mode}.npy"), Xs)
        np.save(os.path.join(save_dir, f"{prefix}flows_{grad_type}_{task_to_explain}_{flow_mode}.npy"), flows)
        np.save(os.path.join(save_dir, f"{prefix}yhat_{grad_type}_{task_to_explain}_{flow_mode}.npy"), yhat)
        np.save(os.path.join(save_dir, f"{prefix}y_{grad_type}_{task_to_explain}_{flow_mode}.npy"), y)
        np.save(os.path.join(save_dir, f"{prefix}xgrads_{grad_type}_{task_to_explain}_{flow_mode}.npy"), x_grads)
        np.save(os.path.join(save_dir, f"{prefix}flowgrads_{grad_type}_{task_to_explain}_{flow_mode}.npy"), flow_grads)

        if len(labels) > 0:
            np.save(os.path.join(save_dir, f"{prefix}labels_{grad_type}_{task_to_explain}_{flow_mode}.npy"), labels)            


    else:
        yhat = np.concatenate(yhat)
        y = np.concatenate(y)
        Xs = np.array(Xs)
        grads = np.array(grads)
        np.save(os.path.join(save_dir, f"{prefix}Xs_{grad_type}_{task_to_explain}.npy"), Xs)
        np.save(os.path.join(save_dir, f"{prefix}yhat_{grad_type}_{task_to_explain}.npy"), yhat)
        np.save(os.path.join(save_dir, f"{prefix}y_{grad_type}_{task_to_explain}.npy"), y)
        np.save(os.path.join(save_dir, f"{prefix}grads_{grad_type}_{task_to_explain}.npy"), grads)

        if len(labels) > 0:
            np.save(os.path.join(save_dir, f"{prefix}labels_{grad_type}_{task_to_explain}.npy"), labels)


    if dataset == 'echo':
        np.save(os.path.join(save_dir, f"{prefix}names_{grad_type}_{task_to_explain}.npy"), ds.fnames)

def run_explanation(grad, dataloader, device, n=1, index=None, toy_classify=""):

    Xs = []
    yhat = []
    y = []
    grads = []
    labels = []


    for (i, data) in enumerate(dataloader):
        if i >= n:
            break

        if len(data) == 2:
            X, outcome = data
            label = None
        else:
            X, outcome, label = data

        if toy_classify == "angle":
            index = int((outcome[0, 0] - 90) // 30)
        elif toy_classify == "sides":
            index = int(outcome[0, 1] - 3 + 6)

        if i % 10 == 0:
            print(i)

        Xs.append(X.numpy())
        y.append(outcome.numpy())
        X = X.to(device)
        outcome = outcome.to(device)

        g, outputs = grad(X, index)

        yhat.append(outputs)
        grads.append(g)
        labels.append(label)

    return Xs, yhat, y, grads, labels


class VanillaGrad(object):
    # https://github.com/hs2k/pytorch-smoothgrad/blob/master/lib/gradients.py
    # (MIT licence)

    def __init__(self, pretrained_model, cuda=False, magnitude=False):
        self.pretrained_model = pretrained_model
        self.cuda = cuda
        self.magnitude = magnitude

    def __call__(self, x, index=None):

        x.requires_grad = True
        output = self.pretrained_model(x)[0]
        if index is None:
            index = np.argmax(output.data.cpu().numpy())
        grad = torch.autograd.grad(output[index], x)
        grad = grad[0].cpu().detach().numpy()
        if self.magnitude:
            grad = grad ** 2

        return grad, output.cpu().detach().numpy()

class SmoothGrad(VanillaGrad):

    def __init__(self, pretrained_model, cuda=False, stdev_spread=0.15,
                 n_samples=100, magnitude=True):
        super(SmoothGrad, self).__init__(pretrained_model, cuda, magnitude)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples

    def __call__(self, x, index=None):
        base_output = self.pretrained_model(x).cpu().detach().numpy()
        x = x.data.cpu().numpy()
        stdev = self.stdev_spread * (np.max(x) - np.min(x))
        total_gradients = np.zeros_like(x)
        for i in range(self.n_samples):
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            if self.cuda:
                x_plus_noise = torch.from_numpy(x_plus_noise).cuda()
            else:
                x_plus_noise = torch.from_numpy(x_plus_noise)

            if index is None:
                index = np.argmax(base_output)

            x_plus_noise.requires_grad = True
            output = self.pretrained_model(x_plus_noise)[0]

            grad = torch.autograd.grad(output[index], x_plus_noise)
            grad = grad[0].cpu().detach().numpy()

            if self.magnitude:
                total_gradients += (grad * grad)
            else:
                total_gradients += grad
             
        avg_gradients = total_gradients / self.n_samples

        return avg_gradients, base_output

class VanillaFlowGrad(object):
    # https://github.com/hs2k/pytorch-smoothgrad/blob/master/lib/gradients.py
    # (MIT licence)

    def __init__(self, pretrained_model, cuda=False, flow_mode="interp_8_8", magnitude=False):
        self.pretrained_model = pretrained_model
        self.cuda = cuda
        self.magnitude = magnitude

        mode_params = flow_mode.split("_")
        self.mode = mode_params[0]
        assert self.mode in ["interp", "decomp"]

        self.skip = 1

        if self.mode == "interp":
            self.interp_level = int(mode_params[1])
            self.lead = int(mode_params[2])
            if len(mode_params) == 4:
                self.skip = int(mode_params[3])
        elif self.mode == "decomp":
            self.lead = int(mode_params[1])
            self.skip = int(mode_params[2])



    def __call__(self, x, index=None):
        output = self.pretrained_model(x)[0]

        if index is None:
            index = np.argmax(output.data.cpu().numpy())

        x = x[0]
        x = x.transpose(1, 0)

        flow = echonet.utils.flow_utils.compute_flow(x.cpu().numpy(), skip=self.skip)
        flow = torch.from_numpy(flow).cuda()

        flow_weights = torch.ones_like(flow[:,:,:,:1])
        flow_weights.requires_grad = True
        flow = flow * flow_weights

        x.requires_grad = True

        ret_data = [output.cpu().detach().numpy(), flow.cpu().detach().numpy()]

        if self.mode == "interp":
            int_x = echonet.utils.flow_utils.torch_interpolate(x, flow, skip=self.skip,
                clip=False, use_torch=True, n_levels=self.interp_level).unsqueeze(0).transpose(1, 2)
            to_grad = [x, flow_weights]


        elif self.mode == "decomp":
            int_x, errors, _ = echonet.utils.flow_utils.decompose(x, flow, skip=self.skip)
            int_x = int_x.unsqueeze(0).transpose(1, 2)
            to_grad = [x, flow_weights] + errors
            ret_data.append(torch.stack(errors).detach().cpu().numpy())

        int_x = int_x[:,:,self.lead:]

        output = self.pretrained_model(int_x)[0]
        grad = torch.autograd.grad(output[index], to_grad, allow_unused=True)

        if self.magnitude:
            grad = [g ** 2 for g in grad]

        if self.mode == "decomp":
            grad = [g.cpu().detach().numpy() for g in grad[:2]] + [torch.stack(grad[2:]).detach().cpu().numpy()] 
        else:
            grad = [g.cpu().detach().numpy() for g in grad]

        return grad, ret_data

class SmoothFlowGrad(VanillaFlowGrad):

    def __init__(self, pretrained_model, cuda=False, stdev_spread=0.15,
                 n_samples=100, magnitude=False, flow_mode="interp_8_8"):
        super(SmoothFlowGrad, self).__init__(pretrained_model, cuda, flow_mode, magnitude)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples

    def __call__(self, x, index=None):

        base_output = self.pretrained_model(x)[0]

        if index is None:
            index = np.argmax(base_output.data.cpu().numpy())

        x = x[0]
        x = x.transpose(1, 0)

        x = x.data.cpu().numpy()
        flow = echonet.utils.flow_utils.compute_flow(x, skip=self.skip)


        x_stdev = self.stdev_spread * (np.max(x) - np.min(x))
        f_stdev = self.stdev_spread

        if self.mode == 'decomp':
            _, error, _ = echonet.utils.flow_utils.decompose(torch.from_numpy(x).cuda(), torch.from_numpy(flow).cuda(), skip=self.skip)
            error = torch.stack(error)
            error_np = error.detach().cpu().numpy()
            e_stddev = self.stdev_spread * (np.max(error_np) - np.min(error_np))
            total_e_gradients = np.zeros_like(error_np)
        else:
            total_e_gradients = 0
            error_np = np.zeros(1)

        total_x_gradients = np.zeros_like(x)
        total_f_gradients = np.zeros_like(flow[:,:,:,:1])
        total_e_gradients = np.zeros_like(error_np)

        for i in range(self.n_samples):
            x_noise = np.random.normal(0, x_stdev, x.shape).astype(np.float32)
            x_plus_noise = x + x_noise
            x_plus_noise = torch.from_numpy(x_plus_noise).cuda()
            x_plus_noise.requires_grad = True

            f_noise = np.random.normal(0, f_stdev, flow[:,:,:,:1].shape).astype(np.float32) + 1
            flow_weights = torch.from_numpy(f_noise).cuda()
            flow_weights.requires_grad = True
            f = torch.from_numpy(flow).cuda()
            f_plus_noise = f * flow_weights

            if self.mode == "interp":
                int_x = echonet.utils.flow_utils.torch_interpolate(x_plus_noise, f_plus_noise, skip=self.skip,
                    clip=False, use_torch=True, n_levels=self.interp_level).unsqueeze(0).transpose(1, 2)
                to_grad = [x_plus_noise, flow_weights]

            elif self.mode == "decomp":
                e_noise = np.random.normal(0, e_stddev, error.shape).astype(np.float32)
                error_plus_noise = torch.from_numpy(e_noise + error_np).cuda()
                error_plus_noise.requires_grad = True

                int_x, errors, _ = echonet.utils.flow_utils.decompose(x_plus_noise, f_plus_noise, skip=self.skip, set_errors=error_plus_noise)
                int_x = int_x.unsqueeze(0).transpose(1, 2)
                to_grad = [x_plus_noise, flow_weights, error_plus_noise]
                # ret_data.append(torch.stack(errors).detach().cpu().numpy())

            int_x = int_x[:,:,self.lead:]

            output = self.pretrained_model(int_x)[0]
            grad = torch.autograd.grad(output[index], to_grad)
            x_grad = grad[0].cpu().detach().numpy()
            flow_grad = grad[1].cpu().detach().numpy()

            if self.magnitude:
                total_x_gradients += (x_grad * x_grad)
                total_f_gradients += (flow_grad * flow_grad)
            else:
                total_x_gradients += x_grad
                total_f_gradients += flow_grad

            if self.mode == "decomp":
                e_grad = grad[2].cpu().detach().numpy()
                total_e_gradients += e_grad ** 2 if self.magnitude else e_grad

        avg_x = total_x_gradients / self.n_samples
        avg_f = total_f_gradients / self.n_samples
        avg_e = total_e_gradients / self.n_samples

        if self.mode == "decomp":
            return (avg_x, avg_f, avg_e), (base_output.data.cpu().numpy(), flow, error_np)
        return (avg_x, avg_f), (base_output.data.cpu().numpy(), flow)



