import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import gc
from copy import deepcopy
from .loss import angle_criterion, dist_criterion
from .dataset import build_dataset, build_dataset_no_transform
DEFAULT_DEVICE = torch.device('cuda:0')

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

def load_checkpoint(model, optimizer, load_path, device_id=DEFAULT_DEVICE):
    checkpoint = torch.load(load_path, map_location=device_id)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch


def get_support_query_dfs(novel_datasets, sc=None, n_shots=10):
    support_dfs = []
    query_dfs = []
    found_ids = []
    for cls in sc:
        df = novel_datasets[(novel_datasets[cls] == 1)&(~novel_datasets.index.isin(
            found_ids))].sample(n_shots*2)

        s_df = df.iloc[:n_shots]
        q_df = df.iloc[n_shots:]
        support_dfs.append(s_df)
        query_dfs.append(q_df)
        found_ids += s_df.index.values.tolist()
        found_ids += q_df.index.values.tolist()

    class_df = pd.concat(support_dfs + query_dfs)
    support_df = pd.concat(support_dfs)
    query_df = pd.concat(query_dfs)
    
    return class_df, support_df, query_df
    
def get_episode_dataset(class_df, support_df, query_df, dt = None, sc = None):
    
    novel_dataset = {}
    labels_dict = class_df.to_dict(orient='index')#base_df.set_index('study_id').to_dict(orient='index')

    novel_dataset['image'] = {i[0]: i[1] for i in dt['image'].items() if i[0] in class_df.index}

    labels_dict = {k: np.array(list(v.values()))[sc] for k,v in labels_dict.items() if k in novel_dataset['image']}

    novel_dataset['label'] = labels_dict

    novel_dataset['split'] = {
        'train': support_df.index.values, 'val1':support_df.index.values, 'val2': support_df.index.values,
        'test': query_df.index.values
    }
    
    return novel_dataset

def get_episode_dataloaders(novel_dataset, bs = None, cfg=None):
    
    dataset = build_dataset(cfg=cfg, out_dir='C:\\Users\\saura\\Notebooks\\output', dataset=novel_dataset)

    val_dataset = build_dataset(mode='val', cfg=cfg, out_dir='C:\\Users\\saura\\Notebooks\\output',  dataset=novel_dataset)

    test_dataset = build_dataset(mode='test', cfg=cfg, out_dir='C:\\Users\\saura\\Notebooks\\output',  dataset=novel_dataset)

    novel_image_datasets = {
        'train': dataset, 'val': val_dataset, 'test': test_dataset
    }

    novel_dataloaders = {x: torch.utils.data.DataLoader(novel_image_datasets[x], batch_size=bs,
                                                 shuffle=True, num_workers=1)
                  for x in ['train', 'val', 'test']}
    
    return novel_dataloaders, novel_image_datasets


def get_episode_dataloaders_no_transform(novel_dataset, bs = None, cfg=None):
    
    dataset = build_dataset_no_transform(cfg=cfg, out_dir='C:\\Users\\saura\\Notebooks\\output', dataset=novel_dataset)

    val_dataset = build_dataset_no_transform(mode='val', cfg=cfg, out_dir='C:\\Users\\saura\\Notebooks\\output',  dataset=novel_dataset)

    test_dataset = build_dataset_no_transform(mode='test', cfg=cfg, out_dir='C:\\Users\\saura\\Notebooks\\output',  dataset=novel_dataset)

    novel_image_datasets = {
        'train': dataset, 'val': val_dataset, 'test': test_dataset
    }

    novel_dataloaders = {x: torch.utils.data.DataLoader(novel_image_datasets[x], batch_size=bs,
                                                 shuffle=True, num_workers=1)
                  for x in ['train', 'val', 'test']}
    
    return novel_dataloaders, novel_image_datasets


def evaluate(to_use, student, device=DEFAULT_DEVICE):
    student.eval()
    with torch.no_grad():
        test_labels = []
        all_predicted = []
        for inputs, labels, uid in to_use:
            _, outputs = student(inputs.to(device))

            predicted = (torch.sigmoid(outputs)).type(torch.FloatTensor).detach().cpu()
            all_predicted.append(predicted)
            test_labels.append(labels)

        y_trues = torch.concat(test_labels).numpy()

        all_preds = torch.concat(all_predicted)

        class_auc = []

        for i in tqdm(range(all_preds.shape[-1])):
            if len(np.unique(y_trues[:, i])) <= 1:
                continue
            class_auc.append(roc_auc_score(y_trues[:,i],all_preds[:,i])) 
  
    return class_auc


def train_fewshot_model_new(model, criterion, optimizer, scheduler, num_epochs=25, pth='resnet18.pth', dataloaders=None, verbose=True,
               class_weight_vector=None, network='', gradient_clipping=True, device=DEFAULT_DEVICE, cfg = None
               ):

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        loss = 0
        for inputs, labels, uid in dataloaders['train']:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)

            if network == 'residual_network':
                _, outputs = model(inputs)
            else:
                outputs = model(inputs)

            bce_loss = criterion(outputs, labels)
            bce_loss = bce_loss.mean()
            bce_loss.backward()
            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)  
            optimizer.step()
            loss += bce_loss.item()
        # print(loss)

    return model

def distill_rkd(student, model_finetune, 
	novel_dataloaders, device= DEFAULT_DEVICE, cfg = None, epochs=25):
    temperature = 5
    n_way = 5

    optimizer_student = optim.Adam(student.parameters(), lr=0.0001)#, weight_decay=0.0001)

    for epoch in range(epochs):
        phase='train'
        running_loss = 0.0
        running_corrects = 0
        print_cnt = 0
        # Iterate over data.
        student.train()
        for inputs, labels, uid in novel_dataloaders[phase]:
            print_cnt+=1
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer_student.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                st_o, st_outputs = student(inputs.to(device))
                t_o, outputs = model_finetune(inputs.to(device))

                bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(st_outputs, labels)
    #                 bce_loss = bce_loss / n_way

#                 total_loss = bce_loss
                kd_loss_1 = nn.functional.kl_div(
                    (st_outputs / temperature).sigmoid().log(),
                    (outputs / temperature).sigmoid(),
                    reduction="batchmean",
                )

                kd_loss_2 = nn.functional.kl_div(
                    (1-(st_outputs / temperature).sigmoid()).log(),
                    (1-(outputs / temperature).sigmoid()),
                    reduction="batchmean",
                )

                kd_loss = (kd_loss_1 + kd_loss_2) / (n_way) #*(temperature ** 2)

    #             st_o = torch.flatten(student_backbone(inputs.to(device)),start_dim=1)
    #             t_o = torch.flatten(teacher_backbone(inputs.to(device)), start_dim=1)

                angle_loss = angle_criterion(st_o, t_o)
                dist_loss = dist_criterion(st_o, t_o)

                total_loss = bce_loss + 10*kd_loss + 50*angle_loss  + 10*dist_loss
                
                if phase == 'train':
                    total_loss.backward()
#                     torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.clip_max_norm)                    
                    optimizer_student.step()
#             if print_cnt%50==0:
#                 print(running_loss/print_cnt)
            # statistics
            running_loss += total_loss.item()        
        print('id epoch:', epoch, running_loss)

        phase = 'test'
        running_loss = 0.0
        print_cnt = 0
        student.train()
        for inputs, labels, uid in novel_dataloaders[phase]:
            print_cnt+=1
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_student.zero_grad()

            t_o, outputs = model_finetune(inputs.to(device))
            st_o, st_outputs = student(inputs.to(device))

            kd_loss_1 = nn.functional.kl_div(
                (st_outputs / temperature).sigmoid().log(),
                (outputs / temperature).sigmoid(),
                reduction="batchmean",
            )

            kd_loss_2 = nn.functional.kl_div(
                (1-(st_outputs / temperature).sigmoid()).log(),
                (1-(outputs / temperature).sigmoid()),
                reduction="batchmean",
            )

            kd_loss = (kd_loss_1 + kd_loss_2) / (n_way)
#             st_o = torch.flatten(student_backbone(inputs.to(device)),start_dim=1)
#             t_o = torch.flatten(teacher_backbone(inputs.to(device)), start_dim=1)

            angle_loss = angle_criterion(st_o, t_o)
            dist_loss = dist_criterion(st_o, t_o)
            total_loss = 10*kd_loss + 50*angle_loss + 10*dist_loss
            
            total_loss.backward()
    #             torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.clip_max_norm)                    
            optimizer_student.step()
            if print_cnt%50==0:
                print(running_loss/print_cnt)
            running_loss += total_loss.item()/cfg.batch_size
        print('td epoch:', epoch, running_loss)
        print('----')
    print('-- Finished Transductive KD')
    return student

def voronoi_plot_2d_patch(vor, ax=None, **kw):
    """
    Plot the given Voronoi diagram in 2-D
    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
        Diagram to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on
    show_points : bool, optional
        Add the Voronoi points to the plot.
    show_vertices : bool, optional
        Add the Voronoi vertices to the plot.
    line_colors : string, optional
        Specifies the line color for polygon boundaries
    line_width : float, optional
        Specifies the line width for polygon boundaries
    line_alpha : float, optional
        Specifies the line alpha for polygon boundaries
    point_size : float, optional
        Specifies the size of points
    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot
    See Also
    --------
    Voronoi
    Notes
    -----
    Requires Matplotlib.
    Examples
    --------
    Set of point:
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> points = rng.random((10,2))
    Voronoi diagram of the points:
    >>> from scipy.spatial import Voronoi, voronoi_plot_2d
    >>> vor = Voronoi(points)
    using `voronoi_plot_2d` for visualisation:
    >>> fig = voronoi_plot_2d(vor)
    using `voronoi_plot_2d` for visualisation with enhancements:
    >>> fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
    ...                 line_width=2, line_alpha=0.6, point_size=2)
    >>> plt.show()
    """
    from matplotlib.collections import LineCollection

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if kw.get('show_points', True):
        point_size = kw.get('point_size', None)
        ax.plot(vor.points[:,0], vor.points[:,1], '.', markersize=point_size)
    if kw.get('show_vertices', True):
        ax.plot(vor.vertices[:,0], vor.vertices[:,1], 'o')

    line_colors = kw.get('line_colors', 'k')
    line_width = kw.get('line_width', 1.0)
    line_alpha = kw.get('line_alpha', 1.0)

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            infinite_segments.append([vor.vertices[i], far_point])

    ax.add_collection(LineCollection(finite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='solid'))
    ax.add_collection(LineCollection(infinite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='dashed'))

#     _adjust_bounds(ax, vor.points)

    return ax.figure