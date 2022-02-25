"""
    Train image to fMRI encoder by supervised learning.
    Date created: 8/25/19
    Python Version: 3.6
"""

__author__ = "Guy Gaziv"
__email__ = "guy.gaziv@weizmann.ac.il"

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import pretrainedmodels as pm
import pretrainedmodels.utils as pmutils
from self_super_reconst.utils import *
from self_super_reconst.config_enc import *
from absl import app
from self_super_reconst.utils.misc import set_gpu
np.seterr(divide='ignore', invalid='ignore')

def main(argv):
    del argv
    cprint1(FLAGS.exp_prefix)
    cprint1('== Summary level: {} =='.format(FLAGS.sum_level))

    set_gpu()

    # Data
    fmri_transform = np.float32
    get_dataset = lambda subset_case: \
        KamitaniDataset(FLAGS.roi, sbj_num=FLAGS.sbj_num, im_res=FLAGS.im_res, fmri_xfm=fmri_transform, subset_case=subset_case, select_voxels=FLAGS.select_voxels, is_rgbd=FLAGS.is_rgbd)

    train = get_dataset(KamitaniDataset.TRAIN)
    val = KamitaniDataset(FLAGS.roi, sbj_num=FLAGS.sbj_num, fmri_xfm=np.float32, subset_case=KamitaniDataset.TEST_AVG, select_voxels=FLAGS.select_voxels, is_rgbd=FLAGS.is_rgbd)

    global voxel_nc, N_VOXELS
    N_VOXELS = train.n_voxels

    voxel_nc = val.get_voxel_score('noise_ceil')

    voxel_indices_nc_sort = np.argsort(voxel_nc)[::-1]
    cprintm(u'(*) n_voxels: {} | noise ceiling {:.2f} \u00B1 {:.2f} (Mean \u00B1 SD)'.format(N_VOXELS, voxel_nc.mean(), voxel_nc.std()))

    if FLAGS.separable:
        model = make_model('SeparableEncoderVGG19ml', train.n_voxels, FLAGS.random_crop_pad_percent, drop_rate=0.5)
    else:
        model = make_model('BaseEncoderVGG19ml', train.n_voxels, FLAGS.random_crop_pad_percent, drop_rate=0.5)

    if FLAGS.is_rgbd:
        if FLAGS.is_rgbd == 1:  # RGBD
            if FLAGS.norm_within_img:
                normalizer = norm_imagenet_norm_depth_img
            else:
                normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.461], std=[0.229, 0.224, 0.225, 0.305])  # RGBD
        else:  # Depth only
            normalizer = norm_depth_img
    else:
        normalizer = NormalizeImageNet()

    img_xfm_basic = transforms.Compose([
        transforms.Resize(size=im_res(), interpolation=Image.BILINEAR),
        transforms.CenterCrop(im_res()),
        transforms.ToTensor(),
        normalizer
        ])

    img_xfm_train = transforms.Compose([
        transforms.Resize(size=im_res(), interpolation=Image.BILINEAR),
        transforms.RandomCrop(size=im_res(), padding=int(FLAGS.random_crop_pad_percent / 100 * im_res()), padding_mode='edge'),
        transforms.ToTensor(),
        normalizer
    ])

    model = model.cuda()
    cudnn.benchmark = True

    # Split to train/val
    train = CustomDataset(train, input_xfm=img_xfm_train)
    val = CustomDataset(val, input_xfm=img_xfm_basic)

    # Loaders
    trainloader, testloader = map(lambda dset: data.DataLoader(dset, batch_size=min(FLAGS.batch_size, len(dset)), shuffle=True,
                                                               num_workers=num_workers(), pin_memory=True), [train, val])
    # Optimizer/Scheduler
    trainable_params = chained([m.parameters() if isinstance(m, nn.Module) else [m] for m in model.trainable])
    if FLAGS.train_bbn:
        trainable_params0 = [x for x in trainable_params]
        trainable_params.extend(list(model.multi_branch_bbn.parameters()))

    print('    Total params for training: %s | %s' % (param_count_str(trainable_params), param_count_str(model.parameters())))

    if FLAGS.train_bbn:
        optimizer = optim.Adam([
            {'params': trainable_params0},
            {'params': model.multi_branch_bbn.parameters(), 'lr': 1e-6}
            ], lr=FLAGS.learning_rate)
    else:
        optimizer = optim.Adam(trainable_params, lr=FLAGS.learning_rate)

    model = nn.DataParallel(model)
    if FLAGS.init_cpt_name:
        # Load pretrained encoder
        assert os.path.isfile(init_cpt_path())
        print('\t==> Loading checkpoint {}'.format(basename(init_cpt_path())))
        model.load_state_dict(torch.load(init_cpt_path())['state_dict'])

    # Regularization
    reg_loss_dict = {}
    tau = 800.
    m = 1/8000.
    def calc_mom2(W, xx, yy, dd):
        m00 = W.view(len(W),-1).sum(1)
        m10 = (xx * W).view(len(W), -1).sum(1)
        m01 = (yy * W).view(len(W), -1).sum(1)
        m02 = ((yy - (m01/m00).view(-1,1,1))**2 * W).view(len(W), -1).sum(1)
        m20 = ((xx - (m10/m00).view(-1,1,1))**2 * W).view(len(W), -1).sum(1)
        return (m02 + m20).sum() / (2 * dd * len(W))

    xx_list = []
    yy_list = []
    for out_shape in model.module.out_shapes:
        dd = out_shape[-1]
        xx, yy = torch.tensor(np.expand_dims(np.mgrid[:dd, :dd], axis=1).repeat(N_VOXELS, axis=1), dtype=torch.float32, requires_grad=False).cuda()
        xx_list.append(xx); yy_list.append(yy)

    def group_reg(reg_type='fcmom2'):
        if FLAGS.separable:
            loss_list = []
            for idx, (out_shape, xx, yy) in enumerate(zip(model.module.out_shapes, xx_list, yy_list)):
                m = model.module.space_maps[str(out_shape[-1])]
                W_sub = list(m.parameters())[0]  # VxS
                W_sub = W_sub.view(len(W_sub), int(np.sqrt(W_sub.shape[-1])), -1)  # VxHxW
                if reg_type=='fcmom2':
                    W_sub = W_sub**2
                    loss_list.append(calc_mom2(W_sub, xx, yy, out_shape[-1]))
                elif reg_type=='gl':
                    Wsq_pad = F.pad(W_sub.unsqueeze(1)**2, [1, 1, 1, 1], mode='reflect').squeeze()
                    reg_l1 = 5e-6
                    reg_gl = 1e-5

                    Wn = (Wsq_pad[..., :-2, 1:-1] + Wsq_pad[...,2:, 1:-1] + Wsq_pad[...,1:-1, :-2] + Wsq_pad[..., 1:-1, 2:])/4
                    reg_loss = reg_l1 * W_sub.abs().sum() + reg_gl * Wn.sqrt().sum()
                    loss_list.append(reg_loss)
                else:
                    raise NotImplementedError
            return torch.stack(loss_list).sum(0)
        elif isinstance(model.module.fc_head, EncFCFWRF):
            W = model.module.fc_head.rf**2
        else:
            W = list(model.module.fc_head[0].parameters())[0]
            loss_list = []
            indices_cumsum = np.insert(np.cumsum([np.prod(s) for s in model.module.out_shapes]), 0, 0)
            for idx, (out_shape, xx, yy) in enumerate(zip(model.module.out_shapes, xx_list, yy_list)):
                W_sub = W[:, indices_cumsum[idx]:indices_cumsum[idx+1]]
                W_sub = W_sub.view(len(W_sub), *out_shape)  # VxCxHxW
                if reg_type=='fcmom2':
                    W_sub = (W_sub**2).sum(axis=1)
                    loss_list.append(calc_mom2(W_sub, xx, yy, out_shape[-1]))
                elif reg_type=='gl':
                    Wsq_pad = F.pad(W_sub**2, [1, 1, 1, 1], mode='reflect')
                    n_reg = .5
                    ch_mult = 1.5
                    reg_l1 = 20 * ch_mult
                    reg_gl = 800 * ch_mult
                    Wn = (W_sub**2 + n_reg / 4 * (Wsq_pad[..., :-2, 1:-1] + Wsq_pad[...,2:, 1:-1] + Wsq_pad[...,1:-1, :-2] + Wsq_pad[..., 1:-1, 2:])) / (1 + n_reg)
                    reg_loss = reg_l1 * W_sub.abs().mean() + reg_gl * Wn.mean(axis=1).sqrt().mean()
                    loss_list.append(reg_loss)
                else:
                    raise NotImplementedError
            return torch.stack(loss_list).sum(0)

    if FLAGS.separable:
        reg_loss_dict = {
            'L1reg_convs': (FLAGS.l1_convs, lambda : sum([param.abs().sum() for param in chained([m.parameters() for m in model.modules() if isinstance(m, nn.Conv2d)]) if param.ndim == 4])),
            'mom2': (FLAGS.fc_mom2, lambda : group_reg('fcmom2')),
            'gl': (FLAGS.fc_gl, lambda : group_reg('gl')),
            'L1chan_mix': (FLAGS.l1_chan_mix, lambda : sum([chan_mix.chan_mix.abs().sum() for chan_mix in model.module.chan_mixes])),
            'L1branch_mix': (FLAGS.l1_branch_mix, lambda : model.module.branch_mix.abs().sum()),
    }
    else:
        reg_loss_dict = {
            'L1reg_convs': (FLAGS.l1_convs, lambda : sum([param.abs().sum() for param in chained([m.parameters() for m in model.modules() if isinstance(m, nn.Conv2d)]) if param.ndim == 4])),
            'mom2': (FLAGS.fc_mom2, lambda : group_reg('fcmom2')),
            'gl': (FLAGS.fc_gl, lambda : group_reg('gl'))
        }

    for reg_loss_name, (w, _) in reg_loss_dict.items():
        if callable(w) or w > 0:
            cprintm('(+) {} {} loss.'.format(w, reg_loss_name))

    scheduler = None
    if FLAGS.scheduler > 0:
        if FLAGS.scheduler == 1:
            cprintm('(+) Using scheduler: On-Plateau.')
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, cooldown=1)
        elif FLAGS.scheduler == 12345:  # Milestones
            milestones = [20, 35, 45, 50]
            cprintm('(+) Using scheduler: tenth by milestones {} epochs.'.format(', '.join([str(x) for x in milestones])))
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
        else:
            cprintm('(+) Using scheduler: {} every {} epochs.'.format(FLAGS.gamma, FLAGS.scheduler))
            scheduler = optim.lr_scheduler.StepLR(optimizer, FLAGS.scheduler, gamma=FLAGS.gamma)

    # Loss
    criterion = lambda pred, actual: FLAGS.mse_loss * F.mse_loss(pred, actual) + FLAGS.cos_loss * cosine_loss(pred, actual)

    # Training
    with SummaryWriter(comment='EncTrain') if FLAGS.sum_level > 0 else dummy_context_mgr() as sum_writer:
        global global_step
        global_step = 0
        best_loss = np.inf
        with tqdm(desc='Epochs', total=FLAGS.n_epochs) if FLAGS.verbose < 1 else dummy_context_mgr() as pbar:
            if FLAGS.verbose < 1:
                pbar.update(0)
            for epoch in range(FLAGS.n_epochs):
                if FLAGS.verbose > 0:
                    print('\nEpoch: [%d | %d]' % (epoch + 1, FLAGS.n_epochs))

                _, _, voxel_pearson_r_avg, collected_fmri = \
                    train_test_regress(trainloader, model, criterion, optimizer, reg_loss_dict=reg_loss_dict, sum_writer=sum_writer)


                if FLAGS.pw_corr_win:
                    voxel_pearson_r_pw = pearson_corr_piecewise(*collected_fmri.values(), win_size=FLAGS.pw_corr_win)
                    sum_writer.add_figure('TrainEnc/Vox_PWCorr_vs_Corr', corr_vs_corr_plot(voxel_pearson_r_avg, voxel_pearson_r_pw, ax_labels=['Pearson R', 'PW R']), epoch)

                test_loss, meters_test, voxel_pearson_r_avg, collected_test = \
                    train_test_regress(testloader, model, criterion, reg_loss_dict=reg_loss_dict, sum_writer=sum_writer)

                # Consider test loss based on criteria
                test_loss = dict(meters_test)['LossCriterion']

                if FLAGS.pw_corr_win:
                    voxel_pearson_r_pw = pearson_corr_piecewise(*collected_test.values(), win_size=FLAGS.pw_corr_win)
                    sum_writer.add_figure('ValEnc/Vox_PWCorr_vs_Corr', corr_vs_corr_plot(voxel_pearson_r_avg, voxel_pearson_r_pw, ax_labels=['Pearson R', 'PW R']), epoch)
                    collected_test = dict((k, v.flatten().tolist()) for k, v in collected_fmri.items())

                if sum_writer:
                    if epoch in [5, FLAGS.n_epochs - 1] or epoch % 10 == 0:
                        if not FLAGS.separable:
                            if isinstance(model.module.fc_head, EncFCFWRF):
                                rf_params, n_out_planes = model.module.fc_head.rf, 1
                            else:
                                rf_params, n_out_planes = list(model.module.fc_head[0].parameters()), model.module.n_out_planes
                            sum_writer.add_figure(

                                'TrainEnc/EpochVoxRF', vox_rf(rf_params, n_out_planes, voxel_indices_nc_sort, mask=isinstance(model.module.fc_head, EncFCFWRF)), epoch)

                        sum_writer.add_figure('ValEnc/OutDist', my_hist_comparison_fig(stack2numpy(collected_test), 100), epoch)
                        sum_writer.add_figure('ValEnc/Vox_Corr_vs_NC', corr_vs_corr_plot(voxel_nc, voxel_pearson_r_avg), epoch)


                    for metric_name, meter_avg in meters_test:
                        sum_writer.add_scalar('ValEnc/{}'.format(metric_name), meter_avg, epoch)

                if scheduler:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(test_loss)
                    else:
                        scheduler.step(epoch)

                # save model
                is_best = test_loss < best_loss
                best_loss = min(test_loss, best_loss)
                if FLAGS.may_save:
                    if is_best or not FLAGS.savebestonly:
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_loss': best_loss,
                            'optimizer': optimizer.state_dict(),
                        }, is_best, filepath=get_checkpoint_out())
                if FLAGS.verbose < 1:
                    pbar.update()

        # Report
        cprintm('    * TRAINING COMPLETE *')
    cprint1(FLAGS.exp_prefix)
    with open(f'{PROJECT_ROOT}/runs/{FLAGS.exp_prefix}.txt', 'w') as f:
        f.write(FLAGS.flags_into_string())

def train_test_regress(loader, model, criterion, optimizer=None, reg_loss_dict={},  sum_writer=None):
    global global_step
    if optimizer:
        mode = 'Train'
        # switch to train mode
        model.train()
    else:
        mode = 'Val'
        model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_names = ['total'] + ['criterion', 'mae', 'cosine_loss'] + list(reg_loss_dict.keys())
    losses = dict(zip(losses_names, [AverageMeter() for _ in range(len(losses_names))]))
    corrs = { metric_tup: AverageMeter() for metric_tup in [(np.median, 'R_median'),
                                                            (lambda x: np.percentile(x, 90), 'R_90'),
                                                            (lambda x: np.percentile(x, 75), 'R_75'),
                                                            ]}
    corrs_nc_norm = copy.deepcopy(corrs)
    corrs_nc_norm = { (metric_func, metric_name.replace('R', 'R_ncnorm')): meter for (metric_func, metric_name), meter in corrs_nc_norm.items() }
    voxel_pearson_r = AverageMeter()
    end = time.time()
    collected_fmri = {'actual': [], 'pred': []}
    with tqdm(desc=mode, total=len(loader)) if FLAGS.verbose > 0 else dummy_context_mgr() as bar:
        for batch_idx, (images, fmri_actual) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if FLAGS.sum_level > 3:
                if FLAGS.sum_level > 4 or not optimizer or FLAGS.pw_corr_win:
                    actual_fmri_list = list(np.squeeze(sample_portion(fmri_actual, 1/10).flatten()))
                    if FLAGS.pw_corr_win:
                        collected_fmri['actual'].append(fmri_actual)
                    else:
                        collected_fmri['actual'].extend(actual_fmri_list)

            fmri_actual = fmri_actual.cuda()
            images = images.cuda()

            with dummy_context_mgr() if optimizer else torch.no_grad():
                fmri_pred = model(images, detach_bbn=FLAGS.allow_bbn_detach)

            loss_criterion = criterion(fmri_pred, fmri_actual)
            losses['criterion'].update(loss_criterion.data, fmri_actual.size(0))
            losses['mae'].update(F.l1_loss(fmri_pred, fmri_actual).data, fmri_actual.size(0))
            losses['cosine_loss'].update(cosine_loss(fmri_pred, fmri_actual).data, fmri_actual.size(0))

            reg_loss_tot = 0
            for loss_name, (w, reg_loss_func) in reg_loss_dict.items():
                reg_loss = reg_loss_func()
                losses[loss_name].update(reg_loss.data, fmri_actual.size(0))
                if callable(w):
                    w = w(global_step)
                reg_loss_tot += w * reg_loss

            loss = loss_criterion + reg_loss_tot

            losses['total'].update(loss.data, fmri_actual.size(0))

            voxel_pearson_r.update(pearson_corr(fmri_pred.data, fmri_actual.data).cpu().numpy())
            for (metric_func, _), meter in corrs.items():
                meter.update(metric_func(voxel_pearson_r.val), fmri_actual.size(0))

            voxel_pearson_r_ncnorm = voxel_pearson_r.val / voxel_nc

            for (metric_func, _), meter in corrs_nc_norm.items():
                meter.update(metric_func(voxel_pearson_r_ncnorm), fmri_actual.size(0))

            if FLAGS.sum_level > 3:
                if FLAGS.sum_level > 4 or not optimizer or FLAGS.pw_corr_win:
                    pred_fmri_list = list(np.squeeze(sample_portion(fmri_pred, 1 / 10).cpu().detach().flatten()))
                    if FLAGS.pw_corr_win:
                        collected_fmri['pred'].append(fmri_pred.cpu().detach())
                    else:
                        collected_fmri['pred'].extend(pred_fmri_list)

            if optimizer:
                model.zero_grad()

                loss.backward()
                optimizer.step()
                global_step += 1
                if sum_writer:
                    for loss_name, meter in losses.items():
                        sum_writer.add_scalar('TrainEnc/Loss{}'.format(loss_name.capitalize()), meter.val, global_step)
                    for (_, metric_name), meter in corrs.items():
                        sum_writer.add_scalar('TrainEnc/{}'.format(metric_name), meter.val, global_step)
                    for (_, metric_name), meter in corrs_nc_norm.items():
                        sum_writer.add_scalar('TrainEnc/{}'.format(metric_name), meter.val, global_step)

                    if FLAGS.sum_level > 4:
                        if (global_step - 1) % 5 == 0:
                            fig = hist_comparison_fig(stack2numpy({'actual': actual_fmri_list, 'pred': pred_fmri_list}), linspace(-2.5, 2.5, 100))
                            sum_writer.add_figure('TrainEnc/BatchDist', fig, global_step)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if FLAGS.verbose > 0:  # plot progress
                bar.set_postfix_str(
                    '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | LossAvg: {lossavg:.4f} ({loss:.4f})'.format(
                        batch=batch_idx + 1,
                        size=len(loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        lossavg=losses['total'].avg,
                        loss=losses['total'].val,
                    ))
                bar.update()

    if FLAGS.pw_corr_win:
        collected_fmri = dict((k, torch.cat(v)) for k, v in collected_fmri.items())
    return losses['total'].avg, \
           [('Loss' + loss_name.capitalize(), meter.avg) for loss_name, meter in losses.items()] + \
           [(metric_name, meter.avg) for (_, metric_name), meter in corrs.items()] + \
           [(metric_name, meter.avg) for (_, metric_name), meter in corrs_nc_norm.items()], \
           voxel_pearson_r.avg, collected_fmri

if __name__ == '__main__':
    app.run(main)
