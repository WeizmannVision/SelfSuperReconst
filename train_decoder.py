"""
    Train fMRI to image decoder by supervised learning and (optionally) unsupervised objectives.
    Date created: 8/28/19
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
from torchvision.utils import make_grid, save_image
import pretrainedmodels as pm
import pretrainedmodels.utils as pmutils
from utils import *
from config_dec import *
from absl import app
from utils.misc import set_gpu

def main(argv):
    del argv
    cprint1(starred(FLAGS.exp_prefix))
    cprint1('== Summary level: {} =='.format(FLAGS.sum_level))
    set_gpu()

    # Data
    fmri_xfm = np.float32

    get_dataset = lambda subset_case: \
        KamitaniDataset(FLAGS.roi, sbj_num=FLAGS.sbj_num, im_res=FLAGS.im_res, fmri_xfm=fmri_xfm, subset_case=subset_case, select_voxels=FLAGS.select_voxels, is_rgbd=FLAGS.is_rgbd)

    train_labeled = get_dataset(KamitaniDataset.TRAIN)
    n_voxels = train_labeled.n_voxels
    val_labeled_avg = KamitaniDataset(FLAGS.roi, sbj_num=FLAGS.sbj_num, fmri_xfm=fmri_xfm, subset_case=KamitaniDataset.TEST_AVG, select_voxels=FLAGS.select_voxels, is_rgbd=FLAGS.is_rgbd)
    val_labeled = KamitaniDataset(FLAGS.roi, sbj_num=FLAGS.sbj_num, fmri_xfm=fmri_xfm, subset_case=KamitaniDataset.TEST, test_avg_qntl=FLAGS.test_avg_qntl, select_voxels=FLAGS.select_voxels, is_rgbd=FLAGS.is_rgbd)

    train_unlabeled_fmri = UnlabeledDataset(val_labeled, return_tup_index=-1)

    if FLAGS.is_rgbd:
        external_images = RGBD_Dataset(depth_only=FLAGS.is_rgbd==2)
    else:
        external_images = ImageFolder('data/imagenet/val')

    #########################################

    img_xfm_train = transforms.Compose([
        transforms.Resize(size=im_res(), interpolation=Image.BILINEAR),
        transforms.RandomCrop(size=im_res(), padding=int(FLAGS.random_crop_pad_percent / 100 * im_res()), padding_mode='edge'),
        transforms.ToTensor(),
    ])
    img_xfm_basic = transforms.Compose([transforms.Resize(size=im_res(), interpolation=Image.BILINEAR), transforms.CenterCrop(im_res()), transforms.ToTensor()])
    global img_xfm_norm

    if FLAGS.is_rgbd:
        if FLAGS.is_rgbd == 1:  # RGBD
            if FLAGS.norm_within_img:
                img_xfm_norm = norm_imagenet_norm_depth_img_batch
            else:
                img_xfm_norm = NormalizeBatch([0.485, 0.456, 0.406, 0.461], [0.229, 0.224, 0.225, 0.305])
        else:  # Depth only
            if FLAGS.norm_within_img:
                img_xfm_norm = norm_batch_within_img
            else:
                img_xfm_norm = NormalizeBatch([0.461], [0.305])

    else:
        img_xfm_norm = norm_batch_imagenet

    img_xfm_norm_inv = identity

    #########################################

    global voxel_nc
    voxel_nc = val_labeled.get_voxel_score('noise_ceil')
    voxel_indices_nc_sort = np.argsort(voxel_nc)[::-1]
    voxel_snr = val_labeled.get_voxel_score('snr')
    voxel_snr_scaled = torch.tensor(voxel_snr / voxel_snr.mean()).cuda()
    voxel_snr_inv = 1 / voxel_snr
    voxel_snr_inv_scaled = torch.tensor(voxel_snr_inv / voxel_snr_inv.mean()).cuda()

    cprintm(u'(*) n_voxels: {} | noise ceiling {:.2f} \u00B1 {:.2f} (Mean \u00B1 SD)'.format(n_voxels, voxel_nc.mean(), voxel_nc.std()))

    if FLAGS.separable:
        cprint1('(*) Separable Encoder')
        enc = make_model('SeparableEncoderVGG19ml', n_voxels, FLAGS.random_crop_pad_percent, drop_rate=0.25)
    else:
        # N.B. not yet supports RGBD
        enc = make_model('BaseEncoderVGG19ml', n_voxels, FLAGS.random_crop_pad_percent, drop_rate=0.25)

    if FLAGS.is_rgbd and not FLAGS.midas_dec:
        if FLAGS.is_rgbd == 1 and not FLAGS.depth_dec:  # RGBD
            n_chan_dec_output = 4
        else:  # Depth only
            n_chan_dec_output = 1
    else:
        n_chan_dec_output = 3

    if FLAGS.midas_dec:
        depth_extractor_dec = DepthExtractor(img_xfm_norm=norm_batch_imagenet)
    else:
        depth_extractor_dec = None

    dec = make_model('BaseDecoder', n_voxels, im_res(), start_CHW=(64, 14, 14), n_conv_layers_ramp=FLAGS.n_conv_layers, n_chan=64,
        n_chan_output=n_chan_dec_output, depth_extractor=depth_extractor_dec)

    train_labeled = CustomDataset(train_labeled, input_xfm=img_xfm_basic)
    val_labeled_avg = CustomDataset(val_labeled_avg, input_xfm=img_xfm_basic)
    val_labeled = CustomDataset(val_labeled, input_xfm=img_xfm_basic)
    external_images = CustomDataset(external_images, input_xfm=img_xfm_basic)

    img_loss_xfm_norm = img_xfm_norm
    if FLAGS.rgbd_loss:
        if FLAGS.norm_within_img:
            img_loss_xfm_norm = norm_imagenet_norm_depth_img_batch
        else:
            img_loss_xfm_norm = NormalizeBatch([0.485, 0.456, 0.406, 0.461], [0.229, 0.224, 0.225, 0.305])

        data_norm_factors_images = UnlabeledDataset(CustomDataset(KamitaniDataset(FLAGS.roi, im_res=FLAGS.im_res, subset_case=KamitaniDataset.TRAIN, is_rgbd=1), input_xfm=img_xfm_basic), 0)
    elif FLAGS.depth_dec:
        if FLAGS.norm_within_img:
            img_loss_xfm_norm = norm_batch_within_img
        else:
            img_loss_xfm_norm = NormalizeBatch([0.461], [0.305])
        data_norm_factors_images = UnlabeledDataset(CustomDataset(KamitaniDataset(FLAGS.roi, im_res=FLAGS.im_res, subset_case=KamitaniDataset.TRAIN, is_rgbd=2), input_xfm=img_xfm_basic), 0)

    else:
        data_norm_factors_images = UnlabeledDataset(train_labeled, 0)

    criterions_dict = {
        'image': ImageLoss(feats_extractor=None, img_xfm_norm=img_loss_xfm_norm, data_norm_factors_images=data_norm_factors_images).cuda(),
        'fmri': lambda pred, actual: 1. * (voxel_snr_scaled.view(1, -1) * (pred - actual).abs()).mean() + 0.2 * cosine_loss(pred, actual),
    }
    main_branch = ['_features.{}'.format(i) for i in range(30)]
    feats_extractor = MultiBranch(pm.vgg16(), {'conv5-3': main_branch[-1]}, main_branch, spatial_out_dims=None)

    if FLAGS.depth_from_rgb:
        if FLAGS.midas_loss:
            depth_extractor = criterions_dict['image'].depth_extractor
        else:
            depth_extractor = DepthExtractor(img_xfm_norm=img_xfm_norm).cuda().eval()
    else:
        depth_extractor = None

    # Optimizer/Scheduler
    cprint1('(*) Training Decoder parameters only.')
    global trainable_params
    trainable_params = list(itertools.chain(*[m.parameters() for m in dec.trainable]))
    print('    Total params for training: %s | %s' % (param_count_str(trainable_params), param_count_str(dec.parameters())))

    if FLAGS.decay > 0:
        cprintm('(+) {} weight decay.'.format(FLAGS.decay))
    optimizer = optim.Adam(trainable_params, lr=FLAGS.learning_rate, amsgrad=True)

    scheduler = None
    if FLAGS.scheduler > 0:
        if FLAGS.scheduler == 1:
            cprintm('(+) Using scheduler: On-Plateau.')
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, cooldown=1)
        elif FLAGS.scheduler == 12345:  # Milestones
            # milestones = [20, 35, 45, 50]
            milestones = [int(x) for x in FLAGS.mslr]
            cprintm('(+) Using scheduler: {} by milestones {} epochs.'.format(FLAGS.sched_gamma, ', '.join(FLAGS.mslr)))
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=FLAGS.sched_gamma)
        else:
            cprintm('(+) Using scheduler: {} every {} epochs.'.format(FLAGS.sched_gamma, FLAGS.scheduler))
            scheduler = optim.lr_scheduler.StepLR(optimizer, FLAGS.scheduler, gamma=FLAGS.sched_gamma)

    enc.cuda()
    dec.cuda()

    dec = nn.DataParallel(dec)
    cudnn.benchmark = True

    if FLAGS.config_train > 1:
        enc = nn.DataParallel(enc)
        # Load pretrained encoder
        assert os.path.isfile(enc_cpt_path())
        print('\t==> Loading checkpoint {}'.format(os.path.basename(enc_cpt_path())))
        enc.load_state_dict(torch.load(enc_cpt_path())['state_dict'])

    data_loaders_labeled = {
        'train': data.DataLoader(train_labeled, batch_size=min(batch_size_list()[0], len(train_labeled)), shuffle=True, num_workers=num_workers(), pin_memory=True),
        'test': data.DataLoader(val_labeled_avg, batch_size=min(batch_size_list()[-1], len(val_labeled_avg)), shuffle=False, num_workers=num_workers(), pin_memory=True),
    }

    # Loss
    # Regularization
    reg_loss_dict = {}
    tau = 800.
    m = 1/8000.

    W = list(dec.module.input_fc.parameters())[0].T
    dd = int(np.sqrt(W.view(len(W), dec.module.start_CHW[0], -1).shape[-1]))
    xx, yy = torch.tensor(np.expand_dims(np.mgrid[:dd, :dd], axis=1).repeat(len(W), axis=1), dtype=torch.float32, requires_grad=False).cuda()
    def group_reg(reg_type='fcmom2'):
        W = list(dec.module.input_fc.parameters())[0].T
        W = W.view(len(W), *dec.module.start_CHW)  # VxCxHxW
        if reg_type=='fcmom2':
            W = (W**2).sum(axis=1)
            m00 = W.view(len(W),-1).sum(1)
            m10 = (xx * W).view(len(W), -1).sum(1)
            m01 = (yy * W).view(len(W), -1).sum(1)
            m02 = ((yy - (m01/m00).view(-1,1,1))**2 * W).view(len(W), -1).sum(1)
            m20 = ((xx - (m10/m00).view(-1,1,1))**2 * W).view(len(W), -1).sum(1)
            return (m02 + m20).sum() / (2 * dd * len(W))
        elif reg_type=='gl':
            Wsq_pad = F.pad(W**2, [1, 1, 1, 1], mode='reflect')
            n_reg = .5

            reg_l1 = FLAGS.gl_l1; reg_gl = FLAGS.gl_gl
            Wn = (W**2 + n_reg / 8 * (Wsq_pad[..., :-2, 1:-1] + Wsq_pad[...,2:, 1:-1] + Wsq_pad[...,1:-1, :-2] + Wsq_pad[..., 1:-1, 2:] +
                                      Wsq_pad[..., :-2, :-2] + Wsq_pad[...,2:, 2:] + Wsq_pad[...,2:, :-2] + Wsq_pad[..., :-2, 2:])) / (1 + n_reg)
            sparse_loss_val = (W.abs() * voxel_snr_inv_scaled[:, None, None, None]).mean()
            gl_loss_val = Wn.mean(axis=1).sqrt().mean()
            reg_loss = reg_l1 * sparse_loss_val + reg_gl * gl_loss_val
            return reg_loss
        else:
            raise NotImplementedError

    eps = 0.01
    reg_loss_dict = {
        'L1reg_convs': (FLAGS.l1_convs, lambda : sum([param.abs().sum() for param in chained([m.parameters() for m in dec.modules() if isinstance(m, nn.Conv2d)]) if param.ndim == 4])),
        'mom2': (FLAGS.fc_mom2, lambda : group_reg('fcmom2')),
        'gl': (FLAGS.fc_gl, lambda : group_reg('gl')),
    }

    for reg_loss_name, (w, _) in reg_loss_dict.items():
        if callable(w) or w > 0:
            cprintm('(+) {} {} loss.'.format(w, reg_loss_name))

    # Training
    print('\n' + '#'*100 + '\n')
    cprintm('\t** Training config {} **'.format(FLAGS.config_train))
    with SummaryWriter(comment='Decoder training') if FLAGS.sum_level > 0 else dummy_context_mgr() as sum_writer:
        global global_step
        global_step = 0
        best_loss = np.inf
        with tqdm(desc='Epochs', total=FLAGS.n_epochs) if FLAGS.verbose < 1 else dummy_context_mgr() as pbar:
            if FLAGS.verbose < 1:
                pbar.update(0)
            for epoch in range(FLAGS.n_epochs):
                if FLAGS.verbose > 0:
                    print('\nEpoch: [%d | %d]' % (epoch + 1, FLAGS.n_epochs))
                train_loss, _, collected_images = \
                    train_test(data_loaders_labeled['train'], dec, criterions_dict, train_unlabeled_fmri, external_images, enc, optimizer, reg_loss_dict, sum_writer=sum_writer)
                if sum_writer:
                    if epoch % 50 == 0 or (epoch % 2 == 0 and epoch in range(10)):
                        domain = np.linspace(0, 1, 100)

                        sum_writer.add_figure('TrainDec/ImagesOutDist', hist_comparison_fig(stack2numpy(collected_images), domain), epoch)

                        create_reconst_summary(enc, dec, data_loaders_labeled, train_labeled, epoch, img_xfm_norm_inv, sum_writer, depth_extractor)

                    if epoch in [5, FLAGS.n_epochs - 1] or epoch % 50 == 0:
                        if isinstance(dec.module.input_fc, nn.Linear):
                            rf_params, n_out_planes = list(dec.module.input_fc.parameters()), dec.module.start_CHW[0]
                        else:  # FWRF
                            rf_params, n_out_planes = dec.module.input_fc.rf, 1
                        sum_writer.add_figure(
                                'TrainDec/EpochVoxRF', vox_rf(rf_params, n_out_planes, voxel_indices_nc_sort, dec_fc=True, mask=not isinstance(dec.module.input_fc, nn.Linear)), epoch)

                    test_loss, meters_test, test_collected_images = \
                        train_test(data_loaders_labeled['test'], dec, criterions_dict, reg_loss_dict=reg_loss_dict, sum_writer=sum_writer)

                    for metric_name, meter_avg in meters_test:
                        if any(s in metric_name for s in ['LossD', 'LossCriteria', 'LossTotal']):
                            sum_writer.add_scalar('ValDec/{}'.format(metric_name), meter_avg, epoch)

                test_loss = train_loss

                if scheduler:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(test_loss)
                    else:
                        scheduler.step()

                # save model
                is_best = test_loss < best_loss
                best_loss = min(test_loss, best_loss)
                if FLAGS.may_save:
                    if is_best or not FLAGS.savebestonly:
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': dec.state_dict(),
                            'best_loss': best_loss,
                            'optimizer': optimizer.state_dict(),
                        }, is_best, filepath=get_checkpoint_out())
                if FLAGS.verbose < 1:
                    pbar.update()
        # Report
        image_D_sbs_gt_train, image_D_sbs_gt_test = create_reconst_summary(enc, dec, data_loaders_labeled, train_labeled, FLAGS.n_epochs, img_xfm_norm_inv, sum_writer, depth_extractor)
    cprintm('\n' + '*'*20 + ' TRAINING COMPLETE ' + '*'*20 + '\n')
    cprintc('(+) Saving reconstructed images.')
    out_folder = overridefolder(pjoin(exp_folder(), 'train'))
    for index, image_tensor in enumerate(image_D_sbs_gt_train):
        if len(image_tensor) >= 3:
            save_image(image_tensor[:3], pjoin(out_folder, str(index) + '.png'))
        if len(image_tensor) == 4 or len(image_tensor) == 1:
            if len(image_tensor) == 4: # RGBD
                image_tensor_depth = image_tensor[3]
            else:  # Depth only
                image_tensor_depth = image_tensor[0]
            save_image(image_tensor_depth, pjoin(out_folder, str(index) + '_gray_depth.png'))
            save_image(gray2color(image_tensor_depth), pjoin(out_folder, str(index) + '_depth.png'))

    out_folder = overridefolder(pjoin(exp_folder(), 'test_avg'))
    for index, image_tensor in enumerate(image_D_sbs_gt_test):
        if len(image_tensor) >= 3:
            save_image(image_tensor[:3], pjoin(out_folder, str(index) + '.png'))
        if len(image_tensor) == 4 or len(image_tensor) == 1:
            if len(image_tensor) == 4: # RGBD
                image_tensor_depth = image_tensor[3]
            else:  # Depth only
                image_tensor_depth = image_tensor[0]
            save_image(image_tensor_depth, pjoin(out_folder, str(index) + '_gray_depth.png'))
            save_image(gray2color(image_tensor_depth), pjoin(out_folder, str(index) + '_depth.png'))

    cprint1(FLAGS.exp_prefix)
    print('\n' + '='*100 + '\n')
    with open("runs/{}.txt".format(FLAGS.exp_prefix), "w") as f:
        f.write(FLAGS.flags_into_string())

def sample_pixels(imgs):
    indices = np.linspace(0, im_res()-1, im_res()//4, dtype='int')
    return imgs[:5, 0, indices, indices].cpu().detach().flatten().squeeze()

def train_test(data_loader_labeled, dec, criterions_dict, train_unlabeled_fmri=None, external_images=None, enc=None, optimizer=None, reg_loss_dict={}, sum_writer=None):
    global global_step
    if optimizer:
        mode = 'Train'
        enc.eval();
        dec.train()
    else:
        mode = 'Val'
        if enc:
            enc.eval();
        dec.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_names = ['total'] + ['criteria', 'D', 'ED', 'DE', 'geom'] + list(reg_loss_dict.keys())
    losses = dict(zip(losses_names, [AverageMeter() for _ in range(len(losses_names))]))
    corrs = { metric_tup: AverageMeter() for metric_tup in [(np.median, 'R_median'),
                                                            (lambda x: np.percentile(x, 90), 'R_90'),
                                                            (lambda x: np.percentile(x, 75), 'R_75'),
                                                            ]}
    corrs_nc_norm = copy.deepcopy(corrs)
    corrs_nc_norm = { (metric_func, metric_name.replace('R', 'R_ncnorm')): meter for (metric_func, metric_name), meter in corrs_nc_norm.items() }
    voxel_pearson_r = AverageMeter()
    end = time.time()
    collected_images = {'actual': [], 'pred': []}
    if optimizer:
        train_loaders_unlabeled_it = \
            dict(zip(['fmri', 'images'], [iter(data.DataLoader(dset, batch_size=min(batch_size_list()[i+1], len(dset)),
                                                            shuffle=True, num_workers=num_workers(), pin_memory=False))  # pin_memory=True))
                                        for i, dset in enumerate([train_unlabeled_fmri, external_images])]))

    main_loader = data_loader_labeled
    if len(main_loader) < 6:
        k_batch_collect = 1
    else:
        k_batch_collect = int(len(main_loader) // 6)
    with tqdm(desc=mode, total=len(main_loader)) if FLAGS.verbose > 0 else dummy_context_mgr() as bar:
        for batch_idx, (images_gt, fmri_gt) in enumerate(main_loader):
            if optimizer:
                unlabeled_fmri = next(train_loaders_unlabeled_it['fmri'], None)
                if unlabeled_fmri is None:
                    train_loaders_unlabeled_it['fmri'] = iter(data.DataLoader(train_unlabeled_fmri, batch_size=min(batch_size_list()[1], len(train_unlabeled_fmri)),
                                                            shuffle=True, num_workers=num_workers(), pin_memory=False))
                    unlabeled_fmri = next(train_loaders_unlabeled_it['fmri'])

                unlabeled_images, _ = next(train_loaders_unlabeled_it['images'], None)
                if unlabeled_images is None:
                    train_loaders_unlabeled_it['images'] = iter(data.DataLoader(external_images, batch_size=min(batch_size_list()[2], len(external_images)),
                                                            shuffle=True, num_workers=num_workers(), pin_memory=False))
                    unlabeled_images, _ = next(train_loaders_unlabeled_it['images'])

            else:
                unlabeled_fmri = unlabeled_images = torch.zeros(0)

            # measure data loading time
            data_time.update(time.time() - end)
            if batch_idx % k_batch_collect == 0:
                if FLAGS.sum_level > 3:
                    if FLAGS.sum_level > 4 or not optimizer:
                        actual_images_list = list(sample_pixels(images_gt))
                        collected_images['actual'].extend(actual_images_list)

            images_gt, fmri_gt, unlabeled_fmri, unlabeled_images = \
                map(lambda x: x.cuda(), [images_gt, fmri_gt, unlabeled_fmri, unlabeled_images])

            with dummy_context_mgr() if optimizer else torch.no_grad():
                images_D = dec(fmri_gt)

            if optimizer:
                if FLAGS.config_train > 1:  # Use unsupervised objectives
                    with torch.no_grad():
                        fmri_E = enc(img_xfm_norm(unlabeled_images)).detach()
                    images_ED = dec(fmri_E)

                    if float(FLAGS.loss_weights[-1]):
                        fmri_DE = enc(img_xfm_norm(dec(unlabeled_fmri)))
                    else:
                        fmri_DE = unlabeled_fmri.clone().detach()

            loss_criteria = 0
            criterions_dict['image'].eval()
            loss_D_list = criterions_dict['image'](images_D, images_gt)
            loss_D = sum(tup2list(loss_D_list, 1))
            losses['D'].update(loss_D.data)
            loss_ED = loss_DE = 0
            if optimizer and FLAGS.config_train > 1:
                loss_ED_list = criterions_dict['image'](images_ED, unlabeled_images)
                loss_ED = sum(tup2list(loss_ED_list, 1))
                losses['ED'].update(loss_ED.data)

                loss_DE = criterions_dict['fmri'](fmri_DE, unlabeled_fmri)
                losses['DE'].update(loss_DE.data)

            loss_criteria += sum([float(w) * l for w, l in zip(FLAGS.loss_weights, [loss_D, loss_ED, loss_DE])])
            losses['criteria'].update(loss_criteria.data)

            reg_loss_tot = 0
            for loss_name, (w, reg_loss_func) in reg_loss_dict.items():
                reg_loss = reg_loss_func()
                losses[loss_name].update(reg_loss.data)
                if callable(w):
                    w = w(global_step)
                reg_loss_tot += w * reg_loss

            loss = loss_criteria + reg_loss_tot

            losses['total'].update(loss.data)

            if batch_idx % k_batch_collect == 0:
                if FLAGS.sum_level > 3:
                    if FLAGS.sum_level > 4 or not optimizer:
                        pred_images_list = list(sample_pixels(images_D))
                        collected_images['pred'].extend(pred_images_list)

            if optimizer:
                criterions_dict['image'].zero_grad()
                dec.zero_grad()
                enc.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                if sum_writer:
                    for loss_name, meter in losses.items():
                        sum_writer.add_scalar('TrainDec/Loss{}'.format(loss_name.capitalize()), meter.val, global_step)

                    if FLAGS.sum_level > 5:
                        for loss_name, loss_val in loss_D_list:
                            sum_writer.add_scalar('ImageLossD/{}'.format(loss_name.capitalize()), loss_val, global_step)

                        for loss_name, loss_val in loss_ED_list:
                            sum_writer.add_scalar('ImageLossED/{}'.format(loss_name.capitalize()), loss_val, global_step)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if FLAGS.verbose > 0:  # plot progress
                bar.set_postfix_str(
                    '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | LossAvg: {lossavg:.4f} ({loss:.4f})'.format(
                        batch=batch_idx + 1,
                        size=len(main_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        lossavg=losses['total'].avg,
                        loss=losses['total'].val,
                    ))
                bar.update()

    return losses['total'].avg, \
           [('Loss' + loss_name.capitalize(), meter.avg) for loss_name, meter in losses.items()] + \
           [('LossD_' + loss_name.capitalize(), loss_val) for loss_name, loss_val in loss_D_list], \
           collected_images

if __name__ == '__main__':
    app.run(main)

