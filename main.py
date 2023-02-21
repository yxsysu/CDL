from config import get_default_config
from trainers import *
import time
import torch.backends.cudnn as cudnn
import sys
cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str,
                        default='config/exp_1.yaml',
                        help='path to config file')
    parser.add_argument('--gpu-devices', type=str, default='0,1', )
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    cfg = get_default_config()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    set_random_seed(cfg.train.seed)

    log_name = 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    loader, gallery_loader, probe_loader = \
        get_reid_dataloaders(cfg.data.image_size,
                             cfg.data.crop_size, cfg.data.padding, cfg.train.batch_size,
                             color_jitter=cfg.aug.color_jitter,
                             random_erase=cfg.aug.random_erase,
                             sampler=cfg.data.sampler,
                             num_instance=cfg.data.num_instance)

    num_classes = loader.dataset.return_num_class()

    checkpoint = torch.load(cfg.train.load_weight)
    state_dict = checkpoint['net']
    pretrained_state_dict = dict()

    for k, v in state_dict.items():
        if 'module' in k:
            pretrained_state_dict[k[7:]] = v
        else:
            pretrained_state_dict[k] = v
    strict = True
    trainer = ReidTrainer(cfg, num_classes=num_classes,
                          pretrained_state_dict=pretrained_state_dict,
                          strict=strict)

    start_epoch = 0
    total_epoch = cfg.train.max_epoch

    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(start_epoch, total_epoch):
        loader.dataset.shuffle_images()
        print('shuffle images ')
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (total_epoch - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, total_epoch, need_time))

        trainer.IDPLoss.transfer_classifier()

        if epoch in cfg.loss.filter.update_epoch:
            trainer.update_IDP_threshold(cfg.loss.filter.step_size)

        meters_trn = trainer.train_epoch(loader, epoch, views=len(num_classes),
                                         lamda=cfg.loss.lamda, RKA=cfg.loss.RKA)
        print('  **Train**  ' + create_stat_string(meters_trn))

        if (epoch+1) % cfg.val.freq == 0:
            trainer.eval_performance(gallery_loader, probe_loader)
            if cfg.train.save:
                print('saving model')
                trainer.save_checkpoint()
        if (epoch+1) % cfg.val.freq == 0:
            trainer.save_checkpoint(name='model_{}'.format(epoch))
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    print('final test')
    trainer.eval_performance(gallery_loader, probe_loader)
    if cfg.train.save:
        print('saving model')
        trainer.save_checkpoint()


if __name__ == '__main__':
    main()

