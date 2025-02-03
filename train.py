import argparse
from time import time
import torch
from torch import nn
import torch.nn.functional as F

import utils
import augmentations
from self_optimal_transport import SOT

def get_args():
    """ Description: Parses arguments at command line. """
    parser = argparse.ArgumentParser()

    # global args
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='miniimagenet', choices=['miniimagenet', 'cifar', 'isic2018', 'breakhis', 'papsmear', 'pathmnist', 'organamnist', 'dermamnist'])
    parser.add_argument('--data_path', type=str, default='./datasets/few_shot/miniimagenet')
    parser.add_argument('--method', type=str, default='pt_map_sot',
                        choices=['pt_map', 'pt_map_sot', 'proto', 'proto_sot'],
                        help="Specify the few shot method to use. ")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval', type=utils.bool_flag, default=False,
                        help=""" Set to true if you want to evaluate trained model on test set. """)
    parser.add_argument('--eval_freq', type=int, default=1,
                        help=""" Evaluate training every n epochs. """)
    parser.add_argument('--eval_first', type=utils.bool_flag, default=False,
                        help=""" Set to true to evaluate the model before training. Useful for fine-tuning. """)

    # wandb args
    parser.add_argument('--wandb', type=utils.bool_flag, default=False, help=""" Log data into wandb. """)
    parser.add_argument('--project', type=str, default='', help=""" Project name in wandb. """)
    parser.add_argument('--entity', type=str, default='', help=""" Your wandb entity name. """)
    parser.add_argument('--log_step', type=utils.bool_flag, default=False, help=""" Log training steps. """)
    parser.add_argument('--log_epoch', type=utils.bool_flag, default=True, help=""" Log epoch. """)

    # few-shot args
    parser.add_argument('--train_way', type=int, default=5, help=""" Number of classes in training batches. """)
    parser.add_argument('--val_way', type=int, default=5, help=""" Number of classes in validation/testing batches. """)
    parser.add_argument('--num_shot', type=int, default=5, help=""" Support size. """)
    parser.add_argument('--num_query', type=int, default=15, help=""" Query size. """)
    parser.add_argument('--train_episodes', type=int, default=200, help=""" Number of episodes for each epoch. """)
    parser.add_argument('--eval_episodes', type=int, default=400, help=""" Number of tasks to evaluate. """)
    parser.add_argument('--test_episodes', type=int, default=400, help=""" Number of tasks to evaluate. """)

    # training args
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--augment', type=utils.bool_flag, default=True, help=""" Apply data augmentation. """)
    parser.add_argument('--early_stopping', type=utils.bool_flag, default=False)
    parser.add_argument('--tolerance', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=64)

    # model args
    parser.add_argument('--backbone', type=str, default='WRN', choices=list(utils.models.keys()))
    parser.add_argument('--pretrained_path', type=str, default=False,
                        help=""" Path to pretrained model. For testing/fine-tuning. """)
    parser.add_argument('--temperature', type=float, default=0.1, help=""" Temperature for ProtoNet. """)
    parser.add_argument('--dropout', type=float, default=0., help=""" Dropout probability. """)

    # sla
    parser.add_argument('--aug', type=str, default=None)
    parser.add_argument('--with_large_loss', action='store_true')
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--sla_reg', type=float, default=0.1)
    
    # distillation
    parser.add_argument('--knowledge_distillation', type=utils.bool_flag, default=False)
    parser.add_argument('--teacher_path', type=str, help="""Path to teacher model to distill student model.""")
    parser.add_argument('--distillation_type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation_alpha', default=0.1, type=float, help="")
    parser.add_argument('--distillation_tau', default=1.0, type=float, help="")


    # SOT args
    parser.add_argument('--ot_reg', type=float, default=0.1,
                        help=""" Entropy regularization. For few-shot methods, 0.1-0.2 works best. """)
    parser.add_argument('--sink_iters', type=int, default=20,
                        help=""" Number of Sinkhorn iterations. """)
    parser.add_argument('--distance_metric', type=str, default='cosine',
                        help=""" Build the cost matrix. """, choices=['cosine', 'euclidean'])
    parser.add_argument('--mask_diag', type=utils.bool_flag, default=True,
                        help=""" If true, apply mask diagonal values before and after the OT. """)
    parser.add_argument('--max_scale', type=utils.bool_flag, default=True,
                        help=""" Scaling range of the SOT values to [0,1]. """)
    return parser.parse_args()


def main():
    args = get_args()

    utils.set_seed(seed=args.seed)
    out_dir = utils.get_output_dir(args=args)

    # define datasets and loaders
    args.set_episodes = dict(train=args.train_episodes, val=args.eval_episodes, test=args.test_episodes)
    if not args.eval:
        train_loader = utils.get_dataloader(set_name='train', args=args, constant=False)
        val_loader = utils.get_dataloader(set_name='val', args=args, constant=True)
    else:
        train_loader = None
        try:
            val_loader = utils.get_dataloader(set_name='test', args=args, constant=False)
        except FileNotFoundError:
            val_loader = utils.get_dataloader(set_name='val', args=args, constant=False)

    ### SLA Transformation
    if args.aug is not None:
        transform, m = augmentations.__dict__[args.aug]()
    else:
        m = 0
        
    # define model and load pretrained weights if available
    model = utils.get_model(args.backbone, args, m)
    model = model.cuda()
    utils.load_weights(model, args)

    # define teacher model for knowledge distillation

    # set criterion for backbone and few-shot method
    method_criterion = utils.get_criterion_by_method(method=args.method)
    backbone_criterion = utils.get_criterion_by_backbone(backbone=args.backbone)

    # define optimizer and scheduler
    optimizer = utils.get_optimizer(args=args, params=model.parameters())
    scheduler = utils.get_scheduler(args=args, optimizer=optimizer)

    # load optimizer and scheduler if available (for training continuation)
    utils.load_criterion_optimizer(method_criterion, optimizer, args)

    # SOT and few-shot classification method (e.g. pt-map...)
    sot = None
    if 'sot' in args.method.lower():
        sot = SOT(distance_metric=args.distance_metric, ot_reg=args.ot_reg, mask_diag=args.mask_diag,
                  sinkhorn_iterations=args.sink_iters, max_scale=args.max_scale)

    method = utils.get_method(args=args, sot=sot)

    # few-shot labels
    train_labels = utils.get_fs_labels(args.method, args.train_way, args.num_query, args.num_shot)
    val_labels = utils.get_fs_labels(args.method, args.val_way, args.num_query, args.num_shot)

    # set logger and criterion
    # criterion = utils.get_criterion_by_method(method=args.method)
    # logger = utils.get_logger(exp_name=out_dir.split('/')[-1], args=args)
    

    # only evaluate
    if args.eval:
        print(f"Evaluate model for {args.test_episodes} episodes... ")
        eval_one_epoch(model, val_loader, method, method_criterion, val_labels, 0, args, set_name='test')
        exit(1)

    # evaluate model before training
    if args.eval_first:
        print("Evaluate model before training... ")
        eval_one_epoch(model, val_loader, method, method_criterion, val_labels, -1, args, set_name='val')

    # initialized wandb
    if args.wandb:
        utils.init_wandb(exp_name=out_dir.split('/')[-1], args=args)

    # main loop
    print("Start training...")
    best_loss = 1000
    best_acc = 0
    epochs_no_improve = 0
    for epoch in range(1, args.max_epochs + 1):
        print("[Epoch {}/{}]...".format(epoch, args.max_epochs))

        # train
        train_one_epoch(model, train_loader, optimizer, method, method_criterion, backbone_criterion, train_labels, epoch, transform, args)
        if scheduler is not None:
            scheduler.step()

        # eval
        if epoch % args.eval_freq == 0:
            eval_loss, eval_acc = eval_one_epoch(model, val_loader, method, method_criterion, val_labels, epoch, args, set_name='val')

        # save best model
        if eval_loss < best_loss:
            best_loss = eval_loss
            epochs_no_improve = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'criterion_state_dict': method_criterion.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        f"{out_dir}/{epoch}_min_loss_{eval_acc}.pt")
        elif eval_acc > best_acc:
            best_acc = eval_acc
            epochs_no_improve = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'criterion_state_dict': method_criterion.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        f"{out_dir}/{epoch}_max_acc_{eval_acc}.pt")
        else:
            epochs_no_improve += 1
        
        if args.early_stopping == True and epoch > epochs_no_improve and epochs_no_improve == args.tolerance:
            print(f"Early stopping at epoch {epoch}. Best val/loss: {best_loss}, Best val/acc: {best_acc}")
            break

        torch.save({'model_state_dict': model.state_dict(),
                    'criterion_state_dict': method_criterion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    f"{out_dir}/last.pt")


def train_one_epoch(model, loader, optimizer, method, method_criterion, backbone_criterion, labels, epoch, transform, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Train Epoch: [{}/{}]'.format(epoch, args.max_epochs)
    log_freq = 50
    n_batches = len(loader)

    model.train()
    # results = {'train/accuracy': 0, 'train/loss': 0}
    # start = time()
    for batch_idx, (input, target) in enumerate(metric_logger.log_every(loader, log_freq, header=header)):
        target = target.type(torch.LongTensor) 
        images, target = input.cuda(), target.cuda()

        # few-shot method
        features = model(images)
        probas, accuracy = method(features, labels=labels, mode='train')
        q_labels = labels if len(labels) == len(probas) else labels[-len(probas):]

        # sla
        batch_size = images.shape[0]
        transformed_images = transform(model, images, target)
        n = transformed_images.shape[0] // batch_size

        feats, preds = model(transformed_images, return_logit=True)
        label_sla = torch.stack([target*n+i for i in range(n)], 1).view(-1)
        loss_sla = backbone_criterion(preds, label_sla) 

        if args.with_large_loss:
            loss_sla = loss_sla * n

        # sla + kd 
        """
        joint_preds, single_preds = model(transformed_images, None)
        single_preds = single_preds[::n]
        joint_labels = torch.stack([target*n+i for i in range(n)], 1).view(-1)

        joint_loss = F.cross_entropy(joint_preds, joint_labels)
        single_loss = F.cross_entropy(single_preds, target)

        if args.with_large_loss:
            joint_loss = joint_loss * n
        agg_preds = 0
        for i in range(n):
            agg_preds = agg_preds + joint_preds[i::n, i::n] / n

        loss_distillation = F.kl_div(F.log_softmax(single_preds / args.T, 1),
                                     F.softmax(agg_preds.detach() / args.T, 1),
                                     reduction="batchmean")
        loss_distillation = loss_distillation.mul(args.T**2)
        """

        # loss (cl + sla)
        loss_cl = method_criterion(probas, q_labels)
        loss = loss_cl + (loss_sla * args.sla_reg)
        
        # loss 
        # loss = method_criterion(probas, q_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.detach().item(),
                             loss_cl = loss_cl.detach().item(),
                             loss_sla = loss_sla.detach().item(),
                            #  loss_distillation = loss_distillation,
                             accuracy=accuracy)

        if batch_idx % log_freq == 0:
            utils.wandb_log(
                {
                    'train/loss_step': loss.item(),
                    'train/loss_cl' : loss_cl.item(),
                    'train/loss_sla': loss_sla.item(),
                    # 'train/joint_loss': joint_loss,
                    # 'train/single_loss' : single_loss,
                    # 'train/loss_distillation' : loss_distillation,
                    'train/step': batch_idx + (epoch * n_batches),
                    'train/accuracy_step': accuracy
                }
            )
        
    print("Averaged stats:", metric_logger)
    utils.wandb_log(
        {
            'lr': optimizer.param_groups[0]['lr'],
            'train/epoch': epoch,
            'train/loss': metric_logger.loss.global_avg,
            'train/loss_cl': metric_logger.loss_cl.global_avg,
            'train/loss_sla': metric_logger.loss_sla.global_avg,
            # 'train/joint_loss': metric_logger.joint_loss.global_avg,
            # 'train/single_loss' : metric_logger.single_loss.global_avg,
            # 'train/loss_distillation' : metric_logger.loss_distillation.global_avg,
            'train/accuracy': metric_logger.accuracy.global_avg,
        }
    )
    return metric_logger

        # results["train/loss"] += loss.item()
        # results["train/accuracy"] += accuracy

        # if log_step and (batch_idx + 1) % 50 == 0:
        #     step = batch_idx+((epoch-1) * len(loader))
        #     utils.log_step(
        #         results={'train/loss_step': loss.item(), 'train/accuracy_step': accuracy, 'train/train_step': step},
        #         logger=logger
        #     )

    # results["train/time"] = time() - start
    # results["train/epoch"] = epoch
    # utils.print_and_log(results=results, n=len(loader), logger=logger)
    # return results


@torch.no_grad()
def eval_one_epoch(model, loader, method, method_criterion, labels, epoch, args, set_name='val'):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:' if set_name == "val" else 'Test:'
    log_freq = 50

    n_batches = len(loader)

    model.eval()
    acc_list = []

    for batch_idx, batch in enumerate(metric_logger.log_every(loader, log_freq, header=header)):
        images = batch[0].cuda()

        features = model(images)

        # apply few_shot method
        probas, accuracy = method(X=features, labels=labels, mode='val')
        q_labels = labels if len(labels) == len(probas) else labels[-len(probas):]

        loss = method_criterion(probas, q_labels)

        acc_list.append(accuracy*100)

        metric_logger.update(loss=loss.detach().item(),
                             accuracy=accuracy)

    a, b = utils.compute_confidence_interval(acc_list)
    print("{}-way {}-shot accuracy with 95% interval : {:.2f}±{:.2f}".format(args.val_way, args.num_shot, a, b))
    print("Averaged stats:", metric_logger)
    utils.wandb_log(
        {
            '{}/epoch'.format(set_name): epoch,
            '{}/loss'.format(set_name): metric_logger.loss.global_avg,
            '{}/accuracy'.format(set_name): metric_logger.accuracy.global_avg,
        }
    )

    return metric_logger.loss.global_avg, metric_logger.accuracy.global_avg

        

    #     if batch_idx % 50 == 0:
    #         step = batch_idx+((epoch-1) * len(loader))
    #         print(f"Batch {batch_idx + 1}/{len(loader)}: ")
    #         utils.log_step(
    #             results={f'{set_name}/loss_step': loss.item(), f'{set_name}/accuracy_step': accuracy,
    #                      f'{set_name}/{set_name}_step': step},
    #             logger=logger
    #         )

    # results["val/epoch"] = epoch
    # utils.print_and_log(results=results, n=len(loader), logger=logger)
    # return results


if __name__ == '__main__':
    main()
