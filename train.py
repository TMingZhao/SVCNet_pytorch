import argparse
import os

import torch
from utils.logger import Logger
from utils.iofile import merge2dict, get_data_iterator
from model import PointDenoising
from tqdm import tqdm
from torch.autograd import Variable
from utils.iofile import get_file_list
from config import DENOISE, ABLATION
from utils.dataloader_punet import PairedPatchDataset, PointCloudDataset, standard_train_transforms, NormalizeUnitSphere
from utils.dataloader_EC import ECDataDataset


os.environ["CUDA_VISIBLE_DEVICES"] = "3"    # 该行代码必须在所有访问GPU代码之前


def evaluate(model, dataloader, device, visual=True, out_dir=""):
    model.eval()
    test_loss = []
    detail_loss = {}
    for batch_i, (input_points, patch_pos, clear_points, gt_points, mesh, whole_pos) in enumerate(tqdm(dataloader, desc="test")):
        input_points = Variable(input_points.to(device))
        clear_points = Variable(clear_points.to(device), requires_grad=False)

        with torch.no_grad():
            loss, log, coords = model(input_points, clear_points)[0:3]

        test_loss.append(loss.cpu())
        merge2dict(log, detail_loss)

        
    test_loss = torch.mean(torch.stack(test_loss))
    for key, value in detail_loss.items():
            detail_loss[key] = "%.6f"% (value / (batch_i+1))

    return test_loss, detail_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Point Upsampling')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size in training')
    parser.add_argument('--start_epoch', type=int,  default=0, help='start of epoch')
    parser.add_argument('--epochs', type=int, default=400, help='number of epoch in training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate in training')
    parser.add_argument('--adjust_lr_time', type=int, default=40, help='learning rate in training')
    parser.add_argument('--n_cpu', type=int, default=8, help='how many process')
    parser.add_argument('--fix_bone', type=bool, default=False, help='whether fix feature network weight')
    parser.add_argument('--pretrained_weights', type=str, default='')   # output/train_5/models/model_epoch_399.pth
    parser.add_argument("--out_dir", type=str, default="output", help="the out of program")
    parser.add_argument('--load_checkpoint', type=bool, default=True, help='whether change log file')
    parser.add_argument("--session", type=str, default="denoise_56", help="the out of folder")

    args = parser.parse_args()
  
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    out_dir = os.path.join(args.out_dir, args.session)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_model_dir = os.path.join(out_dir, 'models')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    logger_path = os.path.join(out_dir, "%s.log"%(args.session))
    logger = Logger(logger_path).get_logger()
    logger.info(" opt : {}".format(args))
    logger.info(" opt : denoise({})".format(str(DENOISE)))
    logger.info(" opt : Ablation({})".format(str(ABLATION)))
    logger.info(" %s: %s"%(args.session, DENOISE.log))


    if args.load_checkpoint:
        pths = get_file_list(save_model_dir, ".pth")
        if len(pths) != 0:

            pths = sorted(pths,key=lambda x: os.path.getmtime(x), reverse=False) 
            last_pth = pths[-1]
            last_epoch = last_pth.split("_")[-1][0:-4]
            args.pretrained_weights = last_pth
            if args.start_epoch == 0:   # 如果没有指定初始epoch则继续训练
                args.start_epoch = int(last_epoch) + 1
            logger.info(" load last checkpoint: train from epoch {}, load {}".format(args.start_epoch, args.pretrained_weights))
            
    if not torch.cuda.is_available():
        print("CUDA is not available, the current version does not support CPU")
        exit()

    device = torch.device("cuda")

    model = PointDenoising().to(device)
    
    # load model weights
    if args.pretrained_weights != '':
        model.load_state_dict(torch.load(args.pretrained_weights, map_location=device), strict=False)
        logger.info("load model: {}".format(args.pretrained_weights))
        

    dataset = ECDataDataset(patch_size=DENOISE.mini_point, sub_batch_size=4, transform=standard_train_transforms(noise_std_max=0.10, noise_std_min=0.02, rotate=True))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size//dataset.sub_batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset.collate_fn,)
    ec_iter = get_data_iterator(dataloader)

    dataset = PairedPatchDataset(
                    datasets=[
                        PointCloudDataset(
                            root="/home/rslab/ztm/DenoiseUpsampling/data",
                            dataset="PUNet",
                            split='train',
                            resolution=resl,
                            transform=standard_train_transforms(noise_std_max=0.020, noise_std_min=0.005, rotate=True)
                        ) for resl in ['5000_poisson', '10000_poisson', '30000_poisson', '50000_poisson']
                    ],
                    patch_size=DENOISE.mini_point,
                    patch_ratio=1,
                    num_patches=10,
                    mini_batch=args.batch_size,
                    on_the_fly=True,
                    transform= NormalizeUnitSphere()
                )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size//dataset.sub_batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset.collate_fn,)
    
    test_dataset = PairedPatchDataset(
                datasets=[
                    PointCloudDataset(
                        root="/home/rslab/ztm/DenoiseUpsampling/data",
                        dataset="PUNet",
                        split='test',
                        resolution=resl,
                        transform=standard_train_transforms(noise_std_max=0.020, noise_std_min=0.005, rotate=True)
                    ) for resl in ['5000_poisson', '10000_poisson', '30000_poisson', '50000_poisson']
                ],
                patch_size=DENOISE.mini_point,
                patch_ratio=1,
                num_patches=1,
                mini_batch=args.batch_size,
                on_the_fly=True,
                transform= NormalizeUnitSphere()
            )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True,collate_fn=test_dataset.collate_fn,) 



    # optimizer
    if args.fix_bone:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)  # 0.0005

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        train_loss = []
        detail_loss = {}
        lr = args.learning_rate * (0.4 ** ((epoch) // args.adjust_lr_time))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if (epoch) % args.adjust_lr_time == 0:
            logger.info(" 调整学习率%f"%(lr))

        data_len = dataset.__len__()
        for batch_i, batch_data in enumerate(tqdm(dataloader, desc="train")):
            input_points, patch_pos, clear_points, gt_points, mesh, edge = batch_data
            
            input_points = Variable(input_points.to(device))
            clear_points = Variable(clear_points.to(device), requires_grad=False)
            gt_points = Variable(gt_points.to(device), requires_grad=False)


            loss, loss_log = model(input_points, clear_points, edge)[0:2]
            loss.backward()

            optimizer.step()
            optimizer.zero_grad() 

            train_loss.append(loss.cpu())

            if ABLATION.edge_constraint:
                # 采用ec_data数据集同时进行训练
                batch_data = next(ec_iter)
                input_points, patch_pos, clear_points, gt_points, mesh, edge = batch_data
                
                input_points = Variable(input_points.to(device))
                clear_points = Variable(clear_points.to(device), requires_grad=False)
                edge = Variable(edge.to(device), requires_grad=False)

                loss, loss_log = model(input_points, clear_points, edge)[0:2]
                loss.backward()

                optimizer.step()
                optimizer.zero_grad() 


            merge2dict(loss_log, detail_loss)

        train_loss = torch.mean(torch.stack(train_loss))
        for key, value in detail_loss.items():
            detail_loss[key] = "%.6f"% (detail_loss[key] / (batch_i+1))

        if epoch % 5 == 0:
            model_path = os.path.join(save_model_dir, 'model_epoch_{}.pth'.format(epoch))
            torch.save(model.state_dict(), model_path)
            # print("model saved : ", model_path)
        
        test_loss, test_log = evaluate(model, test_dataloader, device, False)
        # test_loss, test_log = evaluate_pugan(model, args.batch_size, device)
        logger.info("==> EPOCH [{}], train_loss: {}, detail_loss: {}, test_loss: {}".format(epoch, train_loss, detail_loss, test_log))

        if epoch % 5 == 0:
            print("model saved : ", model_path)
