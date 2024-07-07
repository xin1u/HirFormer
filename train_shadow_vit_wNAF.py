import time,torchvision,argparse,logging,sys,os,gc
import torch,random
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
from utils.UTILS1 import compute_psnr
from utils.UTILS import AverageMeters,print_args_parameters,Lion
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from datasets.datasets_pairs import my_dataset,my_dataset_eval,my_dataset_wTxt
from networks.NAFNet_arch import NAFNet
from networks.MaeVit_arch import MaskedAutoencoderViT
from networks.Split_images import split_image,process_split_image_with_model,merge,process_split_image_with_model_parallel
from networks.image_utils import splitimage, mergeimage
sys.path.append(os.getcwd())
# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:',device)

parser = argparse.ArgumentParser()
# net = MaskedAutoencoderViT(
# patch_size=args.vit_patch_size, embed_dim=args.vit_embed_dim, depth=args.vit_depth, num_heads=args.vit_num_heads,
# decoder_embed_dim=args.vit_decoder_embed_dim, decoder_depth=args.vit_decoder_depth, decoder_num_heads=args.vit_decoder_num_heads,
# mlp_ratio=args.vit_mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), img_size= args.vit_img_size)
parser.add_argument('--vit_patch_size', type=int, default= 8)    
parser.add_argument('--vit_embed_dim', type=int, default= 256)
parser.add_argument('--vit_depth', type=int, default= 6)
parser.add_argument('--vit_num_heads', type=int, default= 8)
parser.add_argument('--vit_decoder_embed_dim', type=int, default= 256)    
parser.add_argument('--vit_decoder_depth', type=int, default= 6)
parser.add_argument('--vit_decoder_num_heads', type=int, default= 8)
parser.add_argument('--vit_mlp_ratio', type=int, default= 4)
parser.add_argument('--vit_img_size', type=int, default= 352)
parser.add_argument('--vit_grid_type', type=str, default= '4x4')
# Flag_process_split_image_with_model_parallel
parser.add_argument('--Flag_process_split_image_with_model_parallel', type=bool, default= True)
# 多阶段复原
parser.add_argument('--Flag_multi_scale', type=bool, default= False)
# path setting
parser.add_argument('--experiment_name', type=str,default= "train_shadow_vit") # modify the experiments name-->modify all save path
parser.add_argument('--unified_path', type=str,default=  '/root/autodl-tmp/SR_1/')
#parser.add_argument('--model_save_dir', type=str, default= )#required=True
parser.add_argument('--training_path', type=str,default= '/root/autodl-tmp/')
parser.add_argument('--training_path_txt', nargs='*', default=['/root/autodl-tmp/ShadowDatasets/ntire24D.txt' , '/root/autodl-tmp/ShadowDatasets/ntire23D.txt', '/root/autodl-tmp/ShadowDatasets/ntire24_test_fake_vit.txt', '/root/autodl-tmp/ShadowDatasets/ntire24_shrem_gen.txt', '/root/autodl-tmp/ShadowDatasets/ntire23_valid_gen_remove0005.txt'])


parser.add_argument('--writer_dir', type=str, default= '/root/tf-logs/')

parser.add_argument('--eval_in_path', type=str,default= '/root/autodl-tmp/ShadowDatasets/NTIRE23_sr_val_inp_remove_0005png/')
parser.add_argument('--eval_gt_path', type=str,default= '/root/autodl-tmp/ShadowDatasets/ntire23_sr_valid_gt_remove_0005png/')
#training setting
parser.add_argument('--EPOCH', type=int, default= 600)
parser.add_argument('--T_period', type=int, default= 50)  # CosineAnnealingWarmRestarts
parser.add_argument('--BATCH_SIZE', type=int, default= 24)
parser.add_argument('--overlap_size', type=int, default= 0)
parser.add_argument('--Crop_patches', type=int, default= 1024)
parser.add_argument('--learning_rate', type=float, default= 0.0004)
parser.add_argument('--print_frequency', type=int, default= 50)
parser.add_argument('--SAVE_Inter_Results', type=bool, default= False)
#during training
parser.add_argument('--max_psnr', type=int, default= 40)
parser.add_argument('--fix_sampleA', type=int, default= 30000)

parser.add_argument('--debug', type=bool, default= False)

parser.add_argument('--Aug_regular', type=bool, default= False)
#training setting (arch)
parser.add_argument('--base_channel', type = int, default= 24)
parser.add_argument('--num_res', type=int, default= 6)
parser.add_argument('--img_channel', type=int, default= 3)
parser.add_argument('--enc_blks', nargs='+', type=int, default= [1, 1, 1, 28], help='List of integers')
parser.add_argument('--dec_blks', nargs='+', type=int, default= [1, 1, 1, 1], help='List of integers')

#loss
parser.add_argument('--base_loss', type=str, default= 'char')
parser.add_argument('--addition_loss', type=str, default= 'fft')
parser.add_argument('--addition_loss_coff', type=float, default= 0.02)
parser.add_argument('--weight_coff', type=float, default= 10.0)

# load load_pre_model
parser.add_argument('--load_pre_model', type=bool, default= False)
parser.add_argument('--pre_model', type=str, default= '/root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.2e-4_4x4_fine_tune/net_epoch_202_PSNR_23.21.pth')
parser.add_argument('--pre_model_0', type=str, default= '/root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.2e-4_4x4_fine_tune/net_epoch_202_PSNR_23.21.pth')
parser.add_argument('--pre_model_1', type=str, default= '')

#optim
parser.add_argument('--optim', type=str, default= 'adam')



args = parser.parse_args()
# print all args params!!!
print_args_parameters(args)

if args.debug ==True:
    fix_sampleA = 400
else:
    fix_sampleA = args.fix_sampleA


exper_name =args.experiment_name
writer = SummaryWriter(args.writer_dir + exper_name)
if not os.path.exists(args.writer_dir):
    #os.mkdir(args.writer_dir)
    os.makedirs(args.writer_dir, exist_ok=True)
    
unified_path = args.unified_path
SAVE_PATH =unified_path  + exper_name + '/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH,exist_ok=True)
if args.SAVE_Inter_Results:
    SAVE_Inter_Results_PATH = SAVE_PATH +'Inter_Temp_results/'
    if not os.path.exists(SAVE_Inter_Results_PATH):
        #os.mkdir(SAVE_Inter_Results_PATH)
        os.makedirs(SAVE_Inter_Results_PATH,exist_ok=True)

logging.basicConfig(filename=SAVE_PATH + args.experiment_name + '.log', level=logging.INFO)

logging.info('======================'*2 + 'args: parameters'+'======================'*2 )
for k in args.__dict__:
    logging.info(k + ": " + str(args.__dict__[k]))
logging.info('======================'*2 + 'args: parameters'+'======================'*2 )
    

trans_eval = transforms.Compose(
        [
         transforms.ToTensor()
        ])

logging.info(f'begin training! ')
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print("=="*50)#Fore.RED +
print( "Check val-RD pairs ???:",os.listdir(args.eval_in_path)==os.listdir(args.eval_gt_path),
      '//      len eval:',len(os.listdir(args.eval_gt_path)))
print("=="*50)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def test(net,net_1,eval_loader,epoch =1,max_psnr_val=26 ,Dname = 'S'):
    net.eval()
    net_1.eval()
    with torch.no_grad():

        #eval_results =
        Avg_Meters_evaling =AverageMeters()
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)

            B, C, H, W = inputs.shape
            if index ==0:
                print(f" val_input.size: {data_in.size()}, gt.size: {label.size()}")
                logging.info(f" val_input.size: {data_in.size()}, gt.size: {label.size()}")

            # overlap_merge:4x4_1408
            split_data, starts = splitimage(inputs, crop_size=args.vit_img_size, overlap_size=0)#352_4x4
            for i, data in enumerate(split_data):             
                # 获得输出的小图
                outputs = net(data)
                outputs = net_1(outputs)
                split_data[i] = outputs
            # 获得输出的大图            
            outputs = mergeimage(split_data, starts, crop_size = args.vit_img_size, resolution=(B, C, H, W),is_mean=True)#352_4x4



            Avg_Meters_evaling.update({ 'eval_output_psnr': compute_psnr(outputs, labels),
                                        'eval_input_psnr': compute_psnr(inputs, labels) })
        Final_output_PSNR = Avg_Meters_evaling['eval_output_psnr']
        Final_input_PSNR = Avg_Meters_evaling['eval_input_psnr'] #/ len(eval_loader)
        writer.add_scalars(exper_name + '/testing', {'eval_PSNR_Output': Final_output_PSNR,
                                                     'eval_PSNR_Input': Final_input_PSNR }, epoch)
        if Final_output_PSNR > max_psnr_val:
            max_psnr_val = Final_output_PSNR
            # saving pre-weighted
            torch.save(net_1.state_dict(),
                       SAVE_PATH + f'net_epoch_{epoch}_PSNR_{round(max_psnr_val, 2)}.pth')

        print("epoch:{}---------Dname:{}--------------[Num_eval:{} In_PSNR:{}  Out_PSNR:{}]--------max_psnr_val:{}, cost time: {}".format(epoch, Dname,len(eval_loader),round(Final_input_PSNR, 2),
                                                                                        round(Final_output_PSNR, 2), round(max_psnr_val, 2), time.time() -st ))
        logging.info("epoch:{}---------Dname:{}--------------[Num_eval:{} In_PSNR:{}  Out_PSNR:{}]--------max_psnr_val:{}, cost time: {}".format(epoch, Dname,len(eval_loader),round(Final_input_PSNR, 2),
                                                                                        round(Final_output_PSNR, 2), round(max_psnr_val, 2), time.time() -st ))
        
    return max_psnr_val


def save_imgs_for_visual(path,inputs,labels,outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path,nrow=3, padding=0)

def get_training_data( Crop_patches=args.Crop_patches):
    rootA = args.training_path
    rootA_txt1_list = args.training_path_txt
    train_Pre_dataset_list = []
    for idx_dataset in range(len(rootA_txt1_list)):
        train_Pre_dataset = my_dataset_wTxt(rootA, rootA_txt1_list[idx_dataset],
                                            crop_size=Crop_patches,
                                            fix_sample_A=fix_sampleA,
                                            regular_aug=args.Aug_regular)  # threshold_size =  args.threshold_size
        train_Pre_dataset_list.append(train_Pre_dataset)
    train_pre_datasets = ConcatDataset(train_Pre_dataset_list)
    
    train_loader = DataLoader(dataset=train_pre_datasets, batch_size=args.BATCH_SIZE, num_workers= 8 ,shuffle=True)
    print('len(train_loader):' ,len(train_loader))
    logging.info('len(train_loader):' ,len(train_loader))
    return train_loader

# def get_training_data_re(fix_sampleA=fix_sampleA, Crop_patches=args.Crop_patches):
#     rootA = args.training_path
#     rootA_txt1_list = args.training_data_txt
#     train_Pre_dataset_list = []
#     for idx_dataset in range(len(rootA_txt1_list)):
#         train_Pre_dataset = my_dataset_wTxt(rootA, rootA_txt1_list[idx_dataset],
#                                             crop_size=Crop_patches,
#                                             fix_sample_A=fix_sampleA,
#                                             regular_aug=args.Aug_regular)  # threshold_size =  args.threshold_size
#         train_Pre_dataset_list.append(train_Pre_dataset)
#     train_pre_datasets = ConcatDataset(train_Pre_dataset_list)
#     train_loader = DataLoader(dataset=train_datasets, batch_size=args.BATCH_SIZE, num_workers= 8 ,shuffle=True)
#     print('len(train_loader):' ,len(train_loader))
#     logging.info('len(train_loader):' ,len(train_loader))
#     return train_loader
    

def get_eval_data(val_in_path=args.eval_in_path,val_gt_path =args.eval_gt_path ,trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label =val_gt_path, transform=trans_eval,fix_sample= 500 )
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers= 4)
    print('len(eval_loader):' ,len(eval_loader))
    logging.info('len(train_loader):' ,len(eval_loader))
    return eval_loader
def print_param_number(net):
    print('#generator parameters:', sum(param.numel() for param in net.parameters()))
    logging.info('#generator parameters:', sum(param.numel() for param in net.parameters()))

if __name__ == '__main__':    
    # 是否多阶段复原
    if args.Flag_multi_scale:
        net_1 = NAFNet(img_channel=6, width=args.base_channel, middle_blk_num=args.num_res,
                        enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks,global_residual = False)
    else:
        net_1 = NAFNet(img_channel=args.img_channel, width=args.base_channel, middle_blk_num=args.num_res,
                        enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks,global_residual = False)
    # #net =  ECFNet_complete(base_channel=args.base_channel, num_res=args.num_res    )
    net = MaskedAutoencoderViT(#img_size= args.vit_img_size,
         patch_size=args.vit_patch_size, embed_dim=args.vit_embed_dim, depth=args.vit_depth, num_heads=args.vit_num_heads,
        decoder_embed_dim=args.vit_decoder_embed_dim, decoder_depth=args.vit_decoder_depth, decoder_num_heads=args.vit_decoder_num_heads,
        mlp_ratio=args.vit_mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6))#
    net.load_state_dict(torch.load(args.pre_model), strict=True)
    print('-----'*20,'successfully load vit-pre-trained weights!!!!!')
        
    if args.load_pre_model:
        net.load_state_dict(torch.load(args.pre_model_0), strict=True)
        print('-----'*20,'successfully load pre-trained weights!!!!!')
        logging.info('-----'*20,'successfully load pre-trained weights!!!!!')
        net_1.load_state_dict(torch.load(args.pre_model_1), strict=True)
        print('-----'*20,'successfully load pre-trained weights!!!!!')
        logging.info('-----'*20,'successfully load pre-trained weights!!!!!')
    
    #if torch.__version__[0] == '2':
    #    net = torch.compile(net)
    #    print('-----'*20, torch.__version__)
    #    logging.info('-----'*20, torch.__version__)
    net.to(device)
    print_param_number(net)
    net_1.to(device)
    print_param_number(net_1)    

        
    train_loader = get_training_data()
    eval_loader  = get_eval_data(val_in_path=args.eval_in_path,val_gt_path =args.eval_gt_path)
    if args.optim.lower() == 'adamw':
        optimizerG = optim.AdamW(net.parameters(), lr=args.learning_rate,betas=(0.9,0.999))
    elif args.optim.lower() == 'lion':
        optimizerG = Lion(net.parameters(), lr=args.learning_rate,betas=(0.9,0.999))
    else:
        # optimizerG = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9,0.999) )
        optimizerG = optim.Adam(net_1.parameters(), lr=args.learning_rate, betas=(0.9,0.999) )
    
    # 冻结分支1的参数
    for param in net.parameters():
        param.requires_grad = False

    scheduler = CosineAnnealingWarmRestarts(optimizerG, T_0=args.T_period,  T_mult=1) #ExponentialLR(optimizerG, gamma=0.98)


    if args.base_loss.lower() == 'char':
        base_loss = losses.CharbonnierLoss()
    elif args.base_loss.lower() == 'weightedchar':
        base_loss = losses.WeightedCharbonnierLoss(eps=1e-4, weight = args.weight_coff)
    else:
        base_loss = nn.L1Loss()

    if args.addition_loss.lower()  == 'vgg':
        criterion = losses.VGGLoss()
    elif args.addition_loss.lower()  == 'fft':
        criterion = losses.fftLoss()
    elif args.addition_loss.lower()  == 'ssim':
        criterion = losses.SSIMLoss()   
        
    criterion_depth = nn.L1Loss()

    # recording values! ( training process~)
    running_results = { 'iter_nums' : 0  , 'max_psnr_valD': 0  }
    # 'max_psnr_valD' :  args.max_psnr, 'total_loss': 0.0,  'total_loss1': 0.0,
    #                         'total_loss2': 0.0,  'input_PSNR_all': 0.0,  'train_PSNR_all': 0.0,

    Avg_Meters_training = AverageMeters()

    #iter_nums = 0
    for epoch in range(args.EPOCH):
        scheduler.step(epoch)
        st = time.time()
        for i,train_data in enumerate(train_loader,0):
            data_in, label, img_name = train_data
            if i ==0:
                print(f" train_input.size: {data_in.size()}, gt.size: {label.size()}")
                logging.info(f" train_input.size: {data_in.size()}, gt.size: {label.size()}")
            running_results['iter_nums'] +=1
            net_1.train()
            net_1.zero_grad()
            optimizerG.zero_grad()
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            


            ################### 分割成小图送入vit##########################
            # 获得输入的小图
            sub_images , positions = split_image(inputs, args.vit_grid_type)
            # 获得输出的小图
            # img_size = torch.tensor([0,0])
            # img_size[0] = sub_images[0].shape[2]
            # img_size[1] = sub_images[0].shape[3]
            # 是否并行输入网络
            if args.Flag_process_split_image_with_model_parallel:
                if args.Flag_multi_scale:
                    results_1 = process_split_image_with_model_parallel(sub_images, net)  
                    # results_1和sub_images是两个列表
                    # results_2 = torch.cat((results_1[:], sub_images[:]), dim=1)
                    results_2 = [torch.cat((img1, img2), dim=1) for img1, img2 in zip(results_1, sub_images)]             
                    results = process_split_image_with_model_parallel(results_2, net_1)
                else:
                    results = process_split_image_with_model_parallel(sub_images, net)                
                    results = process_split_image_with_model_parallel(results, net_1)
            else:
                if args.Flag_multi_scale:
                    results_1 = process_split_image_with_model(sub_images, net)  
                    # results_1和sub_images是两个列表
                    # results_2 = torch.cat((results_1[:], sub_images[:]), dim=1)
                    results_2 = [torch.cat((img1, img2), dim=1) for img1, img2 in zip(results_1, sub_images)]             
                    results = process_split_image_with_model(results_2, net_1)
                else:
                    results = process_split_image_with_model(sub_images, net)                
                    results = process_split_image_with_model(results, net_1)
            # 获得输出的大图
            train_output = merge(results, positions).to(device)
            ################### 分割成小图送入vit##########################



            # # overlap_merge: 4x4_1408
            # B, C, H, W = inputs.shape
            # split_data, starts = splitimage(train_output, crop_size=args.vit_img_size, overlap_size=0)#352_4x4
            # for i, data in enumerate(split_data):             
            #     # 获得输出的小图
            #     # outputs = net(data)
            #     outputs = net_1(data)
            #     split_data[i] = outputs
            # # 获得输出的大图
            # train_output = mergeimage(split_data, starts, crop_size = args.vit_img_size, resolution=(B, C, H, W),is_mean=True)#352_4x4




            # calcuate metrics
            input_PSNR = compute_psnr(inputs, labels)
            trian_PSNR = compute_psnr(train_output, labels)

            loss1 =  base_loss(train_output, labels) # losses.multi_scale_losses(train_output, labels, base_loss )#  ()

            if args.addition_loss.lower() == 'vgg':
                loss2 =  args.addition_loss_coff * criterion(train_output, labels)  # losses.multi_scale_losses(train_output, labels, criterion )
                g_loss = loss1  + loss2
                loss3 = loss1
            elif args.addition_loss.lower() == 'fft':
                loss2 =  args.addition_loss_coff * criterion(train_output, labels)
                g_loss = loss1  + loss2
                loss3 = loss1
            elif args.addition_loss.lower() == 'ssim':
                loss2 =  args.addition_loss_coff * criterion(train_output, labels)
                g_loss = loss1  + loss2
                loss3 = loss1
            else:
                g_loss = loss1 #+ loss2
                loss2 = loss1   # 0.1 * criterion(train_output, labels)
                loss3 = loss1

            # if args.depth_loss :
            #     loss3 = args.lam_DepthLoss * criterion_depth(train_output, labels)
            #     g_loss = loss1 + loss2 + loss3
            # else:
            #     g_loss = loss1 + loss2
            #     loss3 = loss1

            Avg_Meters_training.update({'total_loss': g_loss.item(),  'total_loss1': loss1.item(),   'total_loss2': loss2.item(),
                                        'total_loss3': loss3.item(), 'input_PSNR_all': input_PSNR, 'train_PSNR_all': trian_PSNR
                                         })
            g_loss.backward()
            optimizerG.step()
            if (i+1) % args.print_frequency ==0 and i >1:
                writer.add_scalars(exper_name +'/training' ,{'PSNR_Output':  Avg_Meters_training['train_PSNR_all'], 'PSNR_Input':  Avg_Meters_training['input_PSNR_all'], } , running_results['iter_nums'])
                writer.add_scalars(exper_name +'/training' ,{'total_loss': Avg_Meters_training['total_loss']  ,'loss1_char':  Avg_Meters_training['total_loss1'] , 'loss2': Avg_Meters_training['total_loss2'],
                                                             'loss3': Avg_Meters_training['total_loss3']  } , running_results['iter_nums'])
                print(
                    "epoch:%d,[%d / %d], [lr: %.7f ],[loss:%.5f,loss1:%.5f,loss2:%.5f,loss3:%.5f, avg_loss:%.5f],[in_PSNR: %.3f, out_PSNR: %.3f],time:%.3f" %
                    (epoch, i + 1, len(train_loader), optimizerG.param_groups[0]["lr"], g_loss.item(), loss1.item(),
                     loss2.item(), loss3.item(), Avg_Meters_training['total_loss'], input_PSNR, trian_PSNR, time.time() - st))
                logging.info(
                    "epoch:%d,[%d / %d], [lr: %.7f ],[loss:%.5f,loss1:%.5f,loss2:%.5f,loss3:%.5f, avg_loss:%.5f],[in_PSNR: %.3f, out_PSNR: %.3f],time:%.3f" %
                    (epoch, i + 1, len(train_loader), optimizerG.param_groups[0]["lr"], g_loss.item(), loss1.item(),
                     loss2.item(), loss3.item(), Avg_Meters_training['total_loss'], input_PSNR, trian_PSNR, time.time() - st))
                
                st = time.time()
                
                if args.SAVE_Inter_Results:
                    save_path = SAVE_Inter_Results_PATH + str(running_results['iter_nums']) + '.jpg'
                    save_imgs_for_visual(save_path, inputs, labels, train_output)

        # evaluation
        running_results['max_psnr_valD'] = test(net= net,net_1=net_1,eval_loader = eval_loader,epoch=epoch,max_psnr_val = running_results['max_psnr_valD'], Dname = 'evalD')



# CUDA_VISIBLE_DEVICES=4 python /root/ShadowChallendge/train_shadow_vit_wNAF.py --enc_blks 1 1 1 28 --dec_blks 1 1 1 1 --EPOCH 1000 --BATCH_SIZE  1 --Crop_patches  1408 --learning_rate 0.0004 --experiment_name train_shadow_vit_wNAF_BS.10_PS.1408_LR.4e-4_refine


# CUDA_VISIBLE_DEVICES=8 nohup \
# python /root/ShadowChallendge/train_shadow_vit_wNAF.py \
# --enc_blks 1 1 1 28 \
# --dec_blks 1 1 1 1 \
# --EPOCH 1000 \
# --BATCH_SIZE  1 \
# --Crop_patches  1408 \
# --learning_rate 0.0004 \
# --experiment_name train_shadow_vit_wNAF_BS.10_PS.1408_LR.4e-4_refine & 

# CUDA_VISIBLE_DEVICES=9 \
# python /root/ShadowChallendge/train_shadow_vit_wNAF.py \
# --Flag_multi_scale True \
# --enc_blks 1 1 1 28 \
# --dec_blks 1 1 1 1 \
# --EPOCH 1000 \
# --BATCH_SIZE  1 \
# --Crop_patches  1408 \
# --learning_rate 0.0004 \
# --experiment_name train_shadow_vit_wNAF_BS.10_PS.1408_LR.4e-4_refine &  nohup










