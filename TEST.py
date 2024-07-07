import time,torchvision,argparse,logging,sys,os,gc
import torch,random,tqdm,collections
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
from utils.UTILS1 import compute_psnr
from utils.UTILS import AverageMeters,print_args_parameters, compute_ssim
from datasets.datasets_pairs import my_dataset,my_dataset_eval,my_dataset_wTxt
from networks.MaeVit_arch import MaskedAutoencoderViT
from networks.Split_images import split_image,process_split_image_with_model,merge,process_split_image_with_model_1
from functools import partial
from networks.image_utils import splitimage, mergeimage

sys.path.append(os.getcwd())

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:',device)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--vit_patch_size', type=int, default= 8)    
parser.add_argument('--vit_embed_dim', type=int, default= 256)
parser.add_argument('--vit_depth', type=int, default= 6)
parser.add_argument('--vit_channel', type=int, default= 32)
parser.add_argument('--vit_num_heads', type=int, default= 8)
parser.add_argument('--vit_decoder_embed_dim', type=int, default= 256)    
parser.add_argument('--vit_decoder_depth', type=int, default= 6)
parser.add_argument('--vit_decoder_num_heads', type=int, default= 8)
parser.add_argument('--vit_mlp_ratio', type=int, default= 4)
parser.add_argument('--vit_img_size', type=int, default= 352)
parser.add_argument('--vit_grid_type', type=str, default= '4x4')
# Flag_process_split_image_with_model_parallel
parser.add_argument('--Flag_process_split_image_with_model_parallel', type=int, default= 1)
parser.add_argument('--overlap_size', type=int, default= 0)
parser.add_argument('--Crop_patches', type=int, default= 352)

# path setting
parser.add_argument('--experiment_name', type=str,default= "test") # modify the experiments name-->modify all save path
parser.add_argument('--unified_path', type=str,default=  '/log_file')
parser.add_argument('--result_path', type=str,default=  './running_result/')
parser.add_argument('--eval_in_path', type=str,default= './ntire_24_sh_rem_final_test_inp/')
parser.add_argument('--eval_gt_path', type=str,default= '')
# load load_pre_model
parser.add_argument('--pre_model', type=str, default= '')
parser.add_argument('--pre_model_0', type=str, default= './ckpt/best1.pth')
parser.add_argument('--pre_model_1', type=str, default= './ckpt/best2.pth')
parser.add_argument('--model', type=str, default= 'vit_naf')
parser.add_argument('--pre_model_dir', type=str, default= '')

# save results
parser.add_argument('--models_ensemble', type= str2bool, default= False)
parser.add_argument('--inputs_ensemble', type= str2bool, default= False)

# model setting
parser.add_argument('--base_channel', type = int, default= 24)
parser.add_argument('--num_res', type=int, default= 6)
parser.add_argument('--img_channel', type=int, default= 3)
parser.add_argument('--vgg_blks', type=str, default= './loss/vgg19-dcbb9e9d.pth')
parser.add_argument('--enc_blks', nargs='+', type=int, default= [1, 1, 1, 28], help='List of integers')
parser.add_argument('--dec_blks', nargs='+', type=int, default= [1, 1, 1, 1], help='List of integers')
parser.add_argument('--MultiScale', type=str2bool, default= False)  
parser.add_argument('--global_residual', type=str2bool, default= True) 
parser.add_argument('--evalD', type=str, default= '')


args = parser.parse_args()
args.eval_gt_path = args.eval_in_path 
# print all args params!!!
print_args_parameters(args)


exper_name =args.experiment_name



unified_path = args.unified_path
SAVE_PATH = unified_path  + '/' #+ exper_name
# if not os.path.exists(SAVE_PATH):
#     os.makedirs(SAVE_PATH)
log_dir = args.result_path + SAVE_PATH
if not os.path.exists(log_dir):
    os.makedirs(log_dir)    
    
trans_eval = transforms.Compose(
        [
         transforms.ToTensor()
        ])

results_mertircs = log_dir + exper_name + '.txt' #_pathSAVE_PATH + exper_name + '.txt'


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def test(net,net_1,net_0,eval_loader,Dname = 'S', save_result = False):
    net.eval()
    with torch.no_grad():
        #eval_results =
        Avg_Meters_evaling =AverageMeters()
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            outputs = net_1(inputs)    
            B, C, H, W = inputs.shape
            # print(inputs.shape)
            if index ==0:
                print(f" val_input.size: {data_in.size()}, gt.size: {label.size()}")
                logging.info(f" val_input.size: {data_in.size()}, gt.size: {label.size()}")
            # # overlap_merge:
            # split_data, starts = splitimage(inputs, crop_size=args.Crop_patches, overlap_size=args.overlap_size)
            # for i, data in enumerate(split_data):                
            #     # 是否切块输入网络
            #     if args.Flag_process_split_image_with_model_parallel:
            #         ################### 分割成小图送入vit##########################
            #         # 获得输入的小图
            #         sub_images , positions = split_image(data, args.vit_grid_type)
            #         # 获得输出的小图
            #         # img_size = torch.tensor([0,0])
            #         # img_size[0] = sub_images[0].shape[2]
            #         # img_size[1] = sub_images[0].shape[3]
            #         # 并行输入网络
            #         results = process_split_image_with_model_parallel(sub_images, net)
            #         # 获得输出的大图
            #         outputs = merge(results, positions)
            #         split_data[i] = outputs
            #         ################### 分割成小图送入vit##########################
            #     else:
            #         outputs = net(inputs)
            #         split_data[i] = outputs
            # outputs = mergeimage(split_data, starts, crop_size=args.Crop_patches, resolution=(B, C, H, W),is_mean=False)
                

            # overlap_merge:4x4_1408
            outputs = process_split_image_with_model_1(net,net_0,outputs,name,inputs)
            #352_4x4                       
            out_psnr, out_psnr_wClip, out_ssim = compute_psnr(outputs, labels), compute_psnr(torch.clamp(outputs, 0., 1.), labels), compute_ssim(outputs, labels)
            in_psnr, in_ssim = compute_psnr(inputs, labels), compute_ssim(inputs, labels)
            
            Avg_Meters_evaling.update({ 'eval_output_psnr': out_psnr,
                                       'eval_output_psnr_wClip': out_psnr_wClip,
                                        'eval_input_psnr': in_psnr,
                                      'eval_output_ssim': out_ssim,
                                        'eval_input_ssim': in_ssim  })
            content = f'index: {index} | name :{name[0]}|| [in_psnr :{in_psnr}, in_ssim:{in_ssim}|| out_psnr:{out_psnr}, out_psnr_wClip:{out_psnr_wClip} , out_ssim:{out_ssim} ]'
            print(content)
            with open(results_mertircs, 'a') as file:
                file.write(content)
                file.write('\n')
            
            if save_result:
                save_result_path = args.result_path + '/' #SAVE_PATH   + '/'+ Dname +'/' #+ name[0] + '.png'
                #print(save_path)
                os.makedirs(save_result_path, exist_ok=True)
                torchvision.utils.save_image([ torch.clamp(outputs, 0., 1.).cpu().detach()[0] ], save_result_path  + name[0] , nrow =1, padding=0 )
                #save_imgs_for_visual(save_path, inputs, labels, train_output[-1])
            
        Final_output_PSNR = Avg_Meters_evaling['eval_output_psnr']
        Final_output_PSNR_wclip = Avg_Meters_evaling['eval_output_psnr_wClip']

        Final_input_PSNR = Avg_Meters_evaling['eval_input_psnr'] #/ len(eval_loader)
        Final_output_SSIM = Avg_Meters_evaling['eval_output_ssim']
        Final_input_SSIM = Avg_Meters_evaling['eval_input_ssim']

        

        content_ = f"Dataset:{Dname} || [Num_eval:{len(eval_loader)} In_PSNR:{round(Final_input_PSNR, 3)}  / In_SSIM:{round(Final_input_SSIM, 3)}    ||  Out_PSNR:{round(Final_output_PSNR, 3)} | Out_PSNR_wclip:{round(Final_output_PSNR_wclip, 3)}    / OUT_SSIM:{round(Final_output_SSIM, 3)} ]  cost time: { time.time() -st}"
        
        print(content_)
        with open(results_mertircs, 'a') as file:
                file.write(content_)

# mkdir 
# rsync --delete-before -d /root/autodl-tmp/Shadow_Challendge/new_data/ /root/autodl-tmp/Shadow_Challendge/training_NAFNet_0208SC-wRegAug-w23Dw24D_WweightedCharWlongTrain
def test_overlap(net,eval_loader,Dname = 'S', save_result = False):
    net.eval()
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
            # overlap_merge:
            split_data, starts = splitimage(inputs, crop_size=args.Crop_patches, overlap_size=args.overlap_size)
            for i, data in enumerate(split_data):             
                outputs = net(data)
                split_data[i] = outputs
            outputs = mergeimage(split_data, starts, crop_size=args.Crop_patches, resolution=(B, C, H, W),is_mean=False)


            
            out_psnr, out_psnr_wClip, out_ssim = compute_psnr(outputs, labels), compute_psnr(torch.clamp(outputs, 0., 1.), labels), compute_ssim(outputs, labels)
            in_psnr, in_ssim = compute_psnr(inputs, labels), compute_ssim(inputs, labels)
            
            Avg_Meters_evaling.update({ 'eval_output_psnr': out_psnr,
                                       'eval_output_psnr_wClip': out_psnr_wClip,
                                        'eval_input_psnr': in_psnr,
                                      'eval_output_ssim': out_ssim,
                                        'eval_input_ssim': in_ssim  })
            content = f'index: {index} | name :{name[0]}|| [in_psnr :{in_psnr}, in_ssim:{in_ssim}|| out_psnr:{out_psnr}, out_psnr_wClip:{out_psnr_wClip} , out_ssim:{out_ssim} ]'
            print(content)
            with open(results_mertircs, 'a') as file:
                file.write(content)
                file.write('\n')
            
            if save_result:
                save_result_path = SAVE_PATH  + '/'+ Dname +'/' #+ name[0] + '.png'
                #print(save_path)
                os.makedirs(save_result_path, exist_ok=True)
                torchvision.utils.save_image([ torch.clamp(outputs, 0., 1.).cpu().detach()[0] ], save_result_path + name[0] , nrow =1, padding=0 )
                #save_imgs_for_visual(save_path, inputs, labels, train_output[-1])
            
        Final_output_PSNR = Avg_Meters_evaling['eval_output_psnr']
        Final_output_PSNR_wclip = Avg_Meters_evaling['eval_output_psnr_wClip']

        Final_input_PSNR = Avg_Meters_evaling['eval_input_psnr'] #/ len(eval_loader)
        Final_output_SSIM = Avg_Meters_evaling['eval_output_ssim']
        Final_input_SSIM = Avg_Meters_evaling['eval_input_ssim']

        

        content_ = f"Dataset:{Dname} || [Num_eval:{len(eval_loader)} In_PSNR:{round(Final_input_PSNR, 3)}  / In_SSIM:{round(Final_input_SSIM, 3)}    ||  Out_PSNR:{round(Final_output_PSNR, 3)} | Out_PSNR_wclip:{round(Final_output_PSNR_wclip, 3)}    / OUT_SSIM:{round(Final_output_SSIM, 3)} ]  cost time: { time.time() -st}"
        
        print(content_)
        with open(results_mertircs, 'a') as file:
                file.write(content_)

def test_wInputEnsemble_1(net_1,eval_loader,Dname = 'S', save_result = False):
    net.eval()
    with torch.no_grad():
        #eval_results =
        Avg_Meters_evaling =AverageMeters()
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):          
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            outputs = net_1(inputs)
            # 原始图片
            B, C, H, W = inputs.shape
            if index ==0:
                print(f" val_input.size: {data_in.size()}, gt.size: {label.size()}")
                logging.info(f" val_input.size: {data_in.size()}, gt.size: {label.size()}")
            # overlap_merge:
            normal_outputs = outputs

            
            '''
            flipW
            '''  
            input_flipW = torch.flip(inputs, (-1,))
            outputs = net_1(input_flipW)
            # overlap_merge:


            output_FW = torch.flip(outputs, (-1,))
            normal_outputs += output_FW

            '''
            flipH
            ''' 
            input_flipH = torch.flip(inputs, (-2,))
            outputs = net_1(input_flipH)
            # overlap_merge:

            output_FH = torch.flip(outputs, (-2,))
            normal_outputs += output_FH


            '''
            flipHW
            ''' 
            input_flipHW = torch.flip(inputs, (-2,-1))
            outputs = net_1(input_flipHW)
            # overlap_merge:

            output_FHW = torch.flip(outputs, (-2,-1))
            normal_outputs += output_FHW
            
            # final output
            outputs = normal_outputs / 4.0

            
            out_psnr, out_psnr_wClip, out_ssim = compute_psnr(outputs, labels), compute_psnr(torch.clamp(outputs, 0., 1.), labels), compute_ssim(outputs, labels)
            in_psnr, in_ssim = compute_psnr(inputs, labels), compute_ssim(inputs, labels)
            
            Avg_Meters_evaling.update({ 'eval_output_psnr': out_psnr,
                                       'eval_output_psnr_wClip': out_psnr_wClip,
                                        'eval_input_psnr': in_psnr,
                                      'eval_output_ssim': out_ssim,
                                        'eval_input_ssim': in_ssim  })
            content = f'index: {index} | name :{name[0]}|| [in_psnr :{in_psnr}, in_ssim:{in_ssim}|| out_psnr:{out_psnr}, out_psnr_wClip:{out_psnr_wClip} , out_ssim:{out_ssim} ]'
            
            print(content)
            with open(results_mertircs, 'a') as file:
                file.write(content)
                file.write('\n')
            
            if save_result:
                save_result_path = SAVE_PATH  + '/'+ Dname +'.InputEnsemble' +'/' #save_result_path = SAVE_PATH  + '/cleanImg.wInputEnsemble/' #+ name[0] + '.png'
                #print(save_path)
                os.makedirs(save_result_path, exist_ok=True)
                torchvision.utils.save_image([ torch.clamp(outputs, 0., 1.).cpu().detach()[0] ], save_result_path + name[0] , nrow =1, padding=0 )
            
        Final_output_PSNR = Avg_Meters_evaling['eval_output_psnr']
        Final_input_PSNR = Avg_Meters_evaling['eval_input_psnr'] #/ len(eval_loader)
        Final_output_SSIM = Avg_Meters_evaling['eval_output_ssim']
        Final_input_SSIM = Avg_Meters_evaling['eval_input_ssim']
        Final_output_PSNR_wclip = Avg_Meters_evaling['eval_output_psnr_wClip']

        
        
        #save_imgs_for_visual(save_path, inputs, labels, train_output[-1])
        
        content_ = f"Dataset:{Dname} || [Num_eval:{len(eval_loader)} In_PSNR:{round(Final_input_PSNR, 3)}  / In_SSIM:{round(Final_input_SSIM, 3)}    ||  Out_PSNR:{round(Final_output_PSNR, 3)} | Out_PSNR_wclip:{round(Final_output_PSNR_wclip, 3)}    / OUT_SSIM:{round(Final_output_SSIM, 3)} ]  cost time: { time.time() -st}"
        
        print(content_)
        with open(results_mertircs, 'a') as file:
            file.write(content_)
        
def test_wInputEnsemble(net,eval_loader,Dname = 'S' , save_result = False):
    net.eval()
    with torch.no_grad():
        #eval_results =
        Avg_Meters_evaling =AverageMeters()
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):          
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            # 原始图片
            B, C, H, W = inputs.shape
            if index ==0:
                print(f" val_input.size: {data_in.size()}, gt.size: {label.size()}")
                logging.info(f" val_input.size: {data_in.size()}, gt.size: {label.size()}")
            # overlap_merge:
            split_data, starts = splitimage(inputs, crop_size=args.Crop_patches, overlap_size=args.overlap_size)
            for i, data in enumerate(split_data):                
                # 是否切块输入网络
                if args.Flag_process_split_image_with_model_parallel:
                    ################### 分割成小图送入vit##########################
                    # 获得输入的小图
                    sub_images , positions = split_image(data, args.vit_grid_type)
                    # 获得输出的小图
                    # img_size = torch.tensor([0,0])
                    # img_size[0] = sub_images[0].shape[2]
                    # img_size[1] = sub_images[0].shape[3]
                    # 并行输入网络
                    results = process_split_image_with_model_parallel(sub_images, net)
                    # 获得输出的大图
                    outputs = merge(results, positions)
                    split_data[i] = outputs
                    ################### 分割成小图送入vit##########################
                else:
                    outputs = net(inputs)
                    split_data[i] = outputs
            normal_outputs = mergeimage(split_data, starts, crop_size=args.Crop_patches, resolution=(B, C, H, W),is_mean=False)

            
            '''
            flipW
            '''  
            input_flipW = torch.flip(inputs, (-1,))
            # overlap_merge:
            split_data, starts = splitimage(input_flipW, crop_size=args.Crop_patches, overlap_size=args.overlap_size)
            for i, data in enumerate(split_data):                
                # 是否切块输入网络
                if args.Flag_process_split_image_with_model_parallel:
                    ################### 分割成小图送入vit##########################
                    # 获得输入的小图
                    sub_images , positions = split_image(data, args.vit_grid_type)
                    # 获得输出的小图
                    # img_size = torch.tensor([0,0])
                    # img_size[0] = sub_images[0].shape[2]
                    # img_size[1] = sub_images[0].shape[3]
                    # 并行输入网络
                    results = process_split_image_with_model_parallel(sub_images, net)
                    # 获得输出的大图
                    outputs = merge(results, positions)
                    split_data[i] = outputs
                    ################### 分割成小图送入vit##########################
                else:
                    outputs = net(inputs)
                    split_data[i] = outputs
            outputs = mergeimage(split_data, starts, crop_size=args.Crop_patches, resolution=(B, C, H, W),is_mean=False)
            output_FW = torch.flip(outputs, (-1,))
            normal_outputs += output_FW

            '''
            flipH
            ''' 
            input_flipH = torch.flip(inputs, (-2,))
            # overlap_merge:
            split_data, starts = splitimage(input_flipH, crop_size=args.Crop_patches, overlap_size=args.overlap_size)
            for i, data in enumerate(split_data):                
                # 是否切块输入网络
                if args.Flag_process_split_image_with_model_parallel:
                    ################### 分割成小图送入vit##########################
                    # 获得输入的小图
                    sub_images , positions = split_image(data, args.vit_grid_type)
                    # 获得输出的小图
                    # img_size = torch.tensor([0,0])
                    # img_size[0] = sub_images[0].shape[2]
                    # img_size[1] = sub_images[0].shape[3]
                    # 并行输入网络
                    results = process_split_image_with_model_parallel(sub_images, net)
                    # 获得输出的大图
                    outputs = merge(results, positions)
                    split_data[i] = outputs
                    ################### 分割成小图送入vit##########################
                else:
                    outputs = net(inputs)
                    split_data[i] = outputs
            outputs = mergeimage(split_data, starts, crop_size=args.Crop_patches, resolution=(B, C, H, W),is_mean=False)
            output_FH = torch.flip(outputs, (-2,))
            normal_outputs += output_FH


            '''
            flipHW
            ''' 
            input_flipHW = torch.flip(inputs, (-2,-1))
            # overlap_merge:
            split_data, starts = splitimage(input_flipHW, crop_size=args.Crop_patches, overlap_size=args.overlap_size)
            for i, data in enumerate(split_data):                
                # 是否切块输入网络
                if args.Flag_process_split_image_with_model_parallel:
                    ################### 分割成小图送入vit##########################
                    # 获得输入的小图
                    sub_images , positions = split_image(data, args.vit_grid_type)
                    # 获得输出的小图
                    # img_size = torch.tensor([0,0])
                    # img_size[0] = sub_images[0].shape[2]
                    # img_size[1] = sub_images[0].shape[3]
                    # 并行输入网络
                    results = process_split_image_with_model_parallel(sub_images, net)
                    # 获得输出的大图
                    outputs = merge(results, positions)
                    split_data[i] = outputs
                    ################### 分割成小图送入vit##########################
                else:
                    outputs = net(inputs)
                    split_data[i] = outputs
            outputs = mergeimage(split_data, starts, crop_size=args.Crop_patches, resolution=(B, C, H, W),is_mean=False)
            output_FHW = torch.flip(outputs, (-2,-1))
            normal_outputs += output_FHW
            
            # final output
            outputs = normal_outputs / 4.0

            
            out_psnr, out_psnr_wClip, out_ssim = compute_psnr(outputs, labels), compute_psnr(torch.clamp(outputs, 0., 1.), labels), compute_ssim(outputs, labels)
            in_psnr, in_ssim = compute_psnr(inputs, labels), compute_ssim(inputs, labels)
            
            Avg_Meters_evaling.update({ 'eval_output_psnr': out_psnr,
                                       'eval_output_psnr_wClip': out_psnr_wClip,
                                        'eval_input_psnr': in_psnr,
                                      'eval_output_ssim': out_ssim,
                                        'eval_input_ssim': in_ssim  })
            content = f'index: {index} | name :{name[0]}|| [in_psnr :{in_psnr}, in_ssim:{in_ssim}|| out_psnr:{out_psnr}, out_psnr_wClip:{out_psnr_wClip} , out_ssim:{out_ssim} ]'
            
            print(content)
            with open(results_mertircs, 'a') as file:
                file.write(content)
                file.write('\n')
            
            if save_result:
                save_result_path = SAVE_PATH  + '/'+ Dname +'.InputEnsemble' +'/' #save_result_path = SAVE_PATH  + '/cleanImg.wInputEnsemble/' #+ name[0] + '.png'
                #print(save_path)
                os.makedirs(save_result_path, exist_ok=True)
                torchvision.utils.save_image([ torch.clamp(outputs, 0., 1.).cpu().detach()[0] ], save_result_path + name[0] , nrow =1, padding=0 )
            
        Final_output_PSNR = Avg_Meters_evaling['eval_output_psnr']
        Final_input_PSNR = Avg_Meters_evaling['eval_input_psnr'] #/ len(eval_loader)
        Final_output_SSIM = Avg_Meters_evaling['eval_output_ssim']
        Final_input_SSIM = Avg_Meters_evaling['eval_input_ssim']
        Final_output_PSNR_wclip = Avg_Meters_evaling['eval_output_psnr_wClip']

        
        
        #save_imgs_for_visual(save_path, inputs, labels, train_output[-1])
        
        content_ = f"Dataset:{Dname} || [Num_eval:{len(eval_loader)} In_PSNR:{round(Final_input_PSNR, 3)}  / In_SSIM:{round(Final_input_SSIM, 3)}    ||  Out_PSNR:{round(Final_output_PSNR, 3)} | Out_PSNR_wclip:{round(Final_output_PSNR_wclip, 3)}    / OUT_SSIM:{round(Final_output_SSIM, 3)} ]  cost time: { time.time() -st}"
        
        print(content_)
        with open(results_mertircs, 'a') as file:
            file.write(content_)

def test_wInputEnsemble_over(net,eval_loader,Dname = 'S' , save_result = False):
    net.eval()
    with torch.no_grad():
        #eval_results =
        Avg_Meters_evaling =AverageMeters()
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):          
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            # 原始图片
            B, C, H, W = inputs.shape
            if index ==0:
                print(f" val_input.size: {data_in.size()}, gt.size: {label.size()}")
                logging.info(f" val_input.size: {data_in.size()}, gt.size: {label.size()}")
            # overlap_merge:
            split_data, starts = splitimage(inputs, crop_size=args.Crop_patches, overlap_size=args.overlap_size)
            for i, data in enumerate(split_data):               
               
                outputs = net(data)
                split_data[i] = outputs
            normal_outputs = mergeimage(split_data, starts, crop_size=args.Crop_patches, resolution=(B, C, H, W),is_mean=False)

            
            '''
            flipW
            '''  
            input_flipW = torch.flip(inputs, (-1,))
            # overlap_merge:
            split_data, starts = splitimage(input_flipW, crop_size=args.Crop_patches, overlap_size=args.overlap_size)
            for i, data in enumerate(split_data):                
                    outputs = net(data)
                    split_data[i] = outputs
            outputs = mergeimage(split_data, starts, crop_size=args.Crop_patches, resolution=(B, C, H, W),is_mean=False)
            output_FW = torch.flip(outputs, (-1,))
            normal_outputs += output_FW

            '''
            flipH
            ''' 
            input_flipH = torch.flip(inputs, (-2,))
            # overlap_merge:
            split_data, starts = splitimage(input_flipH, crop_size=args.Crop_patches, overlap_size=args.overlap_size)
            for i, data in enumerate(split_data):                
                    outputs = net(data)
                    split_data[i] = outputs
            outputs = mergeimage(split_data, starts, crop_size=args.Crop_patches, resolution=(B, C, H, W),is_mean=False)
            output_FH = torch.flip(outputs, (-2,))
            normal_outputs += output_FH


            '''
            flipHW
            ''' 
            input_flipHW = torch.flip(inputs, (-2,-1))
            # overlap_merge:
            split_data, starts = splitimage(input_flipHW, crop_size=args.Crop_patches, overlap_size=args.overlap_size)
            for i, data in enumerate(split_data):                
                    outputs = net(data)
                    split_data[i] = outputs
            outputs = mergeimage(split_data, starts, crop_size=args.Crop_patches, resolution=(B, C, H, W),is_mean=False)
            output_FHW = torch.flip(outputs, (-2,-1))
            normal_outputs += output_FHW
            
            # final output
            outputs = normal_outputs / 4.0

            
            out_psnr, out_psnr_wClip, out_ssim = compute_psnr(outputs, labels), compute_psnr(torch.clamp(outputs, 0., 1.), labels), compute_ssim(outputs, labels)
            in_psnr, in_ssim = compute_psnr(inputs, labels), compute_ssim(inputs, labels)
            
            Avg_Meters_evaling.update({ 'eval_output_psnr': out_psnr,
                                       'eval_output_psnr_wClip': out_psnr_wClip,
                                        'eval_input_psnr': in_psnr,
                                      'eval_output_ssim': out_ssim,
                                        'eval_input_ssim': in_ssim  })
            content = f'index: {index} | name :{name[0]}|| [in_psnr :{in_psnr}, in_ssim:{in_ssim}|| out_psnr:{out_psnr}, out_psnr_wClip:{out_psnr_wClip} , out_ssim:{out_ssim} ]'
            
            print(content)
            with open(results_mertircs, 'a') as file:
                file.write(content)
                file.write('\n')
            
            if save_result:
                save_result_path = SAVE_PATH  + '/'+ Dname +'.InputEnsemble' +'/' #save_result_path = SAVE_PATH  + '/cleanImg.wInputEnsemble/' #+ name[0] + '.png'
                #print(save_path)
                os.makedirs(save_result_path, exist_ok=True)
                torchvision.utils.save_image([ torch.clamp(outputs, 0., 1.).cpu().detach()[0] ], save_result_path + name[0] , nrow =1, padding=0 )
            
        Final_output_PSNR = Avg_Meters_evaling['eval_output_psnr']
        Final_input_PSNR = Avg_Meters_evaling['eval_input_psnr'] #/ len(eval_loader)
        Final_output_SSIM = Avg_Meters_evaling['eval_output_ssim']
        Final_input_SSIM = Avg_Meters_evaling['eval_input_ssim']
        Final_output_PSNR_wclip = Avg_Meters_evaling['eval_output_psnr_wClip']

        
        
        #save_imgs_for_visual(save_path, inputs, labels, train_output[-1])
        
        content_ = f"Dataset:{Dname} || [Num_eval:{len(eval_loader)} In_PSNR:{round(Final_input_PSNR, 3)}  / In_SSIM:{round(Final_input_SSIM, 3)}    ||  Out_PSNR:{round(Final_output_PSNR, 3)} | Out_PSNR_wclip:{round(Final_output_PSNR_wclip, 3)}    / OUT_SSIM:{round(Final_output_SSIM, 3)} ]  cost time: { time.time() -st}"
        
        print(content_)
        with open(results_mertircs, 'a') as file:
            file.write(content_)

def save_imgs_for_visual(path,inputs,labels,outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path,nrow=3, padding=0)



def get_eval_data(val_in_path=args.eval_in_path,val_gt_path =args.eval_gt_path ,trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label =val_gt_path, transform=trans_eval,fix_sample= 500 )
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers= 4)
    return eval_loader
def print_param_number(net):
    print('#generator parameters:', sum(param.numel() for param in net.parameters()))
   
def merge_models( net ,folder_path= args.pre_model_dir):
    #folder_path = '/path/to/your/folder'   
    model_list = []

    models_names = [f for f in os.listdir(folder_path) if f.endswith('.pth')]
    model_path_list = []
    for i in range(len(models_names)):
        model_path_list.append(folder_path + models_names[i])

    # print the models which need to be merged
    for i in range(len(model_path_list)):
        print('i------:',model_path_list[i])

    # load all pre-trained weights
    for model_path in model_path_list:
        net.eval()
        net.load_state_dict(torch.load(model_path))
        model_list.append(net)
    print("All models loaded successfully")
    print("num of models to be merged is : " + str(len(model_list)))

    merge_model = net
    print("*" * 20)
    print("merging ...")
    worker_state_dict = [x.state_dict() for x in model_list]
    weight_keys = list(worker_state_dict[0].keys()) #tqdm(list(worker_state_dict[0].keys()))
    #print('list(worker_state_dict[0].keys())----------------', list(worker_state_dict[0].keys()))
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        
        #print('key:-------------', key, 'len(model_list)----------', len(model_list))
        #weight_keys.set_description("merging weights %s" % key)
        
        key_sum = 0
        for i in range(len(model_list)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(model_list)
    merge_model.load_state_dict(fed_state_dict)

    return merge_model

if __name__ == '__main__':


    if args.model == 'vit':
        from networks.MaeVit_arch import MaskedAutoencoderViT
        net = MaskedAutoencoderViT(#img_size= args.vit_img_size,
            patch_size=args.vit_patch_size, embed_dim=args.vit_embed_dim, depth=args.vit_depth, num_heads=args.vit_num_heads,
            decoder_embed_dim=args.vit_decoder_embed_dim, decoder_depth=args.vit_decoder_depth, decoder_num_heads=args.vit_decoder_num_heads,
            mlp_ratio=args.vit_mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6))#
    if args.model == 'vit_naf':
        from networks.NAFNet_arch import NAFNet
        net = MaskedAutoencoderViT(#img_size= args.vit_img_size,
            patch_size=args.vit_patch_size, embed_dim=args.vit_embed_dim, depth=args.vit_depth, num_heads=args.vit_num_heads,
            decoder_embed_dim=args.vit_decoder_embed_dim, decoder_depth=args.vit_decoder_depth, decoder_num_heads=args.vit_decoder_num_heads,
            mlp_ratio=args.vit_mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        net_0 = NAFNet(img_channel=args.img_channel, width=args.base_channel, middle_blk_num=args.num_res,
                        enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks,global_residual = False)
        net_0.to(device)
    net_1 = NAFNet(img_channel=args.img_channel, width=args.vit_channel, middle_blk_num=args.num_res,
                        enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks,global_residual = False)
    net_1.load_state_dict(torch.load(args.vgg_blks), strict=True)
    if args.models_ensemble:
        net = merge_models(net=net, folder_path=args.pre_model_dir)
        print('-----'*20,'successfully load merged weights!!!!! (with models_ensemble)')

    else:
        net_1.to(device)
        net.load_state_dict(torch.load(args.pre_model_0), strict=True)
        net_0.load_state_dict(torch.load(args.pre_model_1), strict=True)
        print('-----'*20,'successfully load pre-trained weights!!!!! (without models_ensemble)')
            
    #net.load_state_dict(torch.load(args.pre_model), strict=True)
    #print('-----'*8, 'successfully load pre-trained weights!!!!!','-----'*8)
    net.to(device)
    print_param_number(net)
    print_param_number(net_0)
    eval_loader  = get_eval_data(val_in_path=args.eval_in_path,val_gt_path =args.eval_gt_path)

    # test(net= net,eval_loader = eval_loader, Dname = args.evalD , save_result = True)
    # test_overlap(net= net,eval_loader = eval_loader, Dname = args.evalD , save_result = True)
    test(net= net,net_1=net_1,net_0=net_0,eval_loader = eval_loader, Dname = args.evalD , save_result = True)
    # test_wInputEnsemble_1 = (net= net,net_1 = net_1,eval_loader = eval_loader, Dname = args.evalD , save_result = True)
    with open(results_mertircs, 'a') as file:
        file.write('-=-='*50)
    if args.inputs_ensemble:
        # test_wInputEnsemble(net= net,eval_loader = eval_loader, Dname =   args.evalD , save_result = True)
        test_wInputEnsemble_over(net= net,eval_loader = eval_loader, Dname =   args.evalD , save_result = True)

# 240306：
# CUDA_VISIBLE_DEVICES=9 python  /root/ShadowChallendge/testing_shadow_vit.py  --experiment_name testing_shadow_vit_PL_BS.10_PS.800_LR.4e-4_4x4_On24evalD   --model vit    --models_ensemble  False  --inputs_ensemble False  --pre_model /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.10_PS.800_LR.4e-4_4x4/net_epoch_274_PSNR_22.56.pth  --evalD 24-evalD --eval_in_path /root/autodl-tmp/ShadowDatasets/ntire24_shrem_valid_inp/   --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire24_shrem_valid_inp/ 
# CUDA_VISIBLE_DEVICES=10 python  /root/ShadowChallendge/testing_shadow_vit.py  --experiment_name testing_shadow_vit_PL_BS.6_PS.1024_LR.4e-4_4x4_On24evalD_ensemble   --model vit    --models_ensemble  False  --inputs_ensemble True  --pre_model /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.4_PS.1440_LR.4e-4_4x4/net_epoch_471_PSNR_23.09.pth  --evalD 24-evalD_overlap_ensemble --eval_in_path /root/autodl-tmp/ShadowDatasets/ntire24_shrem_valid_inp/   --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire23_sr_valid_gt/ --Flag_process_split_image_with_model_parallel True --overlap_size 16  --Crop_patches  1024


# 240313:
# CUDA_VISIBLE_DEVICES=0 python  /root/ShadowChallendge/testing_shadow_vit.py  --experiment_name testing_shadow_vit_PL_BS.6_PS.1024_LR.4e-4_4x4_On24evalD_ensemble_over   --model vit    --models_ensemble  False  --inputs_ensemble True  --pre_model /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.4_PS.1440_LR.4e-4_4x4/net_epoch_471_PSNR_23.09.pth  --evalD 24-evalD_over_ensemble_128   --eval_in_path /root/autodl-tmp/ShadowDatasets/ntire24_shrem_valid_inp/   --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire23_sr_valid_gt/
# 

# CUDA_VISIBLE_DEVICES=11 python  /root/ShadowChallendge/testing_shadow_vit.py  --experiment_name testing_shadow_vit_PL_BS.3_PS.1408_LR.2e-4_4x4_fine_tune_On24evalD  --model vit    --models_ensemble  False  --inputs_ensemble True  --pre_model /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.2e-4_4x4_fine_tune/net_epoch_202_PSNR_23.21.pth  --evalD 24-evalD1408   --eval_in_path /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp  --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire23_sr_valid_gt/  --Crop_patches 352  --overlap_size 176
# CUDA_VISIBLE_DEVICES=11 python  /root/ShadowChallendge/testing_shadow_vit.py  --experiment_name testing_shadow_vit_PL_BS.3_PS.1408_LR.2e-4_4x4_fine_tune_On24evalD  --model vit    --models_ensemble  False  --inputs_ensemble True  --pre_model /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.2e-4_4x4_fine_tune/net_epoch_202_PSNR_23.21.pth  --evalD 24-evalD1024   --eval_in_path /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp  --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire23_sr_valid_gt/  --Crop_patches 256  --overlap_size 128

# CUDA_VISIBLE_DEVICES=11 python  /root/ShadowChallendge/testing_shadow_vit.py  --experiment_name testing_shadow_vit_PL_Batch.6_Patch.1024_LR.4e-4_8x8_PSNR_23.2_On24evalD  --model vit    --models_ensemble  False  --inputs_ensemble True  --pre_model  /root/autodl-tmp/SR_1/train_shadow_vit_PL_Batch.6_Patch.1024_LR.4e-4_8x8/net_epoch_381_PSNR_23.2.pth  --evalD 24-evalD1024.8   --eval_in_path /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp  --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire23_sr_valid_gt/  --Crop_patches 128  --overlap_size 64

# CUDA_VISIBLE_DEVICES=9 \
# python  /root/ShadowChallendge/testing_shadow_vit_wNAF.py  \
# --experiment_name testing_shadow_vit_naf_On24evalD  \
# --model vit_naf    \
# --models_ensemble  False  \
# --inputs_ensemble False  \
# --pre_model_0  /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.2e-4_4x4_fine_tune/net_epoch_202_PSNR_23.21.pth  \
# --pre_model_1  /root/autodl-tmp/SR_1/train_shadow_vit_wNAF_BS.10_PS.1408_LR.4e-4_refine/net_epoch_8_PSNR_23.68.pth  \
# --evalD 24-evalD_1408.4.naf   \
# --eval_in_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --Crop_patches 352  \
# --overlap_size 176

# CUDA_VISIBLE_DEVICES=9 \
# python  /root/ShadowChallendge/testing_shadow_vit_wNAF.py  \
# --experiment_name testing_shadow_vit_naf_On24evalD  \
# --model vit_naf    \
# --models_ensemble  False  \
# --inputs_ensemble False  \
# --pre_model_0  /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.2e-4_4x4_fine_tune/net_epoch_202_PSNR_23.21.pth  \
# --pre_model_1  /root/autodl-tmp/SR_1/train_shadow_vit_wNAF_BS.10_PS.1408_LR.4e-4_refine/net_epoch_50_PSNR_23.83.pth  \
# --evalD 24-evalD_1408.4.naf_PSNR_23.83   \
# --eval_in_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --vit_img_size 352  \
# --overlap_size 176





 

# CUDA_VISIBLE_DEVICES=8 \
# python  /root/ShadowChallendge/testing_shadow_vit_wNAF.py  \
# --experiment_name testing_shadow_vit_naf_On24evalD  \
# --model vit_naf    \
# --models_ensemble  False  \
# --inputs_ensemble False  \
# --pre_model_0  /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.3e-4_4x4_fine_tune_fakeD_test/net_epoch_26_PSNR_23.15_test_18.85.pth  \
# --pre_model_1  /root/autodl-tmp/SR_1/train_shadow_vit_wNAF_BS.1_PS.1408_LR.3e-4_refine_fakeData_freeze_vit_PSNR_23.15_test_18.85.pth/net_naf_epoch_12_PSNR_23.55_test_18.87_best_23.55.pth  \
# --evalD 24-evalD_1408.4.naf_PSNR_23.55_test_18.87   \
# --eval_in_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --vit_img_size 352  \
# --overlap_size 176


# CUDA_VISIBLE_DEVICES=8 \
# python  /root/ShadowChallendge/testing_shadow_vit_wNAF.py  \
# --experiment_name testing_shadow_vit_naf_On24evalD  \
# --model vit_naf    \
# --models_ensemble  False  \
# --inputs_ensemble False  \ sftp://root@connect.westb.seetacloud.com:33671/root/autodl-tmp/Shadow_Challendge/training_vit_naf/net_epoch_105_PSNR_23.67.pth
# --pre_model_0  /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.3e-4_4x4_fine_tune_fakeD_mix/net_epoch_52_PSNR_23.21_test_19.01.pth  \
# --pre_model_1  /root/autodl-tmp/Shadow_Challendge/training_vit_naf/net_epoch_39_PSNR_23.34.pth  \
# --evalD 24-evalD_1408.4.naf_PSNR_39_PSNR_23.34   \
# --eval_in_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --vit_img_size 352  \
# --base_channel 32 \
# --overlap_size 176




# CUDA_VISIBLE_DEVICES=8 \
# python  /root/ShadowChallendge/testing_shadow_vit_wNAF.py  \
# --experiment_name testing_shadow_vit_naf_On24evalD  \
# --model vit_naf    \
# --models_ensemble  False  \
# --inputs_ensemble False  \
# --pre_model_0   /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.3e-4_4x4_fine_tune_fakeD_test/net_epoch_26_PSNR_23.15_test_18.85.pth \
# --pre_model_1  /root/autodl-tmp/SR_1/train_shadow_vit_wNAF_BS.1_PS.1408_LR.4e-4_refine_fakeData_freeze_vit_PSNR_23.15_test_18.85.pth/net_naf_epoch_17_PSNR_23.6_test_18.91_best_23.6.pth  \
# --evalD 24-evalD_1408.4.naf_PSNR_23.6_test_18.91   \
# --eval_in_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --vit_img_size 352  \
# --base_channel 24 \
# --overlap_size 176



# CUDA_VISIBLE_DEVICES=8 \
# python  /root/ShadowChallendge/testing_shadow_vit_wNAF.py  \
# --experiment_name testing_shadow_vit_naf_On24evalD  \
# --model vit_naf    \
# --models_ensemble  False  \
# --inputs_ensemble False  \
# --pre_model_0   /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.3e-4_4x4_fine_tune_fakeD_test/net_epoch_26_PSNR_23.15_test_18.85.pth \
# --pre_model_1  /root/autodl-tmp/SR_1/train_shadow_vit_wNAF_BS.1_PS.1408_LR.4e-4_refine_fakeData_freeze_vit_PSNR_23.15_test_18.85.pth/net_naf_epoch_18_PSNR_23.62_test_18.87_best_23.62.pth  \
# --evalD 24-evalD_1408.4.naf_PSNR_23.62_test_18.87_ave   \
# --eval_in_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --vit_img_size 352  \
# --base_channel 24 \
# --overlap_size 176



# CUDA_VISIBLE_DEVICES=8 \
# python  /root/ShadowChallendge/testing_shadow_vit_wNAF.py  \
# --experiment_name testing_shadow_vit_naf_On24evalD  \
# --model vit_naf    \
# --models_ensemble  False  \
# --inputs_ensemble False  \
# --pre_model_0  /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.3e-4_4x4_fine_tune_fakeD_mix/net_epoch_52_PSNR_23.21_test_19.01.pth  \
# --pre_model_1   /root/autodl-tmp/Shadow_Challendge/training_vit_naf/net_epoch_105_PSNR_23.67.pth  \
# --evalD 24-evalD_1408.4.naf_PSNR_105_PSNR_23.67   \
# --eval_in_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --vit_img_size 352  \
# --base_channel 32 \
# --overlap_size 176






# CUDA_VISIBLE_DEVICES=8 \
# python  /root/ShadowChallendge/testing_shadow_vit_wNAF.py  \
# --experiment_name testing_shadow_vit_naf_On24evalD  \
# --model vit_naf    \
# --models_ensemble  False  \
# --inputs_ensemble False  \
# --pre_model_0   /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.3e-4_4x4_fine_tune_fakeD_test/net_epoch_26_PSNR_23.15_test_18.85.pth \
# --pre_model_1  /root/autodl-tmp/SR_1/train_shadow_vit_wNAF_BS.1_PS.1408_LR.4e-4_refine_fakeData_freeze_vit_PSNR_23.15_test_18.85.pth/net_naf_epoch_22_PSNR_23.63_test_18.9_best_23.63.pth  \
# --evalD 24-evalD_1408.4.naf_PSNR__23.63_test_18.9   \
# --eval_in_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --vit_img_size 352  \
# --base_channel 24 \
# --overlap_size 176









# CUDA_VISIBLE_DEVICES=8 \
# python  /root/ShadowChallendge/testing_shadow_vit_wNAF.py  \
# --experiment_name testing_shadow_vit_naf_On24evalD  \
# --model vit_naf    \
# --models_ensemble  False  \
# --inputs_ensemble False  \
# --pre_model_0  /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.3e-4_4x4_fine_tune_fakeD_mix/net_epoch_52_PSNR_23.21_test_19.01.pth  \
# --pre_model_1   /root/autodl-tmp/Shadow_Challendge/training_vit_naf/net\(EMA\)_epoch_524_PSNR_23.91.pth  \
# --evalD 24-evalD_1408.4.naf_PSNR_524_PSNR_23.91   \
# --eval_in_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --vit_img_size 352  \
# --base_channel 32 \
# --overlap_size 176


 

# CUDA_VISIBLE_DEVICES=8 \
# python  /root/ShadowChallendge/testing_shadow_vit_wNAF.py  \
# --experiment_name testing_shadow_vit_naf_On24evalD  \
# --model vit_naf    \
# --models_ensemble  False  \
# --inputs_ensemble False  \
# --evalD 24-evalD_1408.4.naf_PSNR__23.63_test_18.9   \
# --pre_model_0 /root/autodl-tmp/SR_1/train_shadow_vit_PL_BS.3_PS.1408_LR.3e-4_4x4_fine_tune_fakeD_test/net_epoch_26_PSNR_23.15_test_18.85.pth  \
# --pre_model_1  /root/autodl-tmp/SR_1/train_shadow_vit_wNAF_BS.1_PS.1408_LR.3e-4_refine_fakeData_freeze_vit_PSNR_23.15_test_18.85.pth/net_naf_epoch_36_PSNR_23.65_test_18.84_best_23.65.pth   \
# --eval_in_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --eval_gt_path  /root/autodl-tmp/ShadowDatasets/ntire_24_sh_rem_final_test_inp/  \
# --vit_img_size 352  \
# --base_channel 24 \
# --overlap_size 176


# CUDA_VISIBLE_DEVICES=0  python  testing_shadow_vit_wNAF.py  --eval_in_path  ./ntire_24_sh_rem_final_test_inp/  --result_path  ./running_result/  

# CUDA_VISIBLE_DEVICES=0  python  TEST.py  --eval_in_path  ./ntire_24_sh_rem_final_test_inp/  --result_path  ./running_result/  
