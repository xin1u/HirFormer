import torch
import numpy as np
import pickle
import cv2
import math

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def splitimage(imgtensor, crop_size=128, overlap_size=64):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts and hstarts[-1] + crop_size >= H:#循环去除不合理的切入点，包含固定点的位置
        hstarts.pop()#移除并返回列表中的最后一个元素
    hstarts.append(H - crop_size)#加入最后一个切入点（固定点）
    #确定w维度的切入点
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts and wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)

    starts = []#记录切入点（hstarts_wstarts）
    split_data = []#记录切成的小图

    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts



def splitimage_fix(imgtensor, crop_size=128, overlap_size=64):
    _, C, H, W = imgtensor.shape
    #确定H维度的切入点
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts and hstarts[-1] + crop_size >= H:#循环去除不合理的切入点
        hstarts.pop()#移除并返回列表中的最后一个元素
    hstarts.append(H - crop_size)
    #确定w维度的切入点
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts and wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)

    starts = []#记录切入点
    split_data = []#记录切成的小图

    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):#遍历图片每个位置求得分
                #此处**2可以适当修改，当作超参数来调整
                #距离图片中心越近的像素获得更高分数
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-6))
    return score

def mergeimage(split_data, starts, crop_size = 128, resolution=(1, 3, 128, 128),is_mean=True):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]#原始大图的维度参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tot_score = torch.zeros((B, C, H, W)).to(device)
    merge_img = torch.zeros((B, C, H, W)).to(device)
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=is_mean).to(device)
    #is_mean可以修改
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart #获取h、w的切入点
        # 将切下来的 小图x权重 按顺序拼成大图
        # print(merge_img.device,scoremap.device)
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        # 将每个像素的权重得分之和也计入大图
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    #最终大图由各小图拼成，重合部分为加权求和
    merge_img = merge_img / tot_score
    # print(merge_img.shape)
    return merge_img


