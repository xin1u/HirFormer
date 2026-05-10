      
from torchvision import transforms
import torch
from PIL import Image
import os
# from MaeVit_arch import MaskedAutoencoderViT
from functools import partial
from networks.image_utils import splitimage, mergeimage
def split_image(image_input, grid_type):
    '''
    分割输入大图为各种小块
    '''

    image = image_input

    # 定义切割函数
    def split_image(image, rows, cols):
        n ,c, h, w = image.shape
        h_stride, w_stride = h // rows, w // cols
        images = []
        positions = []
        for i in range(rows):
            for j in range(cols):
                sub_image = image[:,:, i * h_stride:(i + 1) * h_stride, j * w_stride:(j + 1) * w_stride]
                images.append(sub_image)
                positions.append((i * h_stride, j * w_stride, h_stride, w_stride))
        return images, positions

    # 根据grid_type切割图像
    if grid_type == '2x2':
        return split_image(image, 2, 2)
    elif grid_type == '1x2':
        return split_image(image, 1, 2)
    elif grid_type == '1x3':
        return split_image(image, 1, 3)
    elif grid_type == '1x4':
        return split_image(image, 1, 4)
    elif grid_type == '2x1':
        return split_image(image, 2, 1)
    elif grid_type == '3x1':
        return split_image(image, 3, 1)
    elif grid_type == '4x1':
        return split_image(image, 4, 1)
    elif grid_type == '4x4':
        return split_image(image, 4, 4)
    elif grid_type == '8x8':
        return split_image(image, 8, 8)
    elif grid_type == '16x16':
        return split_image(image, 16, 16)
    else:
        raise ValueError("Unsupported grid type")


# 

def merge(sub_images, positions):
    '''
    将分割的小图合成大图
    '''

    # 获取子图像数量和通道数
    num_sub_images = len(sub_images)
    n, c, h, w = sub_images[0].shape
    # print(c,h,w,positions) #调试用

    # 计算还原后的图像尺寸
    max_h = max(pos[0] + pos[2] for pos in positions)
    max_w = max(pos[1] + pos[3] for pos in positions)
    

    # 创建一个空的大图像张量
    image = torch.zeros((sub_images[0].size()[0],sub_images[0].size()[1], max_h, max_w))
    
    # 将子图放置在大图上
    for img, pos in zip(sub_images, positions):
        x, y , h ,w= pos
        image[:,:, x:x + h, y:y + w] = img

    # print('max_h',max_h,'max_w',max_w ,image.shape)
    # 裁剪图像到原始尺寸
    # image = image[:, :max_h, :max_w]

    return image

def process_split_image_with_model(sub_images,model): #image_path, grid_config, model):
    # sub_images = preprocess_image(image_path, grid_config)
    '''
    用 model处理sub_image(顺序前向传播)
    '''

    processed_sub_images = [model(sub_image) for sub_image in sub_images]  # 模型期望的输入是[batch_size, C, H, W]

    return processed_sub_images

def process_split_image_with_model_1(net,net_0,outputs,name,inputs): #image_path, grid_config, model):
    # sub_images = preprocess_image(image_path, grid_config)
    '''
    用 model处理sub_image(顺序前向传播)
    '''
    if name:
        split_data, starts = splitimage(inputs, 352, 176)#352_4x4
        # print (name)
        for i, data in enumerate(split_data):             
            # 获得输出的小图
            output = net(data)
            output = net_0(output)
            split_data[i] = output
        # 获得输出的大图            
        output = mergeimage(split_data, starts, 352, resolution=(1,3,1440,1920),is_mean=False)
        outputs = output

    return outputs

def process_split_image_with_model_parallel(sub_images,model): #image_path, grid_config, model):
    # sub_images = preprocess_image(image_path, grid_config)
    '''
    用 model处理sub_image(将子图列表长度 变成batch_size 维度,并行前向传播)
    '''
    n ,c, h, w = sub_images[0].shape
    L= len(sub_images)
    # print('L,n,c,h,w',L,n,c,h,w)
    merged_tensor = torch.stack(sub_images, dim=0)
    # print('merged_tensor',merged_tensor.shape)
    reshaped_tensor = merged_tensor.view(n*L, c, h, w)  # 将子图列表长度 变成batch_size 维度
    # print('reshaped_tensor',reshaped_tensor.shape)
    processed_sub_images = model(reshaped_tensor)       # 模型期望的输入是 batch_size, C, H, W
    # print('processed_sub_images',processed_sub_images.shape)
    image = processed_sub_images.view(L,n,c,h,w)        # 将四张图分开,在新的维度堆叠 num, batch_size, C, H, W
    # print('image',image.shape)
    # images = list(image.split(1, dim=0))
    images = [image[i] for i in range(L)]               # 将四张图恢复成列表形式
    return images



def preprocess_image(image_path, grid_type):
    '''
    分割大图像为各种小块
    '''
    # 加载图像
    image = Image.open(image_path)
    # 转换图像为Tensor
    transform = transforms.ToTensor()
    image = transform(image)

    # 定义切割函数
    def split_image(image, rows, cols):
        c, h, w = image.shape
        h_stride, w_stride = h // rows, w // cols
        return [image[:, i * h_stride:(i + 1) * h_stride, j * w_stride:(j + 1) * w_stride] for i in range(rows) for j in
                range(cols)]

    # 根据grid_type切割图像
    if grid_type == '2x2':
        return split_image(image, 2, 2)
    elif grid_type == '1x2':
        return split_image(image, 1, 2)
    elif grid_type == '1x3':
        return split_image(image, 1, 3)
    elif grid_type == '1x4':
        return split_image(image, 1, 4)
    elif grid_type == '2x1':
        return split_image(image, 2, 1)
    elif grid_type == '3x1':
        return split_image(image, 3, 1)
    elif grid_type == '4x1':
        return split_image(image, 4, 1)
    else:
        raise ValueError("Unsupported grid type")

import torch.nn as nn

def process_image_with_model(image_path, grid_config, model):
    sub_images = preprocess_image(image_path, grid_config)
    processed_sub_images = [model(sub_image.unsqueeze(0)) for sub_image in sub_images]  # 模型期望的输入是[batch_size, C, H, W]
    return processed_sub_images

class DynamicInputConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)#out_put_size = (input_size + 2 * padding_size - kernel_size) / stride_size + 1
        self.pool = nn.MaxPool2d(1, 1)#kernel_size=2, stride=2
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x

