import os
import cv2
import numpy as np
import json
# 定义文件夹地址
folder_path = '.\Cu'
# 检查文件夹路径是否存在，如果不存在则创建
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# 读取当前文件夹名
folder_name = os.listdir(folder_path)
#提取上级文件名称
parent_folder_name = os.path.basename(os.path.normpath(folder_path))
# 打印上级文件名称
print(parent_folder_name)
# 遍历读取当前文件夹下所有图片
for file in os.listdir(folder_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(os.path.join(folder_path, file))
            #标注图片标签为上级文件名称
            img = cv2.imread(os.path.join(folder_path, folder_name[0]))
            #调整图片大小
            img = cv2.resize(img, (512, 512))
            #保存图片
            cv2.imwrite(os.path.join(folder_path, folder_name[0]), img)
            #将修改后的图像转换为HSV格式
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            #提取颜色特征
            color_features = img_hsv[:, :, 2].flatten()
            # 保存颜色特征
            try:
             np.save(os.path.join(folder_path, parent_folder_name + '_' + folder_name[0] + '_color_features.npy'), color_features)
            except Exception as e:
             print(f"Failed to save color features: {e}")
            #提取纹理特征
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.reshape(gray, (gray.shape[0], gray.shape[1], 1))
            texture_features = gray.flatten()
            # 保存纹理特征
            try:
                np.save(os.path.join(folder_path, parent_folder_name + '_' + folder_name[0] + '_texture_features.npy'), texture_features)
            except Exception as e:
                print(f"Failed to save texture features: {e}")
            # 提取边缘特征
            edges = cv2.Canny(img, 100, 200)
            edge_features = np.zeros((edges.shape[0], edges.shape[1]))
            for i in range(edges.shape[0]):
                for j in range(edges.shape[1]):
                    if edges[i][j] == 255:
                        edge_features[i][j] = 1
            # 保存边缘特征
            try:
                np.save(os.path.join(folder_path, parent_folder_name + '_' + folder_name[0] + '_edge_features.npy'), edge_features)
            except Exception as e:
                print(f"Failed to save edge features: {e}")
            
            # 创建一个空列表来存储转换后的 NumPy 数组
            img_np_list = []

            # 遍历每个图像并将其转换为 NumPy 数组
            for img in img:
              img_np = np.array(img)
              img_np_list.append(img_np)
            #保存文件
            try:
                np.save(os.path.join(folder_path, parent_folder_name + '_' + folder_name[0] + '.npy'), img_np)
            except Exception as e:
                print(f"Failed to save image: {e}")
           
            
#合并颜色特征文件，纹理特征文件，边缘特征文件和图片npy文件
color_features = np.load(os.path.join(folder_path, parent_folder_name + '_' + folder_name[0] + '_color_features.npy'))
texture_features = np.load(os.path.join(folder_path, parent_folder_name + '_' + folder_name[0] + '_texture_features.npy'))
edge_features = np.load(os.path.join(folder_path, parent_folder_name + '_' + folder_name[0] + '_edge_features.npy'))
img_np = np.load(os.path.join(folder_path, parent_folder_name + '_' + folder_name[0] + '.npy'))
# 将三个特征数组进行拼接
final_data = np.concatenate((color_features, texture_features, edge_features), axis=None)
# 保存合并后的npy文件
try:
    np.save(os.path.join(folder_path, parent_folder_name + '_' + folder_name[0] + '_final.npy'), final_data)
except Exception as e:
    print(f"Failed to save final data: {e}")

# 将合并后的npy文件转为numpy数组
final_data = np.load(os.path.join(folder_path, parent_folder_name + '_' + folder_name[0] + '_final.npy'))

# 将numpy数组归一化到0-1之间
final_data = final_data / 255.0

# 将处理后的数据复制到为训练集和测试集
train_data = final_data[:int(len(final_data)*0.8)]
test_data = final_data[int(len(final_data)*0.8):]
       
# 保存训练集和测试集
try:
            np.save(os.path.join(folder_path, parent_folder_name + '_' + folder_name[0] + '_train.npy'), train_data)
except Exception as e:
 print(f"Failed to save train data: {e}")
try:
            np.save(os.path.join(folder_path, parent_folder_name + '_' + folder_name[0] + '_test.npy'), test_data)
except Exception as e:
  print(f"Failed to save test data: {e}")
        
# 打印训练集和测试集
print(train_data)
print(test_data)
# 删除训练集和测试集npy以外的npy文件，除开图片    
for file in os.listdir(folder_path):
            if not (parent_folder_name + '_' + folder_name[0] + '_train.npy') in file and not (parent_folder_name + '_' + folder_name[0] + '_test.npy') in file:
                os.remove(os.path.join(folder_path, file))
