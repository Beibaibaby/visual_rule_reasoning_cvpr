import os
import torch
import numpy as np
import argparse
import time
import torch.nn.functional as F
from imgaug import augmenters as iaa
import random
from tqdm import trange
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

from dataset_utility import dataset, ToTensor
from dcnet import DCNet
from skimage.util import random_noise
import pdb

visual=False
label_aug=True
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='dcnet')
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--fig_type', type=str, default='*') 
parser.add_argument('--dataset', type=str, default='RAVEN-F')
parser.add_argument('--root', type=str, default='/localdisk2/RAVEN')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--img_size', type=int, default=96)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=123)

parser.add_argument('--num_label_raven', type=int, default=20000,
                        help='the number of labeled data in RAVEN dataset')
parser.add_argument('--gpu', type=str, default='0', choices=['0', '1','2','3'])
parser.add_argument('--ema_loss_weight', type=float, default=0.1)
parser.add_argument('--k_aug', type=int, default=1)

args = parser.parse_args()
logfile="./expri/"+args.dataset+"/expri_nov_3_number="+str(args.num_label_raven)+"_gpu="+args.gpu+"_mixmatch_rule.txt"
device = torch.device('cuda:'+args.gpu if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available:
    torch.cuda.manual_seed(args.seed)

tf = transforms.Compose([ToTensor()])    
ncols_const = 70

seq = iaa.Sequential([
iaa.Fliplr(1)
])
#Weak Picture Aug


seq_strong = iaa.Sequential([
    iaa.Cutout(nb_iterations=1,size=0.3333)
])



def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if torch.cuda.is_available:
    torch.cuda.manual_seed(args.seed)


def add_noise(inputs):
  
    noise = torch.randn_like(inputs)

    return inputs + noise


tf = transforms.Compose([ToTensor()])    
    
train_set_label = dataset(os.path.join(args.root, args.dataset), 'train', args.fig_type, args.img_size, tf,
                          num_label_raven=args.num_label_raven, is_train=True, train_label=True)

train_set_unlabel = dataset(os.path.join(args.root, args.dataset), 'train', args.fig_type, args.img_size, tf,
                          num_label_raven=args.num_label_raven, is_train=True, train_label=False)

train_set_unlabel2 = dataset(os.path.join(args.root, args.dataset), 'train', args.fig_type, args.img_size, tf,
                          num_label_raven=args.num_label_raven, is_train=True, train_label=False)




valid_set = dataset(os.path.join(args.root, args.dataset), 'val', args.fig_type, args.img_size, tf)
test_set = dataset(os.path.join(args.root, args.dataset), 'test', args.fig_type, args.img_size, tf)

train_loader = DataLoader(train_set_label, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

def data_auger(image_tensor):
    image_tensor = image_tensor.numpy()
    image_tensor = seq(images=image_tensor)
    image_tensor = torch.tensor(image_tensor)
    return image_tensor

def data_auger_strong(image_tensor):
    image_tensor = image_tensor.numpy()
    image_tensor = seq_strong(images=image_tensor)
    image_tensor = torch.tensor(image_tensor)
    return image_tensor



train_unlabel_loader = DataLoader(train_set_unlabel, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
train_unlabel_loader_2 = train_unlabel_loader


valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

save_name = args.model_name + '_' + args.fig_type + '_' + str(args.dim) + '_' + str(args.img_size)

save_path_model = os.path.join('/localdisk2/RAVEN-logs',args.dataset, 'models', save_name)    
if not os.path.exists(save_path_model):
    os.makedirs(save_path_model)    
    
save_path_log = os.path.join(args.dataset, 'logs')    
if not os.path.exists(save_path_log):
    os.makedirs(save_path_log)   
   
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
 
model = DCNet(dim=args.dim).to(device)
model.res1.register_forward_hook(get_activation('res1'))

ema_model = DCNet(dim=args.dim).to(device)
for param in ema_model.parameters():
    param.detach()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

time_now = datetime.now().strftime('%D-%H:%M:%S')      
save_log_name = os.path.join(save_path_log, 'log_{:s}.txt'.format(save_name)) 
with open(save_log_name, 'a') as f:
    f.write('\n------ lr: {:f}, batch_size: {:d}, img_size: {:d}, time: {:s} ------\n'.format(
        args.lr, args.batch_size, args.img_size, time_now))
f.close() 



def contrast_loss(output, target):
    zeros = torch.zeros_like(output)
    zeros.scatter_(1, target.view(-1, 1), 1.0)
        
    return F.binary_cross_entropy_with_logits(output, zeros)


def sharpening(label,T):
    return torch.pow(label,T)/np.sum(torch.pow(label,T))

stack=[]

def train(epoch):
    model.train()    
    ema_model.train()

    counter = 0  # global_step, and the step here is the iteration

    metrics = {'loss': [], 'correct': [], 'count': []}
    
    train_loader_iter = iter(train_loader)
    train_unlabel_loader_iter = iter(train_unlabel_loader)
    #train_unlabel_loader_iter_2 = iter(train_unlabel_loader_2)
    for batch_idx in trange(len(train_loader_iter), ncols=ncols_const):
        
            

        counter += 1

        image, target = next(train_loader_iter)



        if visual:
            imageforshow = image
            imageforshow = imageforshow.numpy()
            print(imageforshow.shape)
            np.save('test.npy', imageforshow)

        image_unlabel, _ = next(train_unlabel_loader_iter)
        image_unlabel_for_c = Variable(image_unlabel, requires_grad=True).to(device)
        model_output_unlabel_for_c = model(image_unlabel_for_c)
        encoded_unlabelled = activation['res1']
        
        
        
        for in_batch_iter in range(args.batch_size):
            first_row=encoded_unlabelled[in_batch_iter*10]
            second_row=encoded_unlabelled[in_batch_iter*10+1]
            stack.append(first_row)
            stack.append(second_row)    
        ###########Add Strong Rule
        for ii in range(args.k_aug):
            for i in range(image.shape[-4]):

                    image[i, :, :, :] = data_auger_strong(image[i, :, :, :])

        #################Add Weak Picture
        if label_aug:
            for i in range(image.shape[-4]):

                if (random.uniform(0, 1) > 0.5):
                    image[i, :, :, :] = data_auger(image[i, :, :, :])
        
        ##############Add Weak Rule
        
        for i in range(image.shape[-4]):
                    key=np.random.randint(3)
                    image_tem=image[i, :, :, :]
                    if key==0:##Change row 1 and row 2
                        image[i, 0, :, :]=image_tem[3,:,:]
                        image[i, 1, :, :]=image_tem[4,:,:]
                        image[i, 2, :, :]=image_tem[5,:,:]
                    elif key==1:#Change row 2 and row 3
                        image[i, 3, :, :]=image_tem[6,:,:]
                        image[i, 4, :, :]=image_tem[7,:,:]
                        image[i, 5, :, :]=image_tem[8,:,:]
                    elif key == 2:
                        image[i, 0, :, :]=image_tem[6,:,:]
                        image[i, 1, :, :]=image_tem[7,:,:]
                        image[i, 2, :, :]=image_tem[8,:,:]                                      
        
        list_output_unlabel=[]
        list_stu_model_output=[]
        list_stu_model_output_strong=[]


        for ii in range(args.k_aug):
            for i in range(image_unlabel.shape[-4]):

                    image_unlabel[i, :, :, :] = data_auger_strong(image_unlabel[i, :, :, :])
            
            for i in range(image_unlabel.shape[-4]):

                if (random.uniform(0, 1) > 0.5):
                    image_unlabel[i, :, :, :] = data_auger(image_unlabel[i, :, :, :])
            
            for i in range(image_unlabel.shape[-4]):
                    key=np.random.randint(3)
                    image_unlabel_tem=image_unlabel[i, :, :, :]
                    if key==0:##Change row 1 and row 2
                        image_unlabel[i, 0, :, :]=image_unlabel_tem[3,:,:]
                        image_unlabel[i, 1, :, :]=image_unlabel_tem[4,:,:]
                        image_unlabel[i, 2, :, :]=image_unlabel_tem[5,:,:]
                    elif key==1:#Change row 2 and row 3
                        image_unlabel[i, 3, :, :]=image_unlabel_tem[6,:,:]
                        image_unlabel[i, 4, :, :]=image_unlabel_tem[7,:,:]
                        image_unlabel[i, 5, :, :]=image_unlabel_tem[8,:,:]
                    elif key == 2:
                        image_unlabel[i, 0, :, :]=image_unlabel_tem[6,:,:]
                        image_unlabel[i, 1, :, :]=image_unlabel_tem[7,:,:]
                        image_unlabel[i, 2, :, :]=image_unlabel_tem[8,:,:]        
                    
            image_unlabel_temp = Variable(image_unlabel, requires_grad=True).to(device)
            stu_model_output = model(image_unlabel_temp)

                
                
            #print('unlablled')
            #print(activation['res1'][0])
            # list_output_unlabel.append(stu_model_output)
            
            if ii == 0:
                sum = stu_model_output
            else:
                sum = sum + stu_model_output
            list_stu_model_output.append(stu_model_output)
    
        image_unlabel=Variable(data_auger(image_unlabel), requires_grad=True).to(device)
        
        
        
        guass_label=sum/args.k_aug
 
        unlabel_loss = 0
        for stu_model_output in list_stu_model_output:
            unlabel_loss += F.mse_loss(guass_label, stu_model_output)
        unlabel_loss=unlabel_loss/args.k_aug
        
        


        image = Variable(image, requires_grad=True).to(device)

        #image_unlabel_2 = Variable(image_unlabel_2, requires_grad=True).to(device)

        target = Variable(target, requires_grad=False).to(device)
        
        predict = model(image)

        #stu_model_output = model(image_unlabel)
        #stu_model_output_2 = model(image_unlabel_2)
        #stu_model_output= (stu_model_output+stu_model_output_2)/2
        #ema_model_output = ema_model(image_unlabel)


        label_loss = contrast_loss(predict, target)

        #unlabel_loss = F.mse_loss(guass_label, ema_model_output)

        loss = label_loss + args.ema_loss_weight * unlabel_loss
        
        encoded_labelled = activation['res1']
        
        pred = torch.max(predict, 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()      
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # update the teacher model
        #update_ema_variables(model, ema_model, 0.999, counter)

        metrics['loss'].append(loss.item())
        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))
        
        accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count']) 

    print ('Training Epoch: {:d}/{:d}, Loss: {:.3f}, Accuracy: {:.3f} \n'.format(
                epoch, args.epochs, np.mean(metrics['loss']), accuracy))
    f = open(logfile, "a")
    f.write('Training Epoch: {:d}/{:d}, Loss: {:.3f}, Accuracy: {:.3f} \n'.format(
                epoch, args.epochs, np.mean(metrics['loss']), accuracy))
    f.close()
            
    return metrics


def validate(epoch):
    model.eval()    
    metrics = {'loss': [], 'correct': [], 'count': []}
            
    valid_loader_iter = iter(valid_loader)
    for _ in trange(len(valid_loader_iter), ncols=ncols_const):
        image, target = next(valid_loader_iter)
        
        image = Variable(image, requires_grad=True).to(device)
        target = Variable(target, requires_grad=False).to(device)

        with torch.no_grad():   
            predict = model(image)        

        loss = contrast_loss(predict, target) 
        pred = torch.max(predict, 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()       
                            
        metrics['loss'].append(loss.item())
        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))

        accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count']) 

    print ('Validation Epoch: {:d}/{:d}, Loss: {:.3f}, Accuracy: {:.3f} \n'.format(
                epoch, args.epochs, np.mean(metrics['loss']), accuracy))
    f = open(logfile, "a")
    f.write('Validation Epoch: {:d}/{:d}, Loss: {:.3f}, Accuracy: {:.3f} \n'.format(
                epoch, args.epochs, np.mean(metrics['loss']), accuracy))
    f.close()
            
    return metrics


def test(epoch):
    model.eval()
    metrics = {'correct': [], 'count': []}
            
    test_loader_iter = iter(test_loader)
    for _ in trange(len(test_loader_iter), ncols=ncols_const):
        image, target = next(test_loader_iter)
        
        image = Variable(image, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)
                
        with torch.no_grad():   
            predict = model(image)    
            
        pred = torch.max(predict, 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        
        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))
        
        accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count']) 

    print ('Testing Epoch: {:d}/{:d}, Accuracy: {:.3f} \n'.format(epoch, args.epochs, accuracy))
    f = open(logfile, "a")
    f.write('Testing Epoch: {:d}/{:d}, Accuracy: {:.3f} \n'.format(epoch, args.epochs, accuracy))
    f.close()
    return metrics


if __name__ == '__main__':
    stack=[] #a empty stack at the begining
    for epoch in range(1, args.epochs+1):
        metrics_train = train(epoch)
        metrics_val = validate(epoch)
        metrics_test = test(epoch)

        # Save model
        if epoch > 0:
            save_name = os.path.join(save_path_model, 'model_{:02d}.pth'.format(epoch))
            torch.save(model.state_dict(), save_name)

        loss_train = np.mean(metrics_train['loss'])
        acc_train = 100 * np.sum(metrics_train['correct']) / np.sum(metrics_train['count']) 
        
        loss_val = np.mean(metrics_val['loss'])
        acc_val = 100 * np.sum(metrics_val['correct']) / np.sum(metrics_val['count']) 

        acc_test = 100 * np.sum(metrics_test['correct']) / np.sum(metrics_test['count']) 
                
        time_now = datetime.now().strftime('%H:%M:%S')            
        with open(save_log_name, 'a') as f:
            f.write('Epoch {:02d}: Accuracy: {:.3f} ({:.3f}, {:.3f}), Loss: ({:.3f}, {:.3f}), Time: {:s}\n'.format(
                epoch, acc_test, acc_train, acc_val, loss_train, loss_val, time_now))
        f.close() 
           