import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable as V
from PIL import Image
import torch

from model import Generator
from model import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
#from utils import weights_init_normal
from datasets import ImageData

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--batchsize', type=int, default=1, help='Batch size')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra', help='Root dir of dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate')
parser.add_argument('--decay-epoch', type=int, default=100, help='Epoch to start linearly decrease learning rate')
parser.add_argument('--size', type=int, default=100, help='Size of data crop')
parser.add_argument('--input-nc', type=int, default=3, help='Number of input channels')
parser.add_argument('--output-nc', type=int, default=3, help='Number of output channels')
parser.add_argument('--cuda', action='store_true', help='Use GPU')
parser.add_argument('--n-cpu', type=int, default=8, help='Number of CPU threads to use')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print('WARNING: This is a GAN. You have GPU. Give it a try. Use --cuda')

gen_A2B = Generator(opt.input_nc, opt.output_nc)
gen_B2A = Generator(opt.output_nc, opt.input_nc)
dis_A = Discriminator(opt.input_nc)
dis_B = Discriminator(opt.output_nc)

if opt.cuda:
    gen_A2B.cuda()
    gen_B2A.cuda()
    dis_A.cuda()
    dis_B.cuda()
"""
gen_A2B.apply(weights_init_normal)
gen_B2A.apply(weights_init_normal)
dis_A.apply(weights_init_normal)
dis_B.apply(weights_init_normal)
"""
#Losses:
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers and learning rate scheduler:
optim_G = torch.optim.Adam(itertools.chain(gen_A2B.parameters(), gen_B2A.parameters()), 
                            lr = opt.lr, betas=(0.5, 0.999))
optim_DA = torch.optim.Adam(dis_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optim_DB = torch.optim.Adam(dis_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_DA = torch.optim.lr_scheduler.LambdaLR(optim_DA, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_DB = torch.optim.lr_scheduler.LambdaLR(optim_DB, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs and targets memory allocation:
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
inputA = Tensor(opt.batchsize, opt.input_nc, opt.size, opt.size)
inputB = Tensor(opt.batchsize, opt.output_nc, opt.size, opt.size)
target_real = V(Tensor(opt.batchsize).fill_(1.0), requires_grad=False)
target_fake = V(Tensor(opt.batchsize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset Loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                transforms.RandomCrop(opt.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]

dataloader = DataLoader(ImageData(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchsize, shuffle=True, num_workers=opt.n_cpu)

# Loss Plot:
logger = Logger(opt.n_epochs, len(dataloader))

# Training
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input:
        real_A = V(inputA.copy_(batch['A']))
        real_B = V(inputB.copy_(batch['B']))

        optim_G.zero_grad()

        # Identity loss
        # G_A2B(B) should be equal to B if real B is fed
        same_B = gen_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        same_A = gen_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN Loss:
        fake_B = gen_A2B(real_A)
        pred_fake = dis_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
        fake_A = gen_B2A(real_B)
        pred_fake = dis_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle Loss:
        recovered_A = gen_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0
        recovered_B = gen_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # Total loss:
        loss_G = loss_identity_A + loss_identity_B + loss_cycle_ABA + loss_cycle_BAB + loss_GAN_A2B + loss_GAN_B2A
        loss_G.backward()

        optim_G.step()

        ###### Discriminator A ######
        optim_DA.zero_grad()

        # Real loss
        pred_real = dis_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = dis_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optim_DA.step()

        ###### Discriminator B ######
        optim_DB.zero_grad()

        # Real loss
        pred_real = dis_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = dis_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optim_DB.step()

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_DA.step()
    lr_scheduler_DB.step()

    # Save models checkpoints
    torch.save(gen_A2B.state_dict(), 'output/gen_A2B.pth')
    torch.save(gen_B2A.state_dict(), 'output/gen_B2A.pth')
    torch.save(dis_A.state_dict(), 'output/dis_A.pth')
    torch.save(dis_B.state_dict(), 'output/dis_B.pth')
    