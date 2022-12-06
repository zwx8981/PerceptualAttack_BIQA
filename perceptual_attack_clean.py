import torch
from PIL import Image
import matplotlib.pyplot as plt
import piq
import torchvision.transforms as transforms
import scipy.stats
import numpy as np
from tqdm import tqdm
from CORNIA import CORNIA
from mapping2 import logistic_mapping
from DISTS_pytorch import DISTS
from BaseCNN import BaseCNN
from E2euiqa import E2EUIQA
import pandas as pd
import os
import argparse
import scipy.io as scio
from IQA_pytorch import LPIPSvgg

zwx_seed = 19890801
torch.manual_seed(zwx_seed)
np.random.seed(zwx_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
use_cuda = True
imageset = '../IQA_database/databaserelease2/'

test_transform = transforms.Compose([
    #transforms.Grayscale(),
    transforms.ToTensor(),
])

DISTS_model = DISTS().cuda()
LPIPS_model = LPIPSvgg(channels=3).cuda()

#0:linf 1:ssim 2:lpips 3:dists
def FR_regularizer(config):
    if config.fr_reg == 0:
        regularizer = ''
        method = 'LINF'
    elif config.fr_reg == 1:
        regularizer = piq.ssim
        method = 'SSIM'
    elif config.fr_reg == 2:
        regularizer = LPIPS_model
        method = 'LPIPS'
    if config.fr_reg == 3:
        regularizer = DISTS_model
        method = 'DISTS'

    return regularizer, method

def rgb2gray(x):
    r = x[:, 0, :, :]
    g = x[:, 1, :, :]
    b = x[:, 2, :, :]

    x = 0.2989 * r + 0.587 * g + 0.114 * b
    x = x.unsqueeze(1)

    return x

def jnd_attack_adam(x, model, regularizer, config):
    adv = x.clone().detach()
    ref = x.clone().detach()
    alpha = config.alpha
    cnt = 0

    for i in range(config.num_step):
        adv.requires_grad = True

        if (config.fr_reg != 5) & (config.fr_reg != 6):
            optimizer = torch.optim.Adam([adv], lr=alpha)

        s = quality_prediction(adv, model, config.quality_model, seed=i)

        if i == 0:
            s_init = s.detach()
            init_noise = torch.randint(-1, 1, adv.size()).to(adv)
            init_noise = init_noise.float() / 255
            adv = torch.clamp(adv.detach() + init_noise, 0, 1)
            adv.requires_grad = True
            if config.fr_reg != 5:
                optimizer = torch.optim.Adam([adv], lr=alpha)
            s = quality_prediction(adv, model, config.quality_model, seed=i)

        if (config.lamda != 0) & (config.fr_reg != 6):
            if config.fr_reg == 2:  # lpips
                FR_reg = 1 - regularizer(ref, adv, as_loss=True)
            elif config.fr_reg == 3: #dists
                FR_reg = 0
            else:
                FR_reg = regularizer(ref, adv)
        else:
            FR_reg = 0

        obj = -((s - s_init).pow(2) + config.lamda * FR_reg)
        optimizer.zero_grad()

        obj.backward()
        adv.grad.data[torch.isnan(adv.grad.data)] = 0

        adv.grad.data = adv.grad.data / (adv.grad.data.reshape(adv.grad.data.size(0), -1) + 1e-12).norm(dim=1)

        if (config.fr_reg == 0) | (config.fr_reg == 1):
            adv.grad.data[:, :, 0:5, 0:5] = 0
            adv.grad.data[:, :, -5:, -5:] = 0

        optimizer.step()
        adv.data.clamp_(min=0, max=1)
        cnt += 1

    with torch.no_grad():
        ssim_value = piq.ssim(ref, adv)
        dists_value = DISTS_model(ref, adv)
        lpips_value = 1 - LPIPS_model(ref, adv, as_loss=False)
        linf_value = (ref - adv).abs().max()
    fr_value = {'SSIM':ssim_value.item(), 'DISTS':dists_value.item(), 'LPIPS':lpips_value.item(), 'Linf': linf_value.item()}
    return adv, s_init, fr_value

def quality_prediction(x, model, quality_model, seed=zwx_seed):
    if quality_model == 0:
        s = model(x, data_range=1., reduction='none', interpolation='bicubic')
    elif quality_model == 1:
        s = model(x, seed)
        s = - s + 114.4147
    elif quality_model == 2:
        s, _ = model(x)
        #s = s + 3
    elif quality_model == 3:
        s, _ = model(x)

    s = logistic_mapping(s, quality_model)

    return s

def linear_rescale(mos):
    min_mos = 0
    max_mos = 114.4147
    rescaled_mos = min_mos + 10 * (mos - min_mos) / (max_mos - min_mos)
    return rescaled_mos

def query_method(quality_model):
    if quality_model == 0:
        method = 'brisque'
    elif quality_model == 1:
        method = 'cornia'
    elif quality_model == 2:
        method = 'unique'
    elif quality_model == 3:
        method = 'lfc'

    return method

def quantize(x):
    quantizer = transforms.ToPILImage()
    x = quantizer(x.squeeze())
    return x

def do_attack(config, model):
    q_mos = []
    q_hat = []
    o_hat = []
    ssim = []
    dists = []
    lpips = []
    linf = []
    data = pd.read_csv(config.csv_file, sep='\t', header=None)
    cnt = 0
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    regularizer, fr_method = FR_regularizer(config)
    nr_max = 10.0
    nr_min = 0.0
    for idx in tqdm(range(0, len(data))):
        image_name = os.path.join(config.img_dir, data.iloc[idx, 0])
        im = Image.open(image_name)
        im = np.array(im).astype(np.float32)
        x = torch.from_numpy(im).to(device)
        x = x / 255
        x = torch.transpose(x, 0, 2).transpose(1, 2)

        x = x.unsqueeze(0)
        mos = data.iloc[idx, 1]
        mos = linear_rescale(mos)
        q_mos.append(mos)

        if config.attack_trigger == 1:
            adv, ret_s, fr_value = jnd_attack_adam(x, model, regularizer, config)
            adv = torch.round(adv * 255) / 255
            with torch.no_grad():
                sa = quality_prediction(adv, model, config.quality_model)
            o_hat.append(ret_s.item())
        else:
            s = quality_prediction(x, model, config.quality_model)
            sa = s
            fr_value = torch.Tensor([1.])
            o_hat.append(sa.item())

        # q_mos.append(mos)
        if config.attack_trigger == 1:
            ssim.append(fr_value['SSIM'])
            dists.append(fr_value['DISTS'])
            lpips.append(fr_value['LPIPS'])
            linf.append(fr_value['Linf'])
            q_hat.append(sa.item())
        else:
            ssim.append(1.0)
            dists.append(1.0)
            lpips.append(1.0)
            linf.append(0.0)
            q_hat.append(sa.item())

        method_folder = query_method(config.quality_model)

        if config.save_original:
            save_name = str(cnt) + '.png'
            folder = config.original_folder
            if not os.path.exists(folder):
                os.mkdir(folder)
            path = os.path.join(folder, save_name)
            x = quantize(x)
            x.save(path)

        if config.fr_reg == 0:
            folder = os.path.join(config.attack_folder, method_folder, fr_method,
                                  'epsilon_' + str(int(255*config.epsilon)))
        else:
            folder = os.path.join(config.attack_folder, method_folder, fr_method,
                                  'lambda_' + str(config.lamda))

        if not os.path.exists(folder):
            os.makedirs(folder)

        if config.save_attack & config.attack_trigger:
            save_name = str(cnt) + '.png'
            path = os.path.join(folder, save_name)
            adv = quantize(adv)
            adv.save(path)
        cnt += 1

    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
    print('srcc={}'.format(srcc))

    krcc = scipy.stats.kendalltau(x=q_mos, y=q_hat)[0]
    print('krcc={}'.format(krcc))

    plcc = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]
    print('plcc={}'.format(plcc))

    rmse = np.sqrt(((np.array(q_mos) - np.array(q_hat)) ** 2).mean())
    print('rmse={}'.format(rmse))

    srcco = scipy.stats.mstats.spearmanr(x=o_hat, y=q_hat)[0]
    print('srcco={}'.format(srcco))

    krcco = scipy.stats.kendalltau(x=o_hat, y=q_hat)[0]
    print('krcco={}'.format(krcco))

    plcco = scipy.stats.mstats.pearsonr(x=o_hat, y=q_hat)[0]
    print('plcco={}'.format(plcco))

    rmseo = np.sqrt(((np.array(o_hat) - np.array(q_hat)) ** 2).mean())
    print('rmseo={}'.format(rmseo))

    dist2max = nr_max - np.array(q_hat)
    dist2min = np.array(q_hat) - nr_min
    max_diff = np.mean(np.maximum(dist2max, dist2min))
    print('max_diff={}'.format(max_diff))

    mean_ssim = np.mean(np.array(ssim))
    max_ssim = np.max(np.array(ssim))
    min_ssim = np.min(np.array(ssim))

    mean_dists = np.mean(np.array(dists))
    max_dists = np.max(np.array(dists))
    min_dists = np.min(np.array(dists))

    mean_lpips = np.mean(np.array(lpips))
    max_lpips = np.max(np.array(lpips))
    min_lpips = np.min(np.array(lpips))

    mean_linf = np.mean(np.array(linf))
    max_linf = np.max(np.array(linf))
    min_linf = np.min(np.array(linf))

    if config.attack_trigger == 1:
        formatted_output = 'SSIM:' + ' mean={}, max={}, min={}'
        print(formatted_output.format(mean_ssim, max_ssim, min_ssim))
        formatted_output = 'DISTS:' + ' mean={}, max={}, min={}'
        print(formatted_output.format(mean_dists, max_dists, min_dists))
        formatted_output = 'LPIPS:' + ' mean={}, max={}, min={}'
        print(formatted_output.format(mean_lpips, max_lpips, min_lpips))
        formatted_output = 'Linf:' + ' mean={}, max={}, min={}'
        print(formatted_output.format(mean_linf, max_linf, min_linf))

    results = {'srcc':srcc, 'plcc':plcc, 'krcc':krcc, 'rmse':rmse, 'srcco':srcco, 'plcco':plcco, 'krcco':krcco,
               'rmseo':rmseo, 'ssim':ssim, 'max_diff':max_diff, 'dists':dists, 'lpips':lpips, 'linf':linf, 'q_mos':q_mos,
               'q_hat':q_hat, 'o_hat':o_hat}

    if config.eval_step:
        results_path = os.path.join(config.attack_folder, method_folder,
                                    'step_' + str(config.num_step), 'results.mat')
        folder = os.path.join(config.attack_folder, method_folder,
                                    'step_' + str(config.num_step))
        if not os.path.exists(folder):
            os.mkdir(folder)
        scio.savemat(results_path, {'results': results})
    else:
        if config.attack_trigger == 0:
            if not os.path.exists(os.path.join(config.attack_folder, method_folder)):
                os.mkdir(folder)
            results_path = os.path.join(config.attack_folder, method_folder, 'unattacked_results.mat')
        else:
            if config.fr_reg == 5:
                results_path = os.path.join(config.attack_folder, method_folder, fr_method,
                                      'epsilon_' + str(int(255*config.epsilon)), 'results.mat')
            elif config.fr_reg == 6:
                results_path = os.path.join(config.attack_folder, method_folder, fr_method,
                                            'iteration_' + str(int(config.num_step)), 'results.mat')
            else:
                results_path = os.path.join(config.attack_folder, method_folder, fr_method,
                                            'lambda_' + str(config.lamda), 'results.mat')
        scio.savemat(results_path, {'results': results})


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default='resnet34')
    parser.add_argument("--fc", type=bool, default=True)
    parser.add_argument("--network", type=str, default="basecnn") #basecnn or dbcnn
    parser.add_argument("--representation", type=str, default="BCNN")
    parser.add_argument("--std_modeling", type=bool,
                        default=True)  # True for modeling std False for not
    parser.add_argument("--imageset", type=str, default='../IQA_Database/databaserelease2/')
    parser.add_argument("--quality_model", type=int, default=1) ## 0:brisque 1:cornia 2: ma19 3: unique
    parser.add_argument("--num_step", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--lamda", type=int, default=5000)
    parser.add_argument("--original_folder", type=str, default="./original")
    parser.add_argument("--attack_folder", type=str, default="./counterexample")
    parser.add_argument("--fr_reg", type=int, default=0) #0: linf 1: ssim 2: lpips 3: dists
    parser.add_argument("--epsilon", type=int, default=8/255)

    parser.add_argument("--attack_trigger", type=int, default=1)
    parser.add_argument("--save_original", type=bool, default=True)
    parser.add_argument("--save_attack", type=bool, default=False)
    #parser.add_argument("--csv_file", type=str, default='/home/redpanda/codebase/IQA_database/databaserelease2/splits2/1/live_test.txt')
    parser.add_argument("--csv_file", type=str,
                        default='./live_selected_test2.txt')
    parser.add_argument("--img_dir", type=str,
                        default='/home/redpanda/codebase/IQA_Database/databaserelease2')

    parser.add_argument("--eval_step", type=bool,
                        default=False)
    return parser.parse_args()

def main():
    config = parse_config()
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    if config.quality_model == 0:
        BIQA = piq.brisque
    elif config.quality_model == 1:
        BIQA = CORNIA(method_path='./CORNIA')
        BIQA = BIQA.to(device)
    elif config.quality_model == 2:
        BIQA = E2EUIQA()
        BIQA.to(device)
        ckpt = torch.load('f48_max_f128_a9.pt')
        BIQA.load_state_dict(ckpt['state_dict'])
    elif config.quality_model == 3:
        BIQA = BaseCNN(config)
        BIQA = torch.nn.DataParallel(BIQA).cuda()
        ckpt = './UNIQUE.pt'
        checkpoint = torch.load(ckpt)
        BIQA.load_state_dict(checkpoint)
        BIQA.eval()


    lambdas = [0, 0.2, 0.4, 0.6, 0.8, 1, 3, 5, 7, 9, 20, 40, 60, 80, 100, 300, 500, 700, 900, 2000, 4000, 6000, 8000,
              10000, 30000, 50000, 70000, 90000, 200000, 600000, 1000000, 5000000, 9000000]

    for lamda in lambdas:
        if lamda > 0:
            config.save_original = False
        config.lamda = lamda
        if lamda == -1:
            config.attack_trigger = 0
        else:
            config.attack_trigger = 1
        print('attack with lambda = {}'.format(lamda))
        do_attack(config, BIQA)

    ###########################
    # linf
    # epss = [1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255, 8/255, 9/255, 10/255, 11/255, 12/255, 13/255, 14/255,
    #       15/255, 16/255, 17/255, 18/255, 19/255, 20/255, 21/255, 22/255, 23/255, 24/255, 25/255, 26/255, 27/255,
    #        28/255, 29/255, 30/255, 31/255, 32/255, 33/255]
    # for eps in epss:
    #     config.epsilon = eps
    #     config.alpha = eps/config.num_step * 1.5
    #     config.alpha = 0.1
    #     print('attack with eps = {}'.format(eps))
    #     do_attack(config, BIQA)


if __name__ == "__main__":
    main()