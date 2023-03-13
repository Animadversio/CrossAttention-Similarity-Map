"""
Small lib to calculate RF by back prop towards the image.
It has functions that calculate population RF based on a tensor of weights recombining.
"""
import numpy as np
import torch, torchvision
import matplotlib.pylab as plt
import scipy.optimize as opt
from os.path import join
from easydict import EasyDict
from core.utils.CNN_scorers import get_activation, activation
from core.utils.layer_hook_utils import register_hook_by_module_names, get_module_names
import torch.nn.functional as F

def grad_RF_estimate(model, target_layer, target_unit, input_size=(3,227,227),
                     device="cuda", show=True, reps=200, batch=1):
    # (slice(None), 7, 7)
    handle, module_names, module_types = register_hook_by_module_names(target_layer,
        get_activation("record", target_unit, ingraph=True), model,
        input_size, device=device, )

    cnt = 0
    # graddata = torch.zeros((1, 3, 227, 227)).cuda()
    gradabsdata = torch.zeros(input_size).cuda()
    for i in range(reps):
        intsr = torch.rand((batch, *input_size)).cuda() * 2 - 1
        intsr.requires_grad_(True)
        model(intsr)
        act_vec = activation['record']
        if act_vec.numel() > 1:
            act = act_vec.mean()
        else:
            act = act_vec
        if not torch.isclose(act, torch.tensor(0.0)):
            act.backward()
            # graddata += intsr.grad
            gradabsdata += intsr.grad.abs().mean(dim=0)
            cnt += 1
        else:
            continue
    if cnt == 0:
        raise ValueError("Unit Not activated by random noise")
    for h in handle:
        h.remove()
    gradAmpmap = gradabsdata.permute([1, 2, 0]).abs().mean(dim=2).cpu() / cnt
    if show:
        plt.figure(figsize=[6, 6.5])
        plt.pcolor(gradAmpmap)
        plt.gca().invert_yaxis()
        plt.axis("image")
        plt.title("L %s Unit %s"%(target_layer, target_unit))
        plt.show()
        plt.figure(figsize=[6, 6.5])
        gradAmpmap = torch.nan_to_num(gradAmpmap, nan=0.0, posinf=0.0, neginf=0.0)
        plt.hist(np.log10(1E-15 + gradAmpmap.flatten().cpu().numpy()), bins=100)
        plt.xlabel("log10(gradAmp) histogram")
        plt.title("L %s Unit %s"%(target_layer, target_unit))
        plt.show()
    activation.pop('record')
    del intsr, act_vec
    return gradAmpmap.numpy()


def GAN_grad_RF_estimate(G, model, target_layer, target_unit, input_size=(3,227,227),
                     device="cuda", show=True, reps=200, batch=1):
    # (slice(None), 7, 7)
    handle, module_names, module_types = register_hook_by_module_names(target_layer,
        get_activation("record", target_unit, ingraph=True), model,
        input_size, device=device, )

    cnt = 0
    # graddata = torch.zeros((1, 3, 227, 227)).cuda()
    gradabsdata = torch.zeros(input_size).cuda()
    for i in range(reps):
        intsr = G.visualize(torch.randn(batch, 4096).cuda()) * 2 - 1
        intsr = F.interpolate(intsr, size=input_size[1:], mode='bilinear', align_corners=True)
        # intsr = torch.rand((batch, *input_size)).cuda() * 2 - 1
        intsr.requires_grad_(True)
        model(intsr)
        act_vec = activation['record']
        if act_vec.numel() > 1:
            act = act_vec.mean()
        else:
            act = act_vec
        if not torch.isclose(act, torch.tensor(0.0)):
            act.backward()
            # graddata += intsr.grad
            gradabsdata += intsr.grad.abs().mean(dim=0)
            cnt += 1
        else:
            continue

    for h in handle:
        h.remove()
    gradAmpmap = gradabsdata.permute([1, 2, 0]).abs().mean(dim=2).cpu() / cnt
    if show:
        plt.figure(figsize=[6, 6.5])
        plt.pcolor(gradAmpmap)
        plt.gca().invert_yaxis()
        plt.axis("image")
        plt.title("L %s Unit %s"%(target_layer, target_unit))
        plt.show()
        plt.figure(figsize=[6, 6.5])
        gradAmpmap = torch.nan_to_num(gradAmpmap, nan=0.0, posinf=0.0, neginf=0.0)
        plt.hist(np.log10(1E-15 + gradAmpmap.flatten().cpu().numpy()), bins=100)
        plt.xlabel("log10(gradAmp) histogram")
        plt.title("L %s Unit %s"%(target_layer, target_unit))
        plt.show()
    activation.pop('record')
    del intsr, act_vec
    return gradAmpmap.numpy()


def grad_population_RF_estimate(model, target_layer, target_layer_weights, input_size=(3,227,227),
                     device="cuda", show=True, reps=200, batch=1, label="", figdir=None):
    # (slice(None), 7, 7)
    handle, module_names, module_types = register_hook_by_module_names(target_layer,
        get_activation("record", unit=None, ingraph=True), model,
        input_size, device=device, )

    cnt = 0
    # graddata = torch.zeros((1, 3, 227, 227)).cuda()
    gradabsdata = torch.zeros(input_size).cuda()
    for i in range(reps):
        intsr = torch.rand((batch, *input_size)).cuda() * 2 - 1
        intsr.requires_grad_(True)
        model(intsr)
        act_tsr = activation['record']
        act_vec = (act_tsr * target_layer_weights).flatten(start_dim=1).sum(dim=1)
        if act_vec.numel() > 1:
            act = act_vec.sum()
        else:
            act = act_vec
        if not torch.isclose(act, torch.tensor(0.0)):
            act.backward()
            # graddata += intsr.grad
            gradabsdata += intsr.grad.abs().mean(dim=0)
            cnt += 1
        else:
            continue

    for h in handle:
        h.remove()
    gradAmpmap = gradabsdata.permute([1, 2, 0]).abs().mean(dim=2).cpu() / cnt
    if show:
        plt.figure(figsize=[6, 6.5])
        plt.pcolor(gradAmpmap)
        plt.gca().invert_yaxis()
        plt.axis("image")
        plt.title("L %s Weighted sum %s"%(target_layer, label))
        if figdir is not None:
            plt.savefig(join(figdir, f"{label}_rawgradAmpMap.png"))
        plt.show()
        plt.figure(figsize=[6, 6.5])
        plt.hist(np.log10(1E-15 + gradAmpmap.flatten().cpu().numpy()), bins=100)
        plt.xlabel("log10(gradAmp) histogram")
        plt.title("L %s Weighted sum %s"%(target_layer, label))
        if figdir is not None:
            plt.savefig(join(figdir, f"{label}_gradAmp_loghist.png"))
        plt.show()
    activation.pop('record')
    del intsr, act_vec
    return gradAmpmap.numpy()


def gradmap2RF_square(gradAmpmap, absthresh=None, relthresh=0.01, square=True):
    maxAct = gradAmpmap.max()
    relthr = maxAct * relthresh
    if absthresh is None:
        thresh = relthr
    else:
        thresh = max(relthr, absthresh)
    Yinds, Xinds = np.where(gradAmpmap > thresh)
    Xlim = (Xinds.min(), Xinds.max()+1)
    Ylim = (Yinds.min(), Yinds.max()+1)
    if square:
        Xrng = Xlim[1] - Xlim[0]
        Yrng = Ylim[1] - Ylim[0]
        if Xrng == Yrng:
            pass
        elif Xrng > Yrng:
            print("Modify window to be square, before %s, %s"%(Xlim, Ylim))
            incre = (Xrng - Yrng) // 2
            Ylim = (Ylim[0] - incre, Ylim[1] + (Xrng - Yrng - incre))
            if Ylim[1] > gradAmpmap.shape[0]:
                offset = Ylim[1] - gradAmpmap.shape[0]
                Ylim = (Ylim[0] - offset, Ylim[1] - offset)
            if Ylim[0] < 0:
                offset = 0 - Ylim[0]
                Ylim = (Ylim[0] + offset, Ylim[1] + offset)
            print("After %s, %s"%(Xlim, Ylim))
        elif Yrng > Xrng:
            print("Modify window to be square, before %s, %s" % (Xlim, Ylim))
            incre = (Yrng - Xrng) // 2
            Xlim = (Xlim[0] - incre, Xlim[1] + (Yrng - Xrng - incre))
            if Xlim[1] > gradAmpmap.shape[1]:
                offset = Xlim[1] - gradAmpmap.shape[1]
                Xlim = (Xlim[0] - offset, Xlim[1] - offset)
            if Xlim[0] < 0:
                offset = 0 - Xlim[0]
                Xlim = (Xlim[0] + offset, Xlim[1] + offset)
            print("After %s, %s" % (Xlim, Ylim))
    return Xlim, Ylim


def twoD_Gaussian(XYstack, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    # From https://stackoverflow.com/a/21566831
    xo = float(xo)
    yo = float(yo)
    x = XYstack[0]
    y = XYstack[1]
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()


def fit_2dgauss(gradAmpmap_, pop_str, outdir="", plot=True):
    if isinstance(gradAmpmap_, torch.Tensor):
        gradAmpmap_ = gradAmpmap_.numpy()

    H, W = gradAmpmap_.shape
    YY, XX = np.meshgrid(np.arange(H), np.arange(W))
    Xcenter = (gradAmpmap_ * XX).sum() / gradAmpmap_.sum()
    Ycenter = (gradAmpmap_ * YY).sum() / gradAmpmap_.sum()
    XXVar = (gradAmpmap_ * (XX - Xcenter) ** 2).sum() / gradAmpmap_.sum()
    YYVar = (gradAmpmap_ * (YY - Ycenter) ** 2).sum() / gradAmpmap_.sum()
    XYCov = (gradAmpmap_ * (XX - Xcenter) * (YY - Ycenter)).sum() / gradAmpmap_.sum()
    print(f"Gaussian Fitting center ({Xcenter:.1f}, {Ycenter:.1f})\n"
          f" Cov mat XX {XXVar:.1f} YY {YYVar:.1f} XY {XYCov:.1f}")
    #% covariance

    # MLE estimate? not good... Not going to use
    # covmat = torch.tensor([[XXVar, XYCov], [XYCov, YYVar]]).double()
    # precmat = torch.linalg.inv(covmat)
    # normdensity = torch.exp(-((XX - Xcenter)**2*precmat[0, 0] +
    #                           (YY-Ycenter)**2*precmat[1, 1] +
    #                           2*(XX - Xcenter)*(YY - Ycenter)*precmat[0, 1]))
    # var = multivariate_normal(mean=torch.tensor([Xcenter, Ycenter]), cov=covmat)
    # xystack = np.dstack((xplot, yplot))
    # densitymap = var.pdf(xystack)

    # curve fitting , pretty good.
    xplot, yplot = np.mgrid[0:H:1, 0:W:1]
    initial_guess = (gradAmpmap_.max().item(),
                     Xcenter.item(), Ycenter.item(),
                     np.sqrt(XXVar).item()/4, np.sqrt(YYVar).item()/4,
                     0, 0)  # 5, 5, 0, 0)
    # -np.inf, np.inf
    # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
    LB = [0, 0, 0, 0, 0, -2*np.pi, -np.inf]
    UB = [np.inf, H, W, 2*H, 2*W, 2*np.pi, np.inf]
    popt, pcov = opt.curve_fit(twoD_Gaussian, np.stack((xplot, yplot)).reshape(2, -1),
                               gradAmpmap_.reshape(-1), p0=initial_guess,
                               maxfev=10000, bounds=(LB, UB))
    ffitval = twoD_Gaussian(np.stack((xplot, yplot)).reshape(2, -1),
                            *popt).reshape(H, W)
    amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt
    fitdict = EasyDict(popt=popt, amplitude=amplitude, xo=xo, yo=yo,
            sigma_x=sigma_x, sigma_y=sigma_y, theta=theta, offset=offset,
            gradAmpmap=gradAmpmap_, fitmap=ffitval)
    np.savez(join(outdir, f"{pop_str}_gradAmpMap_GaussianFit.npz"),
            **fitdict)
    if plot:
        plt.figure(figsize=[5.8, 5])
        plt.imshow(ffitval)
        plt.colorbar()
        plt.title(f"{pop_str}\n"
                  f"Ampl {amplitude:.1e} Cent ({xo:.1f}, {yo:.1f}) std: ({sigma_x:.1f}, {sigma_y:.1f})\n Theta: {theta:.2f}, Offset: {offset:.1e}", fontsize=14)
        plt.tight_layout()
        plt.savefig(join(outdir, f"{pop_str}_gradAmpMap_GaussianFit.png"))
        plt.savefig(join(outdir, f"{pop_str}_gradAmpMap_GaussianFit.pdf"))
        plt.show()

        figh, axs = plt.subplots(1,2,figsize=[9.8, 6])
        axs[0].imshow(gradAmpmap_)
        # plt.colorbar(axs[0])
        axs[1].imshow(ffitval)
        # plt.colorbar(axs[1])
        plt.suptitle(f"{pop_str}\n"
                  f"Ampl {amplitude:.1e} Cent ({xo:.1f}, {yo:.1f}) std: ({sigma_x:.1f}, {sigma_y:.1f})\n Theta: {theta:.2f}, Offset: {offset:.1e}", fontsize=14)
        plt.tight_layout()
        plt.savefig(join(outdir, f"{pop_str}_gradAmpMap_GaussianFit_cmp.png"))
        plt.savefig(join(outdir, f"{pop_str}_gradAmpMap_GaussianFit_cmp.pdf"))
        plt.show()
    return fitdict


def show_gradmap(gradAmpmap, ):
    plt.figure(figsize=[5.8, 5])
    plt.imshow(gradAmpmap)
    plt.colorbar()
    # plt.title(f"{pop_str}", fontsize=14)
    # plt.savefig(join(rfdir, f"{pop_str}_gradAmpMap.png"))
    plt.show()



if __name__ == "__main__":
    from collections import OrderedDict
    resnet101 = torchvision.models.resnet101(pretrained=True).cuda()
    for param in resnet101.parameters():
        param.requires_grad_(False)
    resnet101.eval()
    # module_names, module_types = get_module_names(resnet101, (3,227,227))
    resnet_feat = torch.nn.Sequential(OrderedDict({"conv1": resnet101.conv1,
                                                   "bn1": resnet101.bn1,
                                                   "relu": resnet101.relu,
                                                   "maxpool": resnet101.maxpool,
                                                   "layer1": resnet101.layer1,
                                                   "layer2": resnet101.layer2,
                                                   "layer3": resnet101.layer3,
                                                   "layer4": resnet101.layer4}))
    unit_list = [("resnet101", ".ReLUrelu", 5, 56, 56),
                 ("resnet101", ".layer1.Bottleneck0", 5, 28, 28),
                 ("resnet101", ".layer1.Bottleneck1", 5, 28, 28),
                 ("resnet101", ".layer2.Bottleneck0", 5, 14, 14),
                 ("resnet101", ".layer2.Bottleneck3", 5, 14, 14),
                 ("resnet101", ".layer3.Bottleneck0", 5, 7, 7),
                 ("resnet101", ".layer3.Bottleneck2", 5, 7, 7),
                 ("resnet101", ".layer3.Bottleneck6", 5, 7, 7),
                 ("resnet101", ".layer3.Bottleneck10", 5, 7, 7),
                 ("resnet101", ".layer3.Bottleneck14", 5, 7, 7),
                 ("resnet101", ".layer3.Bottleneck18", 5, 7, 7),
                 ("resnet101", ".layer3.Bottleneck22", 5, 7, 7), ]
    for unit in unit_list:
        print("Unit %s" % (unit,))
        gradAmpmap = grad_RF_estimate(resnet_feat, unit[1], (slice(None), unit[3], unit[4]), input_size=(3, 227, 227),
                                      device="cuda", show=True, reps=40, batch=1)
        Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
        print("Xlim %s Ylim %s\nimgsize %s corner %s" % (
        Xlim, Ylim, (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0]), (Xlim[0], Ylim[0])))