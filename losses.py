import torch

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# F21 approximation based off https://projecteuclid.org/download/pdf_1/euclid.aos/1031689021

def F21(a, b, c, x):
    y = y_hat(a, b, c, x)
    r = r21(a, b, c, x, y)
    out = (c**(c-.5))*(r**(-.5))
    out *= (y/a)**a
    out *= ((1-y)/(c-a))**(c-a)
    out *= (1-x*y)**(-b)
    return out


def r21(a, b, c, x, y):
    # y = y_hat(a, b, c, x)
    out = (y**2)/a
    out += ((1-y)**2)/(c-a)
    out -= ((b*x**2)*(y**2)*((1-y)**2))/(((1-x*y)**2)*a*(c-a))
    return out

def y_hat(a, b, c, x):
    t = tau(a, b, c, x)
    return 2*a/(torch.sqrt(t**2 - 4*a*x*(c - b)) - t)

def tau(a, b, c, x):
    return x*(b-a)-c

def betainc(x, a, b):
    f = F21(a, 1-b, a+1, x)
    return (x**a)*f/a

def beta(a, b):
    return torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a+b))

def reg_betainc(x, a, b):
    return betainc(x, a, b)/beta(a, b)

def tcdf(t, v, device = device):
    t = t.to(device)
    v = v.to(device)
    # t = torch.Tensor(t)
    # v = torch.Tensor(t)
    x = v/(t**2+v)
    half = torch.Tensor([1/2]).to(device)
    i = reg_betainc(x, v/2, half)

    p = 1 - i/2
    # where_neg = t < 0
    # p[where_neg] = 1 - p[where_neg]
    # p = 2*(p-.5)

    return p

def fcdf(x, d1, d2, device = device):
    d1 = d1.to(device)
    d2 = d2.to(device)
    x = x.to(device)

    y = d1*x/(d1*x + d2)
    p = reg_betainc(y, d1/2, d2/2)

    return p

def f_loss(x1, x2, y, ignore = .9, epsilon = 1e-4, device = device, min_p = 0):
    if type(x1) is tuple:
        v = x1[0].shape[-1]
        # print(v)
    else:
        v = x1.shape[-1]
    v = torch.Tensor([v]).to(device)

    paired = y == 1

    dist = torch.norm(x1 - x2, p = 2, dim = 1)**2

    # get parameters for f distribution. not sure these are right..
    # d1 = x1.shape[0] - v + 1
    d1 = torch.Tensor([1]).to(device)
    d2 = v

    # compute p-value
    p = reg_betainc(d1*dist/(d1*dist+d2), d1/2, d2/2)
    p = p.to(device)
    p[~paired] = 1 - p[~paired]

    if min_p > 0:
        p[paired] = torch.max(p[paired], torch.Tensor([min_p]).to(device))

    # reject hypothesis that this model explains the data if p is significant
    usage = p < ignore
    # or interpret as 'these values are too extreme for this model to meaningfully optimize'
    # ie let discovered features determine their locations
    # usage = torch.abs(p - .5) > ignore - .5

    paired_loss = torch.sum(p[paired]*usage[paired])
    assert not torch.isnan(paired_loss), str(p[paired])
    not_paired_loss = torch.sum(p[~paired]*usage[~paired])
    assert not torch.isnan(not_paired_loss), str(p[not_paired])

    total_loss = torch.sum(usage[paired])*not_paired_loss + torch.sum(usage[~paired])*paired_loss
    total_loss = total_loss / len(paired)

    total_loss = total_loss / torch.sum(usage)

    assert not torch.isnan(total_loss), str(t)

    total_loss = total_loss.to(device)

    del p
    del usage

    return total_loss