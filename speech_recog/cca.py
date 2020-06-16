from torch.autograd import Function
import torch
import numpy as np
from scipy import optimize
from numpy.linalg import norm, svd, inv, eig


class CCA(Function):
    """ An autograd supported function for compute the correlation among inputs"""
    @staticmethod
    def forward(ctx, X, Y, config):
        """ compute the correlation and gradients w.r.t inputs
        Args:
            X, Y - sample * features which represent the data from view1 and view2 respectively.
            r1, r2 - regularization parameters for view1/2.
            proj_k - dimensionality of subspace.
        Returns:
            corr - the correlation coefficience between X&Y.
        """
        r1 = 1e-4
        r2 = 1e-4
        proj_k = config['calc_k']
        n_sample, n_feat = X.shape
        I = torch.eye(n_feat, device=config['device'])

        xbar = X - X.mean(0, keepdim=True)
        ybar = Y - Y.mean(0, keepdim=True)

        sigmaxy = xbar.t() @ ybar * (1./(n_sample-1))
        sigmaxx = xbar.t() @ xbar * (1./(n_sample-1)) + I*r1
        sigmayy = ybar.t() @ ybar * (1./(n_sample-1)) + I*r2

        ux, sx, vx = torch.svd(sigmaxx)
        uy, sy, vy = torch.svd(sigmayy)

        sigmaxx_inv = ux @ torch.diag(torch.rsqrt(sx)) @ vx.t()
        sigmayy_inv = uy @ torch.diag(torch.rsqrt(sy)) @ vy.t()
        T = sigmaxx_inv @ sigmaxy @ sigmayy_inv
        u, s, v = torch.svd(T)
        u = u[:, :proj_k]
        s = s[:proj_k]
        v = v[:, :proj_k]

        corr = s.sum()

        deltaxx = - sigmayy_inv @ u @ torch.diag(s) @ u.t() @ sigmaxx_inv * .5
        deltaxy = sigmaxx_inv @ u @ v.t() @ sigmayy_inv
        deltayy = - sigmayy_inv @ v @ torch.diag(s) @ v.t() @ sigmayy_inv * .5

        gradx = (2 * xbar @deltaxx + ybar @ deltaxy.t())* (1./(n_sample-1))
        grady = (xbar @ deltaxy + 2 * ybar @ deltayy) * (1./(n_sample-1))
        ctx.save_for_backward(gradx, grady)

        return corr

    @staticmethod
    def backward(ctx, grad_output):
        gradx, grady = ctx.saved_tensors
        grad_out = grad_output.clone()
        return grad_out*gradx, grad_out*grady, None

def optsolver(h, alpha=.1, k=20):
    '''
    function to solve proximal mapping optimization problem.

    Args:
        h (matrix): it consists of [hx, hy] vertically.
        alpha (float): trade-off parameter.
        k (int): dimensionaly of the subspace.
    
    Returns:
        obj (float): objective function value. CCA value.
        grad (matrix): gradient w.r.t hx and hy
    '''
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.numpy()
        k = k.numpy()
    hx, hy = torch.split(h, h.shape[0]//2, dim=0)
    hx = hx.numpy()
    hy = hy.numpy()
    N, d = hx.shape
    def foo(x, alpha=alpha, k=k):
        r1 = r2 = 1e-4
        K = k

        X = x[:N*d].reshape((N, d))
        Y = x[N*d:].reshape((N, d))

        Xbar = X - np.mean(X,0, keepdims=True)
        Ybar = Y - np.mean(Y,0, keepdims=True)

        sigmaXY = Xbar.T @ Ybar * (1.0/(N-1))
        sigmaXX = Xbar.T @ Xbar * (1.0/(N-1)) + np.eye(d)*r1
        sigmaYY = Ybar.T @ Ybar * (1.0/(N-1)) + np.eye(d)*r2

        sx, vx = eig(sigmaXX)
        sy, vy = eig(sigmaYY)

        sigmaXXinv = vx @ np.diag(sx**-.5) @ vx.T
        sigmaYYinv = vy @ np.diag(sy**-.5) @ vy.T

        T = sigmaXXinv @ sigmaXY @ sigmaYYinv

        u, s, v = svd(T)

        obj = (.5*norm(X-hx)**2 + .5*norm(Y - hy)**2)/N*alpha - s[:K].sum()
        # print((.5*norm(X-hx)**2 + .5*norm(Y - hy)**2)/N, s[:K].sum()*alpha)

        DeltaXX = -sigmaXXinv @ u @ np.diag(s) @ u.T @ sigmaXXinv *.5
        DeltaXY = sigmaXXinv @ u @ v @ sigmaYYinv
        DeltaYY = -sigmaYYinv @ v.T @ np.diag(s) @ v @ sigmaYYinv *.5
        
        grad_x = (X - hx)/N*alpha - (2*Xbar @ DeltaXX +  Ybar @ DeltaXY.T) * (1.0/(N-1))
        grad_y = (Y - hy)/N*alpha - (2*Ybar @ DeltaYY +  Xbar @ DeltaXY) * (1.0/(N-1))

        grad = np.concatenate((grad_x.flatten(), grad_y.flatten()), axis=0)
        return obj, grad

    x0 = h
    opts = optimize.fmin_l_bfgs_b(foo, x0)
    x = opts[0]
    X = x[:N*d].reshape((N, d))
    Y = x[N*d:].reshape((N, d))
    out = np.concatenate((X, Y), 0)
    return torch.from_numpy(out.astype(np.float32))

class opt_layer(Function):
    '''
    proximal mapping layer. 
    Use implicity differentiation method to approximate the gradient
    for backpropagation.
    '''
    @staticmethod
    def forward(ctx, h, alpha=.01, k=20, device='cuda'):
        ctx.alpha = alpha
        ctx.k = k
        ctx.device = device
        ctx.save_for_backward(h.cpu().detach())

        out = optsolver(h.cpu().detach(), alpha=alpha, k=k)
        out = torch.as_tensor(out).cuda()

        return out

    @staticmethod
    def backward(ctx, grad_output):
        h = ctx.saved_tensors[0]
        alpha = ctx.alpha
        k = ctx.k
        device = ctx.device
        grad_out = grad_output.detach().cpu().clone()
        r = (1+torch.norm(h, float('inf'))) / torch.norm(grad_out,  float('inf')) * 1e-2

        mu_plus_inp = h + r*grad_out
        mu_plus = optsolver(mu_plus_inp, alpha, k)

        mu_minus_inp = h - r*grad_out
        mu_minus = optsolver(mu_minus_inp, alpha, k)

        grad = (mu_plus - mu_minus) * (1./(2*r))
        return torch.as_tensor(grad).cuda(device), None, None, None