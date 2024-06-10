import torch
import numpy as np
import math

class SGDVAR(torch.optim.Optimizer):
    def __init__(self, params, adaptive=False, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SGDVAR, self).__init__(params, defaults)
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.state['step'] = 0

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.state['step'] += 1
        step = self.state['step']
        if (step + 1) % 352 or step == 1:
            self.second_grad_norm = self._grad_norm()
            self.weight_norm = self._weight_norm()
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            step_size = group['lr']
            momentum = group['momentum']
            rho = group['rho']
            bias_correction1 = 1 - self.beta1 ** step
            bias_correction2 = 1 - self.beta2 ** step
            for p in group['params']:
                if p.grad is None: continue
                param_state = self.state[p]
                
                if 'exp_avg_g' not in param_state:
                    param_state['exp_avg_g'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg_g'].mul_(self.beta1).add_(p.grad, alpha=1 - self.beta1)
                
                noise = p.grad - param_state['exp_avg_g'] / bias_correction1
                if 'exp_avg_var_g' not in param_state:
                    param_state['exp_avg_var_g'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg_var_g'].mul_(self.beta2).addcmul_(noise, noise.conj(), value=1 - self.beta2)
                
                d_p = p.grad.addcmul_(p.grad, param_state['exp_avg_var_g'] * rho / bias_correction2)
                d_p.clamp_(None, 1)
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                    
                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg'].mul_(momentum).add_(d_p)
                
                p.add_(param_state['exp_avg'], alpha=-step_size)
                
        if zero_grad: self.zero_grad()
        
    @torch.no_grad()
    def get_alpha_highest(self, x, alpha=0.9):
        k = int(alpha * x.numel())
        threshold, _ = torch.topk(x.view(-1), k, largest=True, sorted=False)
        return threshold.min().item()
        
    @torch.no_grad()
    def _grad_norm(self, by=None):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if by is None:
            norm = torch.norm(
                        torch.stack([
                            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                )
            return norm
        else:
            norm = torch.norm(
                        torch.stack([
                            ((torch.abs(p) if group["adaptive"] else 1.0) * self.state[p][by]).norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                )
            return norm
    
    @torch.no_grad()
    def _weight_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.data.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
