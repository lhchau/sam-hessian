import torch
import numpy as np


class SAME(torch.optim.Optimizer):
    def __init__(self, params, rho=0.05, adaptive=False, condition=1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAME, self).__init__(params, defaults)
        self.state['step'] = 0
        self.log_step = 176
        self.total_para = 0
        self.condition = condition
        for group in self.param_groups:
            for p in group['params']:
                self.total_para += p.numel()

    @torch.no_grad()
    def first_step(self, zero_grad=False):   
        self.state['step'] += 1
        step = self.state['step']
        
        if step % self.log_step == 0:
            self.weight_norm = self._weight_norm()
            
        self.first_grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (self.first_grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                param_state = self.state[p]
                
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                
                param_state['first_grad'] = p.grad.clone()
                param_state['e_w'] = e_w.clone()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        step = self.state['step']
        if step % self.log_step == 0:
            self.second_grad_norm = self._grad_norm()
            self.checkpoint1 = 0
            self.checkpoint2 = 0
            self.checkpoint3 = 0
            self.checkpoint4 = 0
            
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            step_size = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None: continue
                param_state = self.state[p]

                ratio = p.grad.div(param_state['first_grad'].add(1e-8))
                # mask = torch.logical_and( ratio > 0, ratio <= 1 ) if self.threshold != -1 else ratio.abs() > 1
                mask = torch.logical_and( ratio > 0, ratio <= 1 )

                d_p = p.grad.data

                if step % self.log_step == 0:
                    self.checkpoint1 += torch.sum( ratio > 1 )
                    self.checkpoint2 += torch.sum( torch.logical_and( ratio < 1, ratio > 0) )
                    self.checkpoint3 += torch.sum( torch.logical_and( ratio < 0, ratio.abs() > 1) )
                    self.checkpoint4 += torch.sum( torch.logical_and( ratio < 0, ratio.abs() < 1) )

                p.sub_(param_state['e_w'])  # get back to "w" from "w + e(w)"

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg'].mul_(momentum).add_(d_p)

                p.add_(param_state['exp_avg'].add( p.grad.mul( torch.logical_not( mask ) ).mul( self.condition ) ), alpha=-step_size)
        if step % self.log_step == 0:
            self.checkpoint1 = (self.checkpoint1 / self.total_para) * 100
            self.checkpoint2 = (self.checkpoint2 / self.total_para) * 100
            self.checkpoint3 = (self.checkpoint3 / self.total_para) * 100
            self.checkpoint4 = (self.checkpoint4 / self.total_para) * 100  
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

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