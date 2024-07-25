import torch
import os
from .utils import *
from .bypass_bn import *
import torch.nn.functional as F


def loop_one_epoch(
    dataloader,
    net,
    criterion,
    optimizer,
    device,
    logging_dict,
    epoch,
    loop_type='train',
    logging_name=None,
    best_acc=0
    ):
    loss = 0
    correct = 0
    total = 0
    
    if loop_type == 'train': 
        net.train()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            opt_name = type(optimizer).__name__
            if opt_name == 'SGD':
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.first_step(zero_grad=True)
            elif opt_name == 'SAMANATOMY' or opt_name == 'USAMANATOMY' or opt_name == 'GEOSAM':
                enable_running_stats(net)  # <- this is the important line
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()        
                optimizer.first_step(zero_grad=True)
                
                disable_running_stats(net)  # <- this is the important line
                criterion(net(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)
                
                criterion(net(inputs), targets).backward()
                optimizer.third_step(zero_grad=True)
            else:
                enable_running_stats(net)  # <- this is the important line
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()        
                optimizer.first_step(zero_grad=True)
                
                disable_running_stats(net)  # <- this is the important line
                criterion(net(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)
                
            try: 
                logging_dict[(f'{loop_type.title()}/hessian_norm', batch_idx)] = [optimizer.hessian_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/num_clamp', batch_idx)] = [optimizer.num_clamp, len(dataloader)]
            except: pass
                
            try: 
                logging_dict[(f'{loop_type.title()}/checkpoint1', batch_idx)] = [optimizer.checkpoint1, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/checkpoint2', batch_idx)] = [optimizer.checkpoint2, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/checkpoint3', batch_idx)] = [optimizer.checkpoint3, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/checkpoint4', batch_idx)] = [optimizer.checkpoint4, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/prev_checkpoint1', batch_idx)] = [optimizer.prev_checkpoint1, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/prev_checkpoint2', batch_idx)] = [optimizer.prev_checkpoint2, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/prev_checkpoint3', batch_idx)] = [optimizer.prev_checkpoint3, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/prev_checkpoint4', batch_idx)] = [optimizer.prev_checkpoint4, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/checkpoint12', batch_idx)] = [optimizer.checkpoint12, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/checkpoint24', batch_idx)] = [optimizer.checkpoint24, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/checkpoint48', batch_idx)] = [optimizer.checkpoint48, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/checkpoint816', batch_idx)] = [optimizer.checkpoint816, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/checkpoint1632', batch_idx)] = [optimizer.checkpoint1632, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/checkpoint3264', batch_idx)] = [optimizer.checkpoint3264, len(dataloader)]
            except: pass

            try: 
                logging_dict[(f'{loop_type.title()}/mean_grad_sq', batch_idx)] = [optimizer.mean_grad_sq, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/var_grad_sq', batch_idx)] = [optimizer.var_grad_sq, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/ckpt1_norm', batch_idx)] = [optimizer.ckpt1_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/ckpt2_norm', batch_idx)] = [optimizer.ckpt2_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/ckpt3_norm', batch_idx)] = [optimizer.ckpt3_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/ckpt4_norm', batch_idx)] = [optimizer.ckpt4_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/d_t_grad_norm', batch_idx)] = [optimizer.d_t_grad_norm, len(dataloader)]
            except: pass
                     
            try: 
                logging_dict[(f'{loop_type.title()}/first_grad_norm', batch_idx)] = [optimizer.first_grad_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/second_grad_norm', batch_idx)] = [optimizer.second_grad_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/third_grad_norm', batch_idx)] = [optimizer.third_grad_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/forget_grad_norm', batch_idx)] = [optimizer.forget_grad_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/learn_grad_norm', batch_idx)] = [optimizer.learn_grad_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/weight_norm', batch_idx)] = [optimizer.weight_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'summary/checkpoint_dict', batch_idx)] = [optimizer.checkpoint1_dict, len(dataloader)]
            except: pass
            
            with torch.no_grad():
                loss += first_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss/(batch_idx+1)
                acc = 100.*correct/total
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (loss_mean, acc, correct, total))
    else:
        if loop_type == 'test':
            print('==> Resuming from best checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            load_path = os.path.join('checkpoint', logging_name, 'ckpt_best.pth')
            checkpoint = torch.load(load_path)
            net.load_state_dict(checkpoint['net'])
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += first_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss/(batch_idx+1)
                acc = 100.*correct/total
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (loss_mean, acc, correct, total))
        if loop_type == 'val':
            if acc > best_acc:
                print('Saving best checkpoint ...')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'loss': loss,
                    'epoch': epoch
                }
                save_path = os.path.join('checkpoint', logging_name)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torch.save(state, os.path.join(save_path, 'ckpt_best.pth'))
                best_acc = acc
            logging_dict[f'{loop_type.title()}/best_acc'] = best_acc
            
    logging_dict[f'{loop_type.title()}/loss'] = loss_mean
    logging_dict[f'{loop_type.title()}/acc'] = acc

    if loop_type == 'val': 
        return best_acc