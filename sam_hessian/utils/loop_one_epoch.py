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
            if opt_name == 'SGD' or opt_name == 'SGDVAR':
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.first_step(zero_grad=True)
            # elif opt_name == 'SGDSAM':
            #     outputs = net(inputs)
            #     first_loss = criterion(outputs, targets)
            #     first_loss.backward()
            #     if (batch_idx + 1) % 5 == 0:
            #         optimizer.perturbed_step(zero_grad=True)
            #         disable_running_stats(net)  # <- this is the important line
            #         criterion(net(inputs), targets).backward()
            #         optimizer.unperturbed_step(zero_grad=True)
            #         enable_running_stats(net)  # <- this is the important line
            #     optimizer.first_step(zero_grad=True)
            elif opt_name == 'SAMATOMY':
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
                
            elif opt_name == 'SAMHESS':
                if batch_idx % 1 == 0:
                    h_outputs = net(inputs)
                    samp_dist = torch.distributions.Categorical(logits=h_outputs)
                    y_sample = samp_dist.sample()
                    h_loss = F.cross_entropy(h_outputs.view(-1, h_outputs.size(-1)), y_sample.view(-1), ignore_index=-1)
                    h_loss.backward()
                    optimizer.update_hessian()
                    optimizer.zero_grad(set_to_none=True)
                enable_running_stats(net)  # <- this is the important line
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward(create_graph=True)        
                optimizer.first_step(zero_grad=True)
                
                disable_running_stats(net)  # <- this is the important line
                criterion(net(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)
            # elif opt_name == 'SGDHESS':
            #     outputs = net(inputs)
            #     first_loss = criterion(outputs, targets)
            #     first_loss.backward(create_graph=True)        
            #     optimizer.first_step(zero_grad=True)
            #     # Zero the gradients explicitly
            #     for param in net.parameters():
            #         param.grad = None
            # elif opt_name == 'EKFAC':
            #     outputs = net(inputs)
            #     if optimizer.steps % optimizer.TCov == 0:
            #         # compute true fisher
            #         optimizer.acc_stats = True
            #         with torch.no_grad():
            #             sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
            #                                         1).squeeze().cuda()
            #         loss_sample = criterion(outputs, sampled_y)
            #         loss_sample.backward(retain_graph=True)
            #         optimizer.acc_stats = False
            #         optimizer.zero_grad()  # clear the gradient for computing true-fisher.
            #     first_loss = criterion(outputs, targets)
            #     first_loss.backward()
            #     optimizer.step()
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