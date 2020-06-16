import os
import argparse
import torch
from torch import optim
from torch import nn
from tqdm import trange
import yaml
import logging

from dataloader import get_loader
from ctcDecoder import phone_error
from ctcModel import CTCLstmProx
from cca import CCA

def main():
    parser = argparse.ArgumentParser(description='dccae for xrmb')
    parser.add_argument('-n', '--noisy', default=0, type=float)
    parser.add_argument('-c', '--ckpt', default='lstm_prox', type=str)
    args = parser.parse_args()

    with open('./config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    logging.basicConfig(filename='./logs/lstm_prox_{}.log'.format(args.noisy), filemode='a', level=logging.INFO)
    logging.info('=='*30)
    logging.info(config)
    '''
    init dataloader
    '''

    train_loader, test_loader = get_loader(config['batch_size'], config['blank'], args.noisy, seq_len=config['seq_len'])
    '''
    building model
    '''
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    model = CTCLstmProx(config).to(device)

    global best_per
    if config['resume']:
        print('==> Resuming from checkpoint ...')
        assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory!'
        checkpoint = torch.load('./checkpoint/{}_{}.ckpt'.format(args.ckpt, args.noisy))
        model.load_state_dict(checkpoint['model'])
        best_per = checkpoint['per']
        current_epoch = checkpoint['epoch']
    else:
        current_epoch = 0
        best_per = 100
    
    criteria = nn.CTCLoss(blank=config['blank'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay'])

    def train(epoch, config):
        print('==> Epoch: {}'.format(epoch))
        model.train()
        proj_k = config['proj_k']
        train_loss = 0
        corr_before = 0
        corr_after = 0

        with trange(len(train_loader)) as t:
            for i, (v1, v2, label, v_len, t_len) in enumerate(train_loader):
                v1, v2, label, v_len, t_len = v1.to(device), v2.to(device), label.to(device), v_len.to(device), t_len.to(device)
                before_prox_1, before_prox_2, after_prox_1, after_prox_2, log_prob = model(v1, v2, config['alpha'])
                corr = CCA.apply(before_prox_1.view(-1, proj_k), before_prox_2.view(-1, proj_k), config)
                loss = criteria(log_prob, label, v_len, t_len) - corr/float(proj_k)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                corr_before += corr.item()
                corr_prox = CCA.apply(after_prox_1.view(-1, proj_k), after_prox_2.view(-1, proj_k), config)
                corr_after += corr_prox.item()

                metrics = {'loss': '{:.3f}'.format(train_loss/(i+1)),
                'corr': '{:.3f}/{:.3f}'.format(corr_before/(i+1), corr_after/(i+1)),
                # 'per': '{:.3f}({}/{})'.format((err_phone/total_phone)*100, err_phone, total_phone)
                }

                t.set_postfix(metrics)
                t.update()
        lr_scheduler.step()
        logging.info('train: epoch {}, loss {:.3f}, corr {:.3f}/{:.3f}'.format(epoch, train_loss/(i+1), corr_before/(i+1), corr_after/(i+1)))
                
    def test(epoch, config): 
        global best_per
        model.eval()
        proj_k = config['proj_k']
        test_loss = 0
        corr_before = 0
        corr_after = 0
        total_phone = 0
        err_phone = 0
        with torch.no_grad():
            with trange(len(test_loader)) as t:
                for i, (v1, v2, label, v_len, t_len) in enumerate(test_loader):
                    v1, v2, label, v_len, t_len = v1.to(device), v2.to(device), label.to(device), v_len.to(device), t_len.to(device)
                    before_prox_1, before_prox_2, after_prox_1, after_prox_2, log_prob = model(v1, v2, config['alpha'])
                    corr = CCA.apply(before_prox_1.view(-1, proj_k), before_prox_2.view(-1, proj_k), config)
                    loss = criteria(log_prob, label, v_len, t_len) - corr/float(proj_k)

                    test_loss += loss.item()
                    corr_before += corr.item()
                    corr_prox = CCA.apply(after_prox_1.view(-1, proj_k), after_prox_2.view(-1, proj_k), config)
                    corr_after += corr_prox.item()
                    tp, ep = phone_error(log_prob.cpu(), label.cpu(), t_len.cpu(), blank=config['blank'])
                    total_phone += tp
                    err_phone += ep

                    metrics = {'loss': '{:.3f}'.format(test_loss/(i+1)),
                    'corr': '{:.3f}/{:.3f}'.format(corr_before/(i+1), corr_after/(i+1)),
                    'per': '{:.3f}({}/{})'.format((err_phone/total_phone)*100, err_phone, total_phone)
                    }
                    t.set_postfix(metrics)
                    t.update()
        per = (err_phone/total_phone)*100
        logging.info('test: epoch {}, loss {:.3f}, corr {:.3f}/{:.3f}, per {:.3f}({}/{})'.format(epoch, test_loss/(i+1), corr_before/(i+1), corr_after/(i+1), per, err_phone, total_phone))
        if per < best_per:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'per': per,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/{}_{}.ckpt'.format(args.ckpt, args.noisy))
            best_per = per

    for epoch in range(current_epoch, current_epoch+config['epochs']):
        # config['alpha'] = （1+epoch)*0.5
        train(epoch, config)
        test(epoch, config)
        logging.info('-'*80)
    logging.info('the best per {:.3f}'.format(best_per))

if __name__ == "__main__":
    main()
