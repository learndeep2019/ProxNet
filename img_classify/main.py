import torch
import torch.optim as optim
import yaml
import os
from tqdm import trange
import logging
import argparse
import numpy as np

from dataset import get_loader
from model import ProxNet
from function import CCA

def main():
    parser = argparse.ArgumentParser(description='ProxNet Pytorch v.1.0')
    parser.add_argument('-c', '--config', default='PROX_125', choices=['PROX_20', 'PROX_50', 'PROX_100', 'PROX_125'])
    args = parser.parse_args()

    'load configuration and dataloader'
    global best_acc
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config = config[args.config]

    if not os.path.isdir('logs'):
        os.mkdir('logs')
    logging.basicConfig(filename='./logs/proxnet_{}.log'.format(args.config), filemode='a', level=logging.INFO)
    logging.info('=='*20)
    logging.info(config)

    train_loader, test_loader = get_loader(config)

    'build model'
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    model = ProxNet(config).to(device)

    if config['resume']:
        print('==> Resuming from checkpoint ... ')
        assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory !'
        checkpoint = torch.load('./checkpoint/{}_{}.ckpt'.format('proxnet', args.config))
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        current_epoch = checkpoint['epoch']
        print('The best accuracy so far :{:.4f}'.format(best_acc))
    else:
        best_acc = 0
        current_epoch = 0

    'initial criteria and optimizor'
    criteria = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay'])

    def train(epoch, config):
        print('Epoch: {}'.format(epoch))
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        corrh = 0
        corrf = 0
        acc = 0

        with trange(len(train_loader)) as t:
            for i, (img1, img2, label) in enumerate(train_loader):
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                prob1, prob2, h1, h2, f1, f2 = model(img1, img2, config['alpha'])
                corr_l = CCA.apply(h1, h2, config)
                corr_r = CCA.apply(f1, f2, config)
                loss = criteria(prob1, label) + criteria(prob2, label) - corr_l/float(config['proj_k'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                prob = prob1+prob2
                predict = torch.argmax(prob, 1)
                total += label.size(0)
                correct += predict.eq(label).sum().item()
                corrh += corr_l.item()
                corrf += corr_r.item()
                acc = correct/total*100

                metrics = {'loss': '{:.3f}'.format(train_loss/(i+1)),
                'acc':'{:.3f}({}/{})'.format(acc, correct, total),
                'corr':'{:.2f}/{:.2f}'.format(corrh/(i+1), corrf/(i+1))}
                t.set_postfix(metrics)
                t.update()
        lr_scheduler.step()
        logging.info('train: epoch {}, loss {:.3f}, acc {:.3f}({}/{}), corr {:.2f}/{:.2f}'.format(e, train_loss/(i+1), acc, correct, total, corrh/(i+1), corrf/(i+1)))

    def test(epoch, config):
        global best_acc
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        corrh = 0
        corrf = 0
        acc = 0

        with torch.no_grad():
            with trange(len(test_loader)) as t:
                for i, (img1, img2, label) in enumerate(test_loader):
                    img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                    prob1, prob2, h1, h2, f1, f2 = model(img1, img2, config['alpha'])
                    corr_l = CCA.apply(h1, h2, config)
                    corr_r = CCA.apply(f1, f2, config)
                    loss = criteria(prob1, label) + criteria(prob2, label) - corr_l/float(config['proj_k'])

                    test_loss += loss.item()
                    prob = prob1+prob2
                    predict = torch.argmax(prob, 1)
                    total += label.size(0)
                    correct += predict.eq(label).sum().item()
                    corrh += corr_l.item()
                    corrf += corr_r.item()
                    acc = correct/total*100

                    metrics = {'loss': '{:.3f}'.format(test_loss/(i+1)),
                    'acc':'{:.3f}({}/{})'.format(acc, correct, total),
                    'corr':'{:.2f}/{:.2f}'.format(corrh/(i+1), corrf/(i+1))}
                    t.set_postfix(metrics)
                    t.update()
        logging.info('test_: epoch {}, loss {:.3f}, acc {:.3f}({}/{}), corr {:.2f}/{:.2f}'.format(e, test_loss/(i+1), acc, correct, total, corrh/(i+1), corrf/(i+1)))
        acc = correct/total*100
        if acc > best_acc:
            print('saving checkpoint {}'.format(epoch))
            save_ckpt = {'model':model.state_dict(),
                        'acc': acc,
                        'epoch': epoch}
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(save_ckpt, './checkpoint/{}_{}.ckpt'.format('proxnet', args.config))
            best_acc = acc

    for e in range(current_epoch, current_epoch + config['epoches']):
        # config['alpha'] = （1+0.5*e)*0.1
        train(e, config)
        test(e, config)
        logging.info('-'*80)
    logging.info('the best acc {:.3f}'.format(best_acc))

if __name__ == "__main__":
    main()
