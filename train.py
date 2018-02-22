import torch
from options import opt
from torch.autograd import Variable
import models.model_factory
import dataset.dataset_factory
from torch.utils.data import DataLoader
from utils import CrossEntropyLoss2d, AverageMeter, evaluate, get_file_path, rle_encoding, prob_to_rles, analyze_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import torch.optim as optim
def main():
    net = models.model_factory.factory(opt.arch)
    net.train()
    net.cuda()
    train_set = dataset.dataset_factory.factory(opt.dataset)
    train_loader = DataLoader(train_set, batch_size = opt.batch_size, num_workers = opt.num_workers, shuffle = True)
    criterion = CrossEntropyLoss2d(size_average = True).cuda()
    optimizer = optim.SGD(net.parameters(), opt.lr, momentum = 0.9)
    curr_epoch = 0
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args['lr_patience'], min_lr=1e-10)
    for epoch in range(curr_epoch, 400):
        train(train_loader, net, criterion, optimizer, epoch)
        make_submit(opt.test_dir, net, opt.submit_file)

    

def train(train_loader, net, criterion, optimizer, epoch, ):
    net.train()
    net.cuda()
    curr_iter = (epoch-1)*len(train_loader)
    train_loss = AverageMeter()
    for i, data in enumerate(train_loader):
        inputs, masks = data
        N = inputs.size(0)
        inputs = Variable(inputs).cuda()
        masks = Variable(masks).cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, masks) 
        loss.backward()
        optimizer.step()
        train_loss.update(loss.data[0], 1)
        curr_iter += 1
        if (i + 1) % 10 == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg))
    save_dir = os.path.join('experiments', opt.experiments)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if epoch%10 == 0:
        torch.save(net.state_dict(), os.path.join(save_dir, str(epoch)+'.pth'))
def validate(val_loader, net, criterion, optimizer, epoch, ):
    net.eval()
    val_loss = AverageMeter()
    inputs_all, gts_all, predications_all = [], [], []
    for vi, data in enumerate(val_loader):
        inputs, gts = data
        N = inputs.size(0)
        inputs = Variable(inputs, volatile = True).cuda()
        gts = Variable(gts, volatile = True).cuda()
        outputs = net(inputs)
        predications = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
        val_loss.update(criterion(outputs, gts).data[0]/N, N)
        for i in inputs:
            pass
def make_submit(test_dir, net, save_results_dir):
    import pandas as pd
    import scipy.misc 
    import scipy
    import numpy as np
    import PIL
    from skimage.transform import resize
    net.eval()
    net.cpu()
    results = pd.DataFrame()
    test_ids = []
    rles = []
    results = pd.DataFrame()
    for test_id in os.listdir(test_dir):
        img_path = os.path.join(test_dir, test_id, 'images')
        img_file_path = get_file_path(img_path)
        #run_length = rle_encoding(prediction)
        im_df = analyze_image(net, img_file_path, test_id)
        results = results.append(im_df, ignore_index = True)
        #rle = list(prob_to_rles(prediction))
        #test_ids.extend([test_id]*len(rle))
        #rles.extend(rle)
    #results['ImageId'] = test_ids
    #results['EncodedPixels'] = pd.Series(rles).apply(lambda x : ' '.join(str(y) for y in x))
    results.to_csv(save_results_dir, index = False)
def test_submit():
    net = models.model_factory.factory(opt.arch)
    model_dir = os.path.join('experiments', opt.experiments, '960.pth')
    net.load_state_dict(torch.load(model_dir))
    make_submit(opt.test_dir, net, opt.submit_file)
if __name__ == '__main__':
    main()
    #test_submit()