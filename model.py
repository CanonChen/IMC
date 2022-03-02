import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
import torchtext
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict


def l2norm(X, dim=1, esp=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + esp
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='vgg19', use_abs=False, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if data_name.endswith('_precomp'):
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, use_abs, no_imgnorm)
    else:
        img_enc = EncoderImageFull(
            embed_size, finetune, cnn_type, use_abs, no_imgnorm)

    return img_enc


class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, word2idx,
                 use_abs=False, rnn_type='LSTM', use_bidirectional_rnn=False, wordemb='bow'):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.wordemb = wordemb

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bidirectional_rnn = use_bidirectional_rnn
        print('=> using bidirectional rnn:{}'.format(use_bidirectional_rnn))
        # self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)
        self.rnn_type = rnn_type
        if rnn_type == 'GRU':
            print('=> using GRU type')
            self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                              batch_first=True, bidirectional=use_bidirectional_rnn)
        elif rnn_type == 'LSTM':
            print('=> using LSTM type')
            self.rnn = nn.LSTM(word_dim, embed_size, num_layers,
                               batch_first=True, bidirectional=use_bidirectional_rnn)
        self.projection = nn.Linear(self.embed_size, self.embed_size)

        self.dropout = nn.Dropout(0.4)

        self.init_weights(self.wordemb, word2idx, word_dim)

    def init_weights(self, wordemb, word2idx, word_dim):

        if wordemb.lower() == 'bow':
            # self.embed.weight.data.uniform_(-0.1, 0.1)
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wordemb.lower():
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wordemb.lower():
                wemb = torchtext.vocab.GloVe()
            else:
                raise Exception(
                    'Unknown word embedding type: {}'.format(wordemb))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace(
                        '-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        x = self.dropout(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bidirectional_rnn:
            cap_emb = (cap_emb[:, :, : int(cap_emb.size(2) / 2)] +
                       cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        if self.rnn_type == 'GRU':
            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
            out = torch.gather(cap_emb, 1, I).squeeze(1)
        elif self.rnn_type == 'LSTM':
            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
            out = torch.gather(cap_emb, 1, I).squeeze(1)
        out = l2norm(out, dim=1)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


def cosine_sim(im, s):
    """Cosine similarity between all the two pairs
    """
    return im.mm(s.t())


def l1_sim(im, s):
    """l1 similarity between two pairs
    """
    scro = torch.cdist(im, s, p=1)
    return scro


def l2_sim(im, s):
    """L2 similarity between two pairs
    """
    scro = torch.cdist(im, s, p=2)
    return scro


def msd_loss(im, s):
    """MSD similarity between two pairs
    """
    scro = torch.cdist(im, s, p=2)
    return scro.pow(2)


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.measure = measure
        self.sim = cosine_sim
        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class IntraLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, up=0.5, down=0.05, lamb=1.0):
        super(IntraLoss, self).__init__()
        self.margin = margin
        self.measure = measure
        if measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'msd':
            self.sim = msd_loss
        elif self.measure == 'l1':
            self.sim = l1_sim
        elif self.measure == 'l2':
            self.sim = l2_sim
        self.max_violation = max_violation
        self.up = up
        self.down = down
        self.lamb = lamb

    def forward(self, mx):
        # compute image-sentence score matrix
        scores = self.sim(mx, mx)

        if self.measure == 'cosine':
            diagonal = scores.diag()
            scores = scores.cuda()
            eye = torch.eye(scores.size(0)).float().cuda()
            scores_non_self = scores - eye
            # scores_non_self.gt_(self.up).lt_(1 - self.down)
            scores_non_self = scores_non_self * (
                scores_non_self.gt(self.up).float())
            scores_non_self = scores_non_self * (
                scores_non_self.lt(1 - self.down).float())
            scores_norm = scores_non_self.sum() / scores.size(0)
            # print(scores_norm.item())

        elif self.measure == 'msd' or self.measure == 'l1' or self.measure == 'l2':
            scores_non_self = torch.nn.functional.normalize(scores).cuda()

            idx_up = round(self.up * scores.size(0))
            idx_down = round(self.down * scores.size(0))
            _, s_index = scores_non_self.sort()
            s_mean = scores_non_self.mean()

            s_up = scores_non_self[0, s_index[0, idx_up]]
            s_down = scores_non_self[0, s_index[0, idx_down]]

            scores_non_self = scores_non_self * (
                scores_non_self.gt(s_down).float())
            scores_non_self = scores_non_self * (
                scores_non_self.lt(s_up).float())
            scores_norm = scores_non_self.sum() / scores.size(0)

        return self.lamb * scores_norm


class IMC(object):
    """
    canon/imc model
    """

    def __init__(self, opt, word2idx):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   word2idx=word2idx,
                                   use_abs=opt.use_abs, rnn_type=opt.rnn_type,
                                   use_bidirectional_rnn=opt.use_bidirectional_rnn, wordemb=opt.wordemb)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.cl_measure,
                                         max_violation=opt.max_violation)
        self.criterion_intra = IntraLoss(margin=opt.margin,
                                         measure=opt.il_measure,
                                         max_violation=opt.max_violation)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """

        loss = self.criterion(
            img_emb, cap_emb) + self.criterion_intra(cap_emb) + self.criterion_intra(img_emb)

        self.logger.update('Le', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
