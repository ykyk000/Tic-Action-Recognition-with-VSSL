import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
sys.path.append('./models')
from pytorch_i3d import InceptionI3d
from torchsummary import summary


class PrimaryCaps(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """

    def __init__(self,A, B, K, P, stride):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=True)

        self.pose.weight.data.normal_(0.0, 0.1)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                            kernel_size=K, stride=stride, bias=True)
        self.a.weight.data.normal_(0.0, 0.1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        p = self.pose(x)
        a = self.a(x)
        a = self.sigmoid(a)
        out = torch.cat([p, a], dim=1)
        out = out.permute(0, 2, 3, 1)
        return out

#ConvCaps(16, 8, (1, 1), P, stride=(1, 1), iters=3)
class ConvCaps(nn.Module):
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.

    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.

    Shape:
        input:  (*, h,  w, B*(P*P+1))
        output: (*, h', w', C*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """

    def __init__(self, B, C, K, P, stride, iters=3,
                 coor_add=False, w_shared=False):
        super(ConvCaps, self).__init__()
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        # constant
        self.eps = 1e-8
        # self._lambda = 1e-03
        self._lambda = 1e-6
        self.ln_2pi = torch.FloatTensor(1).fill_(math.log(2*math.pi))
        # self.ln_2pi = torch.cuda.HalfTensor(1).fill_(math.log(2*math.pi))

        # params
        # Note that \beta_u and \beta_a are per capsul/home/bruce/projects/capsulese type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.randn(C,self.psize))
        self.beta_a = nn.Parameter(torch.randn(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        # (1,32,24,4,4)
        self.weights = nn.Parameter(torch.randn(1, K[0]*K[1]*B, C, P, P))
        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def m_step(self, a_in, r, v, eps, b, B, C, psize):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

            Input:
                a_in:      (b, C, 1)
                r:         (b, B, C, 1)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                r_sum:     (b, C, 1)
            Output:
                a_out:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        cost_h = cost_h.sum(dim=2)


        cost_h_mean = torch.mean(cost_h,dim=1,keepdim=True)

        cost_h_stdv = torch.sqrt(torch.sum(cost_h - cost_h_mean,dim=1,keepdim=True)**2 / C + eps)
        # self._lambda = 1e-03
        # a_out = self.sigmoid(self._lambda * (self.beta_a - cost_h.sum(dim=2)))


        # cost_h_mean = cost_h_mean.sum(dim=2)
        # cost_h_stdv = cost_h_stdv.sum(dim=2)

        a_out = self.sigmoid(self._lambda*(self.beta_a - (cost_h_mean -cost_h)/(cost_h_stdv + eps)))

        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        """
            ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                    - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
            r = softmax(ln(a_j*p_j))
              = softmax(ln(a_j) + ln(p_j))

            Input:
                mu:        (b, 1, C, P*P)
                sigma:     (b, 1, C, P*P)
                a_out:     (b, C, 1)
                v:         (b, B, C, P*P)
            Local:
                ln_p_j_h:  (b, B, C, P*P)
                ln_ap:     (b, B, C, 1)
            Output:
                r:         (b, B, C, 1)
        """
        ln_p_j_h = -1. * (v - mu)**2 / (2 * sigma_sq) \
                    - torch.log(sigma_sq.sqrt()) \
                    - 0.5*self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(eps + a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
        return r

    def caps_em_routing(self, v, a_in, C, eps):
        """
            Input:
                v:         (b, B, C, P*P)
                a_in:      (b, C, 1)
            Output:
                mu:        (b, 1, C, P*P)
                a_out:     (b, C, 1)

            Note that some dimensions are merged
            for computation convenient, that is
            `b == batch_size*oh*ow`,
            `B == self.K*self.K*self.B`,
            `psize == self.P*self.P`
        """
        #(( 2000,32,24,16 ),(2000,32,1), 24, 1e-8)
        # b = 2000
        # B = 32
        # c = 24
        # psize = 16
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape
        r = torch.FloatTensor(b, B, C).fill_(1./C)
        # r = torch.cuda.HalfTensor(b, B, C).fill_(1./C)
        # print(r.dtype)
        # self.iters : 3
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            if iter_ < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)
        #  mu: (2000, 1, 24, 16)
        #  a_out: (2000, 24, 1)
        return mu, a_out

    def add_pathes(self, x, B, K, psize, stride):
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
        """
        b, h, w, c = x.shape
        assert h == w
        assert c == B*(psize+1)
        oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx) \
                for h_idx in range(0, h - K + 1, stride)] \
                for k_idx in range(0, K)]
        x = x[:, idxs, :, :]
        x = x[:, :, :, idxs, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x, oh, ow
    #x:[5, 20, 20, 544]
    #( x,32,(1,1),16,(1,1) )
    def add_pathes2(self,x, B, K=(3, 3), psize=4, stride=(1, 1)):
        b, h, w, c = x.shape
        assert c == B * (psize + 1)
        # oh:20
        oh = int((h - K[0] + 1) / stride[0])
        # ow:20
        ow = int((w - K[1] + 1) / stride[1])
        #
        idxs_h = [[(h_idx + k_idx) for h_idx in range(0, h - K[0] + 1, stride[0])] for k_idx in range(0, K[0])]
        idxs_w = [[(w_idx + k_idx) for w_idx in range(0, w - K[1] + 1, stride[1])] for k_idx in range(0, K[1])]
        x = x[:, idxs_h, :, :]
        x = x[:, :, :, idxs_w, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        return x, oh, ow

    def transform_view(self, x, w, C, P, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        #( (2000,32,16) ,(1,32,24,4,4) ,24 ,4 )
        # b =2000
        # B = 32
        # psize = 16
        b, B, psize = x.shape
        assert psize == P*P
        # x =(2000,32,1,4,4)
        x = x.view(b, B, 1, P, P)
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)
        # w =( 2000, 32, 24, 4 ,4)
        w = w.repeat(b, 1, 1, 1, 1)
        # x = (2000,32 ,24, 4, 4)
        x = x.repeat(1, 1, C, 1, 1)
        v = torch.matmul(x, w)
        # b = 2000
        # B = 32
        # c = 24
        # p*p = 16
        # v = ( 2000,32,24,16)
        v = v.view(b, B, C, P*P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = 1. * torch.arange(h) / h
        coor_h = torch.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)

        # coor_h = torch.cuda.HalfTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        # coor_w = torch.cuda.HalfTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h*w*B, C, psize)
        return v

    def forward(self, x):
        #x:[5, 20, 20, 544]
        b, h, w, c = x.shape
        if not self.w_shared:

            x, oh, ow = self.add_pathes2(x, self.B, self.K, self.psize, self.stride)


            p_in = x[:, :, :, :, :, :self.B*self.psize].contiguous()

            a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()

            p_in=p_in.view(b * oh * ow, self.K[0] * self.K[1] * self.B, self.psize)

            a_in = a_in.view(b * oh * ow, self.K[0] * self.K[1] * self.B, 1)
            v = self.transform_view(p_in, self.weights, self.C, self.P)
            p_out, a_out = self.caps_em_routing(v, a_in, self.C, self.eps)
            p_out = p_out.view(b, oh, ow, self.C*self.psize)
            a_out = a_out.view(b, oh, ow, self.C)
            out = torch.cat([p_out, a_out], dim=3)
        else:
            assert c == self.B*(self.psize+1)
            assert 1 == self.K[0] and 1 == self.K[1]
            assert 1 == self.stride[0] and 1 == self.stride[1]
            # assert 1 == self.K
            # assert 1 == self.stride
            p_in = x[:, :, :, :self.B*self.psize].contiguous()
            p_in = p_in.view(b, h*w*self.B, self.psize)
            a_in = x[:, :, :, self.B*self.psize:].contiguous()
            a_in = a_in.view(b, h*w*self.B, 1)

            # transform view
            v = self.transform_view(p_in, self.weights, self.C, self.P, self.w_shared)

            # coor_add
            if self.coor_add:
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            # em_routing
            _, out = self.caps_em_routing(v, a_in, self.C, self.eps)
        # out:(5,20,20,408)
        return out


class CapsNet(nn.Module):


    def __init__(self, pt_path='../weights/rgb_charades.pt', P=4, pretrained_load='i3d'):
        super(CapsNet, self).__init__()
        self.P = P


        self.conv1 = InceptionI3d(157, in_channels=3, final_endpoint='Mixed_4f')
        pretrained_weights = torch.load(pt_path)
        weights = self.conv1.state_dict()
        loaded_layers = 0

        for name in weights.keys():
            if name in pretrained_weights.keys():
                weights[name] = pretrained_weights[name]
                loaded_layers += 1

        self.conv1.load_state_dict(weights)
        print("Loaded I3D pretrained weights from ", pt_path, " for layers: ", loaded_layers)

        self.primary_caps = PrimaryCaps(832, 32, 9, P, stride=1)

        self.conv_caps = ConvCaps(32, 13, (1, 1), P, stride=(1, 1), iters=3)

        self.upsample1 = nn.ConvTranspose2d(208, 96, kernel_size=9, stride=1, padding=0)
        self.upsample1.weight.data.normal_(0.0, 0.02)
        self.upsample2 = nn.ConvTranspose3d(160,96, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding=1)
        self.upsample2.weight.data.normal_(0.0, 0.02)
        self.upsample3 = nn.ConvTranspose3d(160, 96, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding=1)
        self.upsample3.weight.data.normal_(0.0, 0.02)
        self.upsample4 = nn.ConvTranspose3d(160, 128, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding = (1,1,1))
        self.upsample4.weight.data.normal_(0.0, 0.02)     

        self.dropout3d = nn.Dropout3d(0.5)

        self.smooth = nn.ConvTranspose3d(128, 1, kernel_size=3,padding = 1)
        self.smooth.weight.data.normal_(0.0, 0.02)


        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.conv28 = nn.Conv2d(832, 64, kernel_size=(3, 3), padding=(1, 1))

        self.conv56 = nn.Conv3d(192, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv112 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))



    def load_pretrained_weights(self):
        saved_weights = torch.load('./savedweights/weights_referit')
        self.load_state_dict(saved_weights, strict=False)
        print('loaded referit pretrained weights for whole network')

    def load_previous_weights(self, weightfile):
        saved_weights = torch.load(weightfile)
        self.load_state_dict(saved_weights, strict=False)
        print('loaded weights from previous run: ', weightfile)
        

    def caps_reorder(self, imgcaps):

        h = imgcaps.size()[1]

        w = imgcaps.size()[2]

        img_data = imgcaps.size()[3]

        num_imgcaps = int(img_data / (self.P * self.P))

        pose_range = num_imgcaps * self.P * self.P
        img_poses = imgcaps[:, :, :, :pose_range]
        img_acts = imgcaps[:, :, :, pose_range:pose_range + num_imgcaps]

        combined_caps = torch.cat((img_poses, img_acts), dim=-1)
        return combined_caps
        
        
    def forward(self, img, classification, concat_labels, epoch, thresh_ep):
        '''
        INPUTS:
        img is of shape (B, 3, T, H, W) - B is batch size, T is number of frames (4 in our experiments), H and W are the height and width of frames (224x224 in our experiments)
        classification is of shape (B, ) - B is batch size - this contains the ground-truth class which will be used for masking at training time
        
        OUTPUTS:
        out is a list of segmentation masks (all copies of on another) of shape (B, T, H, W) - B is batch size, T is number of frames (4 in our experiments), H and W is the heights and widths (224x224 in our experiments)
        actor_prediction is the actor prediction (B, C) - B is batch size, C is the number of classes
        
        '''

        x, cross56, cross112 = self.conv1(img)

        x = self.dropout3d(x)
        x = x.view(-1, 832, 28, 28)

        cross28 = x.clone()
        # x: 4 x 20 x 20 x 544
        x = self.primary_caps(x)
   
        # x: 4 x 20 x 20 x 544
        x = self.caps_reorder(x)

        combined_caps = self.conv_caps(x)

        h = combined_caps.size()[1]

        w = combined_caps.size()[2]
   
        caps = int(combined_caps.size()[3] / ((self.P * self.P) + 1))
 
        ranges = int(caps * self.P * self.P)

        activations = combined_caps[:, :, :, ranges:ranges + caps]
    
        poses = combined_caps[:, :, :, :ranges]

        actor_prediction = activations

        feat_shape = activations

        feat_shape = torch.reshape(feat_shape, (feat_shape.shape[0], feat_shape.shape[1]*feat_shape.shape[2], feat_shape.shape[3]) )
        # actor_prediction : (4,20,13)
        actor_prediction = torch.mean(actor_prediction, 1)
        # actor_prediction :(4,13)
        actor_prediction = torch.mean(actor_prediction, 1)
        # poses:(4,20,20,13,16)
        poses = poses.view(-1,h,w,caps,self.P*self.P)

        if self.training:

            activations_labeled = torch.eye(caps)[classification.long()]

            activations_labeled = torch.squeeze(activations_labeled, 1)

            if epoch<thresh_ep:

                activations_unlabeled = torch.ones_like(activations_labeled)
            else:

                activations_unlabeled = torch.eye(caps)[torch.argmax(actor_prediction, dim=1)]
            activations = [activations_unlabeled[act] if concat_labels[act]==0 else activations_labeled[act] for act in range(len(concat_labels)) ]
            # activations: [4, 13]
            activations = torch.stack(activations)
            # activations: [4，13 , 1]
            activations = activations.view(-1, caps, 1)
            # activations: [4, 1 ,13 , 1]
            activations = torch.unsqueeze(activations, 1)
            # activations: [4, 1 , 1, 13 , 1]
            activations = torch.unsqueeze(activations, 1)

            activations = activations.repeat(1, h, w, 1, 1)
            activations = activations

            
        else:
            # activations: ( 1,24 )
            activations = torch.eye(caps)[torch.argmax(actor_prediction, dim=1)]
            #  activations: ( 1,24,1 )
            activations = activations.view(-1, caps, 1)
            # activations: ( 1,1,24,1 )
            activations = torch.unsqueeze(activations, 1)
            # activations: ( 1,1,1,24,1 )
            activations = torch.unsqueeze(activations, 1)
            # activations: ( 1,20,20,24,1 )
            activations = activations.repeat(1, h, w, 1, 1)
            activations = activations
        # poses:(2,20,20,24,16)   
        # activations: [2, 20 , 20, 24 , 1]
        poses = poses * activations
        # poses:(4,20,20,208) 
        poses = poses.view(-1,h,w,ranges)
        # poses : (4, 208, 20,20)
        poses = poses.permute(0, 3, 1, 2)
        # x:(4, 208,20,20)
        x = poses
        
        # x:[4, 64, 28, 28]
        x = self.relu(self.upsample1(x))
        # x:[4, 64, 1,28, 28]
        x = x.view(-1, 96, 1, 28, 28)

        cross28 = cross28.view(-1, 832, 28, 28)
        
        cross28 = self.relu(self.conv28(cross28))
        cross28 = cross28.view(-1,64,1,28,28)

        x = torch.cat((x, cross28), dim=1)

        x = self.relu(self.upsample2(x))
        cross56 = self.relu(self.conv56(cross56))
        x = torch.cat((x, cross56), dim=1)
        x = self.relu(self.upsample3(x))
        cross112 = self.relu(self.conv112(cross112))
        x = torch.cat((x, cross112), dim=1)

        x = self.upsample4(x)

        x = self.dropout3d(x)

        x = self.smooth(x)
        out_1 = x.view(-1,1,8,224,224)
        return out_1, actor_prediction, feat_shape

if __name__ == "__main__":
    model = CapsNet()
    concat_data = torch.randn((4 ,3 ,8 , 224 ,224))
    concat_action = torch.randn((4,1))
    concat_labels = torch.randn((4,1))
    output, predicted_action, feat = model(concat_data, concat_action, concat_labels, 1, 11)
    print(predicted_action.shape)

