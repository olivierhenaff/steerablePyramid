import torch
from torch.autograd import Variable
import torch.nn as nn
import pytorch_fft.fft.autograd as fft

from steerableUtils import *  

class SteerablePyramid(nn.Module):

    def __init__(self, imgSize, K=4, N=4, hilb=False, includeHF=True ):
        super(SteerablePyramid, self).__init__()

        size = [ imgSize, imgSize//2 + 1 ]
        self.hl0 = Variable( HL0_matrix( size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda() )

        self.l = []
        self.b = []
        self.s = []

        self.K    = K 
        self.N    = N 
        self.hilb = hilb
        self.includeHF = includeHF 

        self.indF = [ freq_shift( size[0], True  ) ] 
        self.indB = [ freq_shift( size[0], False ) ] 

        self.fftF =   fft.Rfft2d()
        self.fftB = [ fft.Irfft2d() ]

        for n in range( self.N ):

            l = Variable( L_matrix_cropped( size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda() )
            b = Variable( B_matrix(      K, size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()              )
            s = Variable( S_matrix(      K, size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()              )

            self.l.append( l.div_(4) )
            self.b.append( b )
            self.s.append( s )

            size = [ l.size(-2), l.size(-1) ]

            self.indF.append( freq_shift( size[0], True  ) )
            self.indB.append( freq_shift( size[0], False ) )
            self.fftB.append( fft.Irfft2d() ) 


    def forward(self, x):

        x1, x2 = self.fftF(x)
        x = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1)), 1 ).unsqueeze( -3 )
        x = torch.index_select( x, -2, self.indF[0] )

        x   = self.hl0 * x 
        h0f = x.select( -3, 0 ).unsqueeze( -3 )
        l0f = x.select( -3, 1 ).unsqueeze( -3 )
        lf  = l0f 

        output = []

        for n in range( self.N ):

            bf = self.b[n] *               lf 
            lf = self.l[n] * central_crop( lf ) 
            if self.hilb:
                hbf = self.s[n] * torch.cat( (bf.narrow(1,1,1), -bf.narrow(1,0,1)), 1 )
                bf  = torch.cat( ( bf , hbf ), -3 )
            if self.includeHF and n == 0:
                bf  = torch.cat( ( h0f,  bf ), -3 )

            output.append( bf )

        output.append( lf  ) 

        for n in range( len( output ) ):
            output[n] = torch.index_select( output[n], -2, self.indB[n] )
            output[n] = self.fftB[n]( output[n].select( 1, 0 ), output[n].select( 1, 1 ) )

        if self.includeHF:
            output.insert( 0, output[0].narrow( -3, 0, 1                    ) )
            output[1]       = output[1].narrow( -3, 1, output[1].size(-3)-1 )

        for n in range( len( output ) ):
            if self.hilb:
                if ((not self.includeHF) or 0 < n) and n < len(output)-1:
                    nfeat = output[n].size(-3)//2
                    o1 = output[n].narrow( -3,     0, nfeat ).unsqueeze(1)
                    o2 = output[n].narrow( -3, nfeat, nfeat ).unsqueeze(1)
                    output[n] = torch.cat( (o1, o2), 1 ) 
                else:
                    output[n] = output[n].unsqueeze(1)

        return output


def testSteerable():

    imgSize = 128 
    network = SteerablePyramid( imgSize=imgSize, K=4, N=4 )

    x = torch.Tensor( 1, 1, imgSize, imgSize ).fill_(0)
    x.select( -1, imgSize//2 ).select( -1, imgSize//2 ).fill_( 1 )
    x = x.cuda()
    x = Variable( x, requires_grad=True ) 

    y = network( x ) 

    for i in range(len(y)):
        y[i].backward( y[i] ) 
    z = x.grad


    print( '\ninput size' )
    print( x.size() ) 

    print( '\noutput size' )
    for i in range(len(y)):
        img = y[i].data#.select(0,0)#.select(0,0)           
        # img = img.view( -1, img.size(-2), img.size(-1) )
        print( i, img.size() )
        # for j in range(img.size(0)):
        #     scale = 127.5 / max( img[j].max(), - img[j].min() )
        #     im = Image.fromarray( img[j].mul(scale).add(127.5).cpu().numpy().astype(numpy.uint8) )
        #     im.save( 'steerable_' + str(i) + '_' + str(j) + '.png' )

    print( '\nreconstruction size')
    print( z.size() ) 


    print( '\nreconstruction error', x.dist( z ).data.squeeze() ) 

testSteerable()