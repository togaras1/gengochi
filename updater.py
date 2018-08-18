import chainer
import chainer.functions as F
from chainer import cuda, Variable
from PIL import Image

class DCGANUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        params = kwargs.pop('params')
        super(DCGANUpdater, self).__init__(*args, **kwargs)
        self._dataset = params['dataset']
        self._iter = 0

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        self._iter += 1
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()

        x_real = Variable(self.converter(batch, self.device)) / 255.
        xp = chainer.backends.cuda.get_array_module(x_real.data)

        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        # 入力数と出力数があってない
        y_real = dis(x_real)

        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z)
        y_fake = dis(x_fake)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)

        #if(self._iter % 200 == 0):
        #    img = x_fake.data
        #    img = cuda.to_cpu(img)
        #    img = self._dataset.batch_postprocess_images(img, 5, 7)
        #    Image.fromarray(img).save("result/iter_"+str(self._iter)+".jpg")
