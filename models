import paddle.fluid as fluid

class Unet_hermes():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def double_conv(self, image, out_ch):
        conv = fluid.layers.conv2d(image, num_filters=out_ch, filter_size=3, padding=1)
        conv = fluid.layers.batch_norm(conv, act="relu")
        conv = fluid.layers.conv2d(conv, num_filters=out_ch, filter_size=3, padding=1)
        conv = fluid.layers.batch_norm(conv, act="relu")
        return conv

    def inconv(self, image, out_ch):
        conv = self.double_conv(image, out_ch=out_ch)
        return conv

    def down(self, conv, out_ch):
        conv = fluid.layers.pool2d(conv, pool_size=2, pool_type="max", pool_stride=2)
        conv = self.double_conv(conv, out_ch)
        return conv

    def up(self, conv1, conv2, out_ch, bilinear=True):
        if bilinear:
            conv1 = fluid.layers.resize_bilinear(conv1, scale=2)
        else:
            c = conv1.shape[1] // 2
            conv1 = fluid.layers.conv2d_transpose(conv1, num_filters=c, filter_size=2, stride=2)

        conv = fluid.layers.concat([conv2, conv1], axis=1)
        conv = self.double_conv(conv, out_ch)

        return conv

    def net(self,image,image_s):
        conv1 = self.inconv(image,32)

        conv2 = self.down(conv1,64)
        conv3 = self.down(conv2,128)
        conv4 = self.down(conv3,256)
        conv5 = self.down(conv4,256)

        conv = self.up(conv5,conv4,128,bilinear=True)
        conv = self.up(conv,conv3,64,bilinear=True)
        conv = self.up(conv,conv2,32,bilinear=True)
        conv = self.up(conv,conv1,32,bilinear=True)

        conv1s = self.inconv(image_s, 16)

        conv2s = self.down(conv1s, 32)
        conv3s = self.down(conv2s, 64)
        conv4s = self.down(conv3s, 128)
        conv5s = self.down(conv4s, 128)

        convs = self.up(conv5s, conv4s, 64, bilinear=True)
        convs = self.up(convs, conv3s, 32, bilinear=True)
        convs = self.up(convs, conv2s, 16, bilinear=True)
        convs = self.up(convs, conv1s, 16, bilinear=True)

        convs = fluid.layers.pad(convs,[256,256,128,128])
        out = fluid.layers.concat([conv,convs],axis=1)
        out = self.double_conv(out,out_ch=32)
        conv = fluid.layers.conv2d(out,num_filters=self.num_classes,filter_size=1)
        return conv

class Unet():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def double_conv(self, image, out_ch):
        conv = fluid.layers.conv2d(image, num_filters=out_ch, filter_size=3, padding=1)
        conv = fluid.layers.batch_norm(conv, act="relu")
        conv = fluid.layers.conv2d(conv, num_filters=out_ch, filter_size=3, padding=1)
        conv = fluid.layers.batch_norm(conv, act="relu")
        return conv

    def inconv(self, image, out_ch):
        conv = self.double_conv(image, out_ch=out_ch)
        return conv

    def down(self, conv, out_ch):
        conv = fluid.layers.pool2d(conv, pool_size=2, pool_type="max", pool_stride=2)
        conv = self.double_conv(conv, out_ch)
        return conv

    def up(self, conv1, conv2, out_ch, bilinear=True):
        if bilinear:
            conv1 = fluid.layers.resize_bilinear(conv1, scale=2)
        else:
            c = conv1.shape[1] // 2
            conv1 = fluid.layers.conv2d_transpose(conv1, num_filters=c, filter_size=2, stride=2)

        conv = fluid.layers.concat([conv2, conv1], axis=1)
        conv = self.double_conv(conv, out_ch)

        return conv

    def net(self, image):
        conv1 = self.inconv(image, 64)

        conv2 = self.down(conv1, 128)
        conv3 = self.down(conv2, 256)
        conv4 = self.down(conv3, 512)
        conv5 = self.down(conv4, 512)

        conv = self.up(conv5, conv4, 256, bilinear=True)
        conv = self.up(conv, conv3, 128, bilinear=True)
        conv = self.up(conv, conv2, 64, bilinear=True)
        conv = self.up(conv, conv1, 64, bilinear=True)

        conv = fluid.layers.conv2d(conv, num_filters=self.num_classes, filter_size=1)
        return conv



