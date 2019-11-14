import paddle.fluid as fluid

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



