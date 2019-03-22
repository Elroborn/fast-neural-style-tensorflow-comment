# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import utils
import os

slim = tf.contrib.slim

# gram特征
def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    # [1,height*width ,nums_filters]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    # grams特征就是两个矩阵乘
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)
    # grams shape is [num_images,filter,filter]
    return grams


def get_style_features(FLAGS):
    """
    For the "style_image", the preprocessing step is:
    1. Resize the shorter side to FLAGS.image_size
    2. Apply central crop
    """
    with tf.Graph().as_default():
        # 定义一个网络
        network_fn = nets_factory.get_network_fn(
            FLAGS.loss_model,
            num_classes=1,
            is_training=False)
        # 图片预处理方法
        image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
            FLAGS.loss_model,
            is_training=False)

        width,height = FLAGS.image_width,FLAGS.image_height
        # 读取图片数据
        img_bytes = tf.read_file(FLAGS.style_image)
        # 对图片做解码
        if FLAGS.style_image.lower().endswith('png'):
            image = tf.image.decode_png(img_bytes)
        else:
            image = tf.image.decode_jpeg(img_bytes)
        # 增加一个维度变为 [1,h,w,c]，vgg网络要求喂入数据为[batch_size,h,w,c]
        images = tf.expand_dims(image_preprocessing_fn(image, height,width), 0)
        # 前向传播
        _, endpoints_dict = network_fn(images, spatial_squeeze=False)
        features = []
        # 按照配置文件中的指定层特征进行提取
        for layer in FLAGS.style_layers:
            feature = endpoints_dict[layer]
            # 因为第0维度就是图片个数，肯定是1，所以压缩成一张图片
            feature = tf.squeeze(gram(feature), [0])  # remove the batch dimension

            features.append(feature)
        # 保存那个style图片 我觉得也可以忽略，没有什么实际作用
        with tf.Session() as sess:
            # Restore variables for loss network.
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)

            # Make sure the 'generated' directory is exists.
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            # Indicate cropped style image path
            save_file = 'generated/target_style_' + FLAGS.naming + '.jpg'
            # Write preprocessed style image to indicated path
            with open(save_file, 'wb') as f:
                target_image = image_unprocessing_fn(images[0, :])
                value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(value))
                tf.logging.info('Target style pattern is saved to: %s.' % save_file)

            # 只是自己测试，可以删除
            # save_feature_file = 'generated/target_style_feature' + FLAGS.naming + '.jpg'
            # with open(save_feature_file,'wb') as f:
            #     # 尝试把一个特征图写入
            #     feature_0 = tf.cast(features[2], tf.uint8)
            #     feature_0 = tf.expand_dims(feature_0, 2)
            #     feature_img = tf.image.encode_jpeg(feature_0)
            #     feature_img_v = sess.run(feature_img)
            #     f.write(feature_img_v)
            #     print(feature_img_v)
            #     tf.logging.info('feature style pattern is saved to: %s.' % save_file)

            # Return the features those layers are use for measuring style loss.
            return sess.run(features)

# 风格损失 endpoints_dict里面存的是y_c和y^的各层特征，style_features_t里面是风格特征
def style_loss(endpoints_dict, style_features_t, style_layers):
    style_loss = 0
    style_loss_summary = {}
    for style_gram, layer in zip(style_features_t, style_layers):
        # 因为之前我们送入网络的是[2*batch_szie,h,w,c],所以在这里我们需要拿到两组的分别的特征
        # 所以我们呢把特征分成[bathc_size,..] [batch_size,..]
        # 而我们计算风格损失时，不需要内容特征，所以用_
        generated_images, _ = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        # 一定要先对生成图片特征进行garm计算，然后算特征
        # 而style_features_t里面存储的本身就是经过garm计算的
        layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
        style_loss_summary[layer] = layer_style_loss
        style_loss += layer_style_loss
    return style_loss, style_loss_summary


def content_loss(endpoints_dict, content_layers):
    content_loss = 0
    for layer in content_layers:
        # tf.split把value按照0维度，分为2部分
        # 因为之前我们送入网络的是[2*batch_szie,h,w,c],所以在这里我们需要拿到两组的分别的特征
        # 所以我们呢把特征分成[bathc_size,..] [batch_size,..]
        generated_images, content_images = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        # 对应点相减，然后平方和，最后要除len，因为计算平均每张的损失
        content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  # remain the same as in the paper
    return content_loss

# 没仔细研究，主要是起到平滑的作用
def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss
