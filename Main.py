import collections
import tensorflow as tf
import tensorflow.compat.v1
import time
from datetime import datetime
import math


slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    'A namee tuple describing a ResNet blick'
    """
    collections.nametuple:设计ResNet基本Block模块组，只包含数据结构，不包含具体方法，
    定义一个典型的Block，需要输入三个参数：
    scope:名称
    unit_fn:残差学习单元
    args:一个列表，每一个元素对应一个残差学习单元，维度
    [(256, 64, 1)] * 2 + [(256, 64, 2)]表示：残差学习单元的维度，代表了三个残差学习单元，以最后一个为例表示：深度为256，前两层深度为64，中间步长为2
    """


# 降采样
def subsample(inputs, factor, scope=None):
    """
    inputs:输入张量
    factor:采样因子

    """
    if factor == 1:  # 不采样
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


# 创建卷积层
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    # 判断步长是否为1，为1直接创建，padding为SAME
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope)
    # 不为1，显式的填充0，填充总数为kernel_size-1,
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        # 补零之后，使用Padding为VALID
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)


# 定义堆叠Blocks的函数
@slim.add_arg_scope
def stack_block_dense(net, blocks, outputs_collections=None):
    """
    net:输入
    blocks:Block的class列表
    """
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', 'net') as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    # unit_fn函数，残差单元的生成函数
                    net = block.unit_fn(net, depth=unit_depth, depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net


def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


# 残差学习单元
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        # 对输入进行激活和BN
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        # 如果输入和输出通道一致，则对inputs进行空间的降采样
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        # 如果不一致，则使用1x1的卷积改变其通道数
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')
        # 定义残差，3层
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

        output = shortcut + residual
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v2(inputs, blocks, num_classes=None, global_pool=True, include_root_block=True, reuse=tf.AUTO_REUSE,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_point'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_block_dense], outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                    # 第一层卷积
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                # 池化
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            # 将残差学习模块组生成好，生成堆叠网络。
            net = stack_block_dense(net, blocks)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')
            return net, end_points


# 50层网络
def resnet_v2_50(inputs, num_classes=None, global_pool=True, resuse=tf.AUTO_REUSE, scope='resnet_v2_50'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


# 101层网络
def resnet_v2_101(inputs, num_classes=None, global_pool=True, reuse=tf.AUTO_REUSE, scope='resnet_v2_101'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [1024, 256, 2]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


# 152层网络
def resnet_v2_152(inputs, num_classes=None, global_pool=True, reuse=tf.AUTO_REUSE, scope='resnet_v2_152'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


# 200层网咯
def resnet_v2_200(inputs, num_classes=None, global_pool=True, reuse=tf.AUTO_REUSE, scope='resnet_v2_200'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


# 评估每轮计算时间
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10  # 预热轮数
    total_duration = 0.0  # 总时间
    total_duration_squared = 0.0  # 平方和

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time

        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))

            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch' % (datetime.now(), info_string, num_batches, mn, sd))


print("hello world")
batch_size = 32
height, width = 224, 224
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(resnet_arg_scope(is_training=False)):
    # 评估152层网络
    net, end_points = resnet_v2_152(inputs, 1000)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess, net, "Forward")