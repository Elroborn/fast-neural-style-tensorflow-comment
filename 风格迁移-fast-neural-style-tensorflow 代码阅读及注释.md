# 风格迁移-fast-neural-style-tensorflow 代码阅读及注释

标签： `Tensorflow`

---  

这篇文章主要是记录在使用及阅读[fast-neural-style-tensorflow][1]代码时候的一些疑虑的解决，也可以看做是把fast-neural-style-tensorflow做一个精简化，因为虽然fast-neural-style-tensorflow给出了源代码，但是对于初学者来说并不是十分友好，所以这次相当于是做一个代码增加注释的过程。希望在这个过程中学习到一些常用的Tensorflow操作。    

## 关于风格迁移  

这个网上介绍很多，可以自行去搜索了解，比如可以参考[风格迁移][2]这篇文章，或者直接阅读[这篇论文][3]，里面主要就是两个网路：  

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190322155052526.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI4ODg4ODM3,size_16,color_FFFFFF,t_70)  

一个是生成图片的网络，就是图片中前面那个，主要用来生成图片，qi，后面的是一个vgg网络，主要是提取特征，其实就是用这些特征计算损失的，我们训练的时候只训练前面这个网络，后面的使用基于ImageNet训练好的模型，直接做特征提取。


## 代码结构  

虽然代码看起来很多，但是很大一部分是`slim`（Tensorflow.contrlib中的）中的代码，只是作者直接拷贝过来使用了。  

* conf是配置文件，相当于把以前的通过命令行参数改写到一个配置文件中了，要使用yaml进行解析，解析配置文件的代码在`utils.py`中。
* generated主要保存生成的图片
* img 存储风格图片
* models保存训练时的模型等
* nets 是slim的，你不用关心，主要是一些大型网络的结构定义
* preprocessing也是slim的，主要是图片的预处理
* pretrained用来存储训练好的vgg模型，因为在风格迁移中我们需要一个网络提取特征
* eval.py 在训练好模型后，使用模型生成迁移图片
* export.py 这个不要管，用不到
* losses.py 里面主要定义的是一些计算损失的方法
* model.py 里面定义了一个网络，就是前面的生成网络
* reader.py 主要用来读一个batch的数据，这个读数据如果看不懂可以参考我之前的博文[Tensorflow 文件读取][4]  
* train.py 主入口，主要是进行训练
* utils.py 里面提供了读取配置文件，以及加载模型到sess的操作

所以其实真正重要的只有`losses.py`和`train.py`，其他都是辅助，只要把这两个文件里面内容读懂即可。  

## 得到风格图片的特征  

也就是`y_s`的特征，因为这个特征我们一直要用，所以这里就直接计算出了。`style_features_t = losses.get_style_features(FLAGS)`这句话就是得到了特征，我们进入`get_style_features`函数，里面的`network_fn`只要是定义一个网络，那它是怎么定义的呢，主要是根据`FLAGS.loss_model`，这个就给出了模型的名字，这个名字刚好和`/net/nets_factory.py`里面的`networks_map`里面的对应上，这样就能构造出一个`vgg`网络结构了，这个都是slim做的，里面用了工厂模式，image_preprocessing_fn这一句同样的是定义了一个函数，这个函数能够对图片进行对应预处理，因为不同网络的预处理可能不同，所以这里根据FLAGS.loss_model来确定处理方式。下面就是读文件，然后对文件解码。` tf.expand_dims`是把图片增加一维，因为我们的网络一般喂入的数据是`[batch_szie,h,w,c]`格式，而image是[h,w,c]格式，所以扩大一个维度，变为[1,h,w,c]。` network_fn`这个就是对`images`做前向传播，最后返回`endpoints_dict`里面包含各层的数据，而`spatial_squeeze`这个是vgg模型的一个参数，你可以在`/nets/vgg`中vgg16函数中找到它。下面的for循环主要是取三个层的特征，至于是哪三层，可以在FLAGS.style_layers中找到。找到特征后，我们把对特征计算garm矩阵。而在garm矩阵中我们先把特征`reshape`为[1,h*w,num_filters]，因为garm矩阵就是计算两个矩阵相乘，所以要把后面的三维数据转为二维。而`tf.squeeze`主要是把数据进行去掉为1的维度，因为我们知道我们这里只有一张图片，所以直接把第0维度压缩了，最终feature的shape就是[num_filters,num_filters]。下面几行就是保存这个处理过的图片,`init_func = utils._get_init_fn(FLAGS)`就是把模型加载到sess中去，也是slim写好的。  

## 得到生成图片  

有了上面的基础，下面的看起来就简单一些了，在train中，同样先做一个vgg网络，主要用来提取生成图片以及内容图片的特征，同样有一个图片处理函数，而reader.images主要是读一个批次的图片，并且还按照预处理方式进行了处理，然后通过model.net生成一个图片，这个生成的图片相当于网络结构中的`y^`，而model.net就是生成网络，然后我们把生成图片进行一下预处理，然后把预处理过的内容图片以及生成图片喂入到vgg里面，注意这里有一个`tf.concat`相当于把两个堆起来了，因为axis =0，所以相当于是把原来两个[batch_size,h,w,c]变为了一个[2*btch_size,h,w,c],同样end_dict里面存储的是各层特征信息。  

## 计算损失  

在我们得到了风格的特征，以及内容图片和生成图片的特征后，我们
就可以进行计算损失了，先计算内容损失，也就是`losses.content_loss`，我们进入这个函数，其实计算比较简单，因为我们特征已经拿到了，都在endpoints_dict中存着，所以我们遍历需要的特征（在content_layers）里面，然后我们把生成图片的和内容图片的分开tf.split，因为之前我们前向传播的时候是按照tf.concat来的，所以这里使用tf.split分开，这样generated_images, content_images分别存储的就是生成图片和内容图片的一批次的特征，之后我们进行l2损失计算，并且求平均`/tf.to_float(size)`。  

而风格损失也是类似，我们进入style_loss，同样的先遍历需要的特征层，而对于风格特征，我们之前计算过了，这里存储在`style_features_t`里面，所以我们遍历即可。然后把生成图片的特征拿到，同样用split，并且在计算l2损失之前要面对生成图片的特征进行garm计算，因为风格损失就是以garm矩阵作为度量的。  

除此之外还有一个总变分损失，这个只是针对生成图片的，如果损失小那么图片会越来越平滑，而这个函数我没有仔细看，因为是一些数学计算，所以没有深究。 

然后我们把三个损失按照权重相加即可。  

下面的summary可以跳过，主要是tensorboard中使用，如果你学过tensorboard，那么肯定能看的懂。  

## 定义训练  

`variable_to_train`这个主要是找到能够训练的变量，因为在这个session中有两个网络，一个vgg，一个model.net也就是生成网络，而vgg里面的变量是不需要训练的，所以有`if not(variable.name.startswith(FLAGS.loss_model))`，这样的话`variable_to_train`里面就是生成网络里面需要训练的变量了。 下面的variables_to_restore也是类似，因为我们要保存模型的话，只需要保存生成网络中的即可，所以过滤掉vgg网络的。再往下面同样把模型加载到sess，后面就是开启训练了，`sess.run`。  

## 总结  

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190322165321680.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI4ODg4ODM3,size_16,color_FFFFFF,t_70)




  [1]: https://github.com/hzy46/fast-neural-style-tensorflow
  [2]: https://blog.csdn.net/qq_25737169/article/details/79192211
  [3]: https://arxiv.org/pdf/1603.08155.pdf
  [4]: https://blog.csdn.net/qq_28888837/article/details/88377690