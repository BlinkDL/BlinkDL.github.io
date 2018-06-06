# State of the Art

希望方便大家了解和复现 AI / ML / DL / RL / CV / NLP 中的 SotA 结果，方法和技巧。

如您希望参与，欢迎在此处 star & pull request：<a href="https://github.com/BlinkDL/BlinkDL.github.io" target="_blank">https://github.com/BlinkDL/BlinkDL.github.io</a>

更多参考：<a href="https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems" target="_blank">https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems</a>

TODO：

* 翻译和概括各篇论文的核心思想，加入图片说明。这里的许多论文在网上已有介绍文章，可加入链接。
* 逐步提供这里的所有 SotA 模型。最好能直接在大家的浏览器中运行。从前写过一个简单的浏览器运行库 <a href="https://github.com/BlinkDL/BlinkDL" target="_blank">https://github.com/BlinkDL/BlinkDL</a> 可逐步改进。
* 补充 NLP / 语音 等等的 SotA 结果，以及深度学习理论/训练方法的值得关注的进展。
* 提供 速度-性能 对比图。回顾经典模型。

## CV - 二维图像任务

目前在图像模型中已流行加入 attention，效果显著，大家可关注。代价是对训练的算力要求越来越高。

### Image Data Augmentation

AutoAugment: Learning Augmentation Policies from Data <a href="http://arxiv.org/abs/1805.09501" target="_blank">http://arxiv.org/abs/1805.09501</a>
![1805.09501v1](http://www.arxiv-sanity.com/static/thumbs/1805.09501v1.pdf.jpg)
In this paper, we take a closer look at data augmentation for images, and
describe a simple procedure called AutoAugment to search for improved data
augmentation policies. Our key insight is to create a search space of data
augmentation policies, evaluating the quality of a particular policy directly
on the dataset of interest. In our implementation, we have designed a search
space where a policy consists of many sub-policies, one of which is randomly
chosen for each image in each mini-batch. A sub-policy consists of two
operations, each operation being an image processing function such as
translation, rotation, or shearing, and the probabilities and magnitudes with
which the functions are applied. We use a search algorithm to find the best
policy such that the neural network yields the highest validation accuracy on a
target dataset. Our method achieves state-of-the-art accuracy on CIFAR-10,
CIFAR-100, SVHN, and ImageNet (without additional data). On ImageNet, we attain
a Top-1 accuracy of 83.54%. On CIFAR-10, we achieve an error rate of 1.48%,
which is 0.65% better than the previous state-of-the-art. On reduced data
settings, AutoAugment performs comparably to semi-supervised methods without
using any unlabeled examples. Finally, policies learned from one dataset can be
transferred to work well on other similar datasets. For example, the policy
learned on ImageNet allows us to achieve state-of-the-art accuracy on the fine
grained visual classification dataset Stanford Cars, without fine-tuning
weights pre-trained on additional data.

### Image Super-resolution

Deep Back-Projection Networks For Super-Resolution <a href="http://arxiv.org/abs/1803.02735" target="_blank">http://arxiv.org/abs/1803.02735</a>
![1803.02735v1](http://www.arxiv-sanity.com/static/thumbs/1803.02735v1.pdf.jpg)
The feed-forward architectures of recently proposed deep super-resolution
networks learn representations of low-resolution inputs, and the non-linear
mapping from those to high-resolution output. However, this approach does not
fully address the mutual dependencies of low- and high-resolution images. We
propose Deep Back-Projection Networks (DBPN), that exploit iterative up- and
down-sampling layers, providing an error feedback mechanism for projection
errors at each stage. We construct mutually-connected up- and down-sampling
stages each of which represents different types of image degradation and
high-resolution components. We show that extending this idea to allow
concatenation of features across up- and down-sampling stages (Dense DBPN)
allows us to reconstruct further improve super-resolution, yielding superior
results and in particular establishing new state of the art results for large
scaling factors such as 8x across multiple data sets.

Enhanced Deep Residual Networks for Single Image Super-Resolution <a href="http://arxiv.org/abs/1707.02921" target="_blank">http://arxiv.org/abs/1707.02921</a>
![1707.02921v1](http://www.arxiv-sanity.com/static/thumbs/1707.02921v1.pdf.jpg)
Recent research on super-resolution has progressed with the development of
deep convolutional neural networks (DCNN). In particular, residual learning
techniques exhibit improved performance. In this paper, we develop an enhanced
deep super-resolution network (EDSR) with performance exceeding those of
current state-of-the-art SR methods. The significant performance improvement of
our model is due to optimization by removing unnecessary modules in
conventional residual networks. The performance is further improved by
expanding the model size while we stabilize the training procedure. We also
propose a new multi-scale deep super-resolution system (MDSR) and training
method, which can reconstruct high-resolution images of different upscaling
factors in a single model. The proposed methods show superior performance over
the state-of-the-art methods on benchmark datasets and prove its excellence by
winning the NTIRE2017 Super-Resolution Challenge.

### Image Denoise / Demosaic

Learning to See in the Dark <a href="http://arxiv.org/abs/1805.01934" target="_blank">http://arxiv.org/abs/1805.01934</a>
![1805.01934v1](http://www.arxiv-sanity.com/static/thumbs/1805.01934v1.pdf.jpg)
Imaging in low light is challenging due to low photon count and low SNR.
Short-exposure images suffer from noise, while long exposure can induce blur
and is often impractical. A variety of denoising, deblurring, and enhancement
techniques have been proposed, but their effectiveness is limited in extreme
conditions, such as video-rate imaging at night. To support the development of
learning-based pipelines for low-light image processing, we introduce a dataset
of raw short-exposure low-light images, with corresponding long-exposure
reference images. Using the presented dataset, we develop a pipeline for
processing low-light images, based on end-to-end training of a
fully-convolutional network. The network operates directly on raw sensor data
and replaces much of the traditional image processing pipeline, which tends to
perform poorly on such data. We report promising results on the new dataset,
analyze factors that affect performance, and highlight opportunities for future
work. The results are shown in the supplementary video at
https://youtu.be/qWKUFK7MWvg

### Image Deblurring

Learning a Discriminative Prior for Blind Image Deblurring <a href="http://arxiv.org/abs/1803.03363" target="_blank">http://arxiv.org/abs/1803.03363</a>
![1803.03363v2](http://www.arxiv-sanity.com/static/thumbs/1803.03363v2.pdf.jpg)
We present an effective blind image deblurring method based on a data-driven
discriminative prior.Our work is motivated by the fact that a good image prior
should favor clear images over blurred images.In this work, we formulate the
image prior as a binary classifier which can be achieved by a deep
convolutional neural network (CNN).The learned prior is able to distinguish
whether an input image is clear or not.Embedded into the maximum a posterior
(MAP) framework, it helps blind deblurring in various scenarios, including
natural, face, text, and low-illumination images.However, it is difficult to
optimize the deblurring method with the learned image prior as it involves a
non-linear CNN.Therefore, we develop an efficient numerical approach based on
the half-quadratic splitting method and gradient decent algorithm to solve the
proposed model.Furthermore, the proposed model can be easily extended to
non-uniform deblurring.Both qualitative and quantitative experimental results
show that our method performs favorably against state-of-the-art algorithms as
well as domain-specific image deblurring approaches.

### Image Inpaint

Image Inpainting for Irregular Holes Using Partial Convolutions <a href="http://arxiv.org/abs/1804.07723" target="_blank">http://arxiv.org/abs/1804.07723</a>
![1804.07723v1](http://www.arxiv-sanity.com/static/thumbs/1804.07723v1.pdf.jpg)
Existing deep learning based image inpainting methods use a standard
convolutional network over the corrupted image, using convolutional filter
responses conditioned on both valid pixels as well as the substitute values in
the masked holes (typically the mean value). This often leads to artifacts such
as color discrepancy and blurriness. Post-processing is usually used to reduce
such artifacts, but are expensive and may fail. We propose the use of partial
convolutions, where the convolution is masked and renormalized to be
conditioned on only valid pixels. We further include a mechanism to
automatically generate an updated mask for the next layer as part of the
forward pass. Our model outperforms other methods for irregular masks. We show
qualitative and quantitative comparisons with other methods to validate our
approach.

### Image Generation

Self-Attention Generative Adversarial Networks <a href="http://arxiv.org/abs/1805.08318" target="_blank">http://arxiv.org/abs/1805.08318</a>
![1805.08318v1](http://www.arxiv-sanity.com/static/thumbs/1805.08318v1.pdf.jpg)
In this paper, we propose the Self-Attention Generative Adversarial Network
(SAGAN) which allows attention-driven, long-range dependency modeling for image
generation tasks. Traditional convolutional GANs generate high-resolution
details as a function of only spatially local points in lower-resolution
feature maps. In SAGAN, details can be generated using cues from all feature
locations. Moreover, the discriminator can check that highly detailed features
in distant portions of the image are consistent with each other. Furthermore,
recent work has shown that generator conditioning affects GAN performance.
Leveraging this insight, we apply spectral normalization to the GAN generator
and find that this improves training dynamics. The proposed SAGAN achieves the
state-of-the-art results, boosting the best published Inception score from 36.8
to 52.52 and reducing Frechet Inception distance from 27.62 to 18.65 on the
challenging ImageNet dataset. Visualization of the attention layers shows that
the generator leverages neighborhoods that correspond to object shapes rather
than local regions of fixed shape.

Progressive Growing of GANs for Improved Quality, Stability, and
  Variation <a href="http://arxiv.org/abs/1710.10196" target="_blank">http://arxiv.org/abs/1710.10196</a> （高分辨率）
![1710.10196v3](http://www.arxiv-sanity.com/static/thumbs/1710.10196v3.pdf.jpg)
We describe a new training methodology for generative adversarial networks.
The key idea is to grow both the generator and discriminator progressively:
starting from a low resolution, we add new layers that model increasingly fine
details as training progresses. This both speeds the training up and greatly
stabilizes it, allowing us to produce images of unprecedented quality, e.g.,
CelebA images at 1024^2. We also propose a simple way to increase the variation
in generated images, and achieve a record inception score of 8.80 in
unsupervised CIFAR10. Additionally, we describe several implementation details
that are important for discouraging unhealthy competition between the generator
and discriminator. Finally, we suggest a new metric for evaluating GAN results,
both in terms of image quality and variation. As an additional contribution, we
construct a higher-quality version of the CelebA dataset.

Disentangled Person Image Generation <a href="http://arxiv.org/abs/1712.02621" target="_blank">http://arxiv.org/abs/1712.02621</a> （特定领域）
![1712.02621v3](http://www.arxiv-sanity.com/static/thumbs/1712.02621v3.pdf.jpg)
Generating novel, yet realistic, images of persons is a challenging task due
to the complex interplay between the different image factors, such as the
foreground, background and pose information. In this work, we aim at generating
such images based on a novel, two-stage reconstruction pipeline that learns a
disentangled representation of the aforementioned image factors and generates
novel person images at the same time. First, a multi-branched reconstruction
network is proposed to disentangle and encode the three factors into embedding
features, which are then combined to re-compose the input image itself. Second,
three corresponding mapping functions are learned in an adversarial manner in
order to map Gaussian noise to the learned embedding feature space, for each
factor respectively. Using the proposed framework, we can manipulate the
foreground, background and pose of the input image, and also sample new
embedding features to generate such targeted manipulations, that provide more
control over the generation process. Experiments on Market-1501 and Deepfashion
datasets show that our model does not only generate realistic person images
with new foregrounds, backgrounds and poses, but also manipulates the generated
factors and interpolates the in-between states. Another set of experiments on
Market-1501 shows that our model can also be beneficial for the person
re-identification task.

### Image-to-image Transfer

Multimodal Unsupervised Image-to-Image Translation <a href="http://arxiv.org/abs/1804.04732" target="_blank">http://arxiv.org/abs/1804.04732</a>
![1804.04732v1](http://www.arxiv-sanity.com/static/thumbs/1804.04732v1.pdf.jpg)
Unsupervised image-to-image translation is an important and challenging
problem in computer vision. Given an image in the source domain, the goal is to
learn the conditional distribution of corresponding images in the target
domain, without seeing any pairs of corresponding images. While this
conditional distribution is inherently multimodal, existing approaches make an
overly simplified assumption, modeling it as a deterministic one-to-one
mapping. As a result, they fail to generate diverse outputs from a given source
domain image. To address this limitation, we propose a Multimodal Unsupervised
Image-to-image Translation (MUNIT) framework. We assume that the image
representation can be decomposed into a content code that is domain-invariant,
and a style code that captures domain-specific properties. To translate an
image to another domain, we recombine its content code with a random style code
sampled from the style space of the target domain. We analyze the proposed
framework and establish several theoretical results. Extensive experiments with
comparisons to the state-of-the-art approaches further demonstrates the
advantage of the proposed framework. Moreover, our framework allows users to
control the style of translation outputs by providing an example style image.
Code and pretrained models are available at https://github.com/nvlabs/MUNIT.

### Style Transfer

Neural Style Transfer: A Review <a href="http://arxiv.org/abs/1705.04058" target="_blank">http://arxiv.org/abs/1705.04058</a> （Review）
![1705.04058v5](http://www.arxiv-sanity.com/static/thumbs/1705.04058v5.pdf.jpg)
The seminal work of Gatys et al. demonstrated the power of Convolutional
Neural Networks (CNN) in creating artistic imagery by separating and
recombining image content and style. This process of using CNN to render a
content image in different styles is referred to as Neural Style Transfer
(NST). Since then, NST has become a trending topic both in academic literature
and industrial applications. It is receiving increasing attention and a variety
of approaches are proposed to either improve or extend the original NST
algorithm. This review aims to provide an overview of the current progress
towards NST, as well as discussing its various applications and open problems
for future research.
 
### Image Classification

Learning Transferable Architectures for Scalable Image Recognition <a href="http://arxiv.org/abs/1707.07012" target="_blank">http://arxiv.org/abs/1707.07012</a>
![1707.07012v4](http://www.arxiv-sanity.com/static/thumbs/1707.07012v4.pdf.jpg)
Developing neural network image classification models often requires
significant architecture engineering. In this paper, we study a method to learn
the model architectures directly on the dataset of interest. As this approach
is expensive when the dataset is large, we propose to search for an
architectural building block on a small dataset and then transfer the block to
a larger dataset. The key contribution of this work is the design of a new
search space (the "NASNet search space") which enables transferability. In our
experiments, we search for the best convolutional layer (or "cell") on the
CIFAR-10 dataset and then apply this cell to the ImageNet dataset by stacking
together more copies of this cell, each with their own parameters to design a
convolutional architecture, named "NASNet architecture". We also introduce a
new regularization technique called ScheduledDropPath that significantly
improves generalization in the NASNet models. On CIFAR-10 itself, NASNet
achieves 2.4% error rate, which is state-of-the-art. On ImageNet, NASNet
achieves, among the published works, state-of-the-art accuracy of 82.7% top-1
and 96.2% top-5 on ImageNet. Our model is 1.2% better in top-1 accuracy than
the best human-invented architectures while having 9 billion fewer FLOPS - a
reduction of 28% in computational demand from the previous state-of-the-art
model. When evaluated at different levels of computational cost, accuracies of
NASNets exceed those of the state-of-the-art human-designed models. For
instance, a small version of NASNet also achieves 74% top-1 accuracy, which is
3.1% better than equivalently-sized, state-of-the-art models for mobile
platforms. Finally, the learned features by NASNet used with the Faster-RCNN
framework surpass state-of-the-art by 4.0% achieving 43.1% mAP on the COCO
dataset.

Squeeze-and-Excitation Networks <a href="http://arxiv.org/abs/1709.01507" target="_blank">http://arxiv.org/abs/1709.01507</a>
![1709.01507v2](http://www.arxiv-sanity.com/static/thumbs/1709.01507v2.pdf.jpg)
Convolutional neural networks are built upon the convolution operation, which
extracts informative features by fusing spatial and channel-wise information
together within local receptive fields. In order to boost the representational
power of a network, several recent approaches have shown the benefit of
enhancing spatial encoding. In this work, we focus on the channel relationship
and propose a novel architectural unit, which we term the
"Squeeze-and-Excitation" (SE) block, that adaptively recalibrates channel-wise
feature responses by explicitly modelling interdependencies between channels.
We demonstrate that by stacking these blocks together, we can construct SENet
architectures that generalise extremely well across challenging datasets.
Crucially, we find that SE blocks produce significant performance improvements
for existing state-of-the-art deep architectures at minimal additional
computational cost. SENets formed the foundation of our ILSVRC 2017
classification submission which won first place and significantly reduced the
top-5 error to 2.251%, achieving a ~25% relative improvement over the winning
entry of 2016. Code and models are available at
https://github.com/hujie-frank/SENet.

### Object Detection

Focal Loss for Dense Object Detection <a href="http://arxiv.org/abs/1708.02002" target="_blank">http://arxiv.org/abs/1708.02002</a> （更准）
![1708.02002v2](http://www.arxiv-sanity.com/static/thumbs/1708.02002v2.pdf.jpg)
The highest accuracy object detectors to date are based on a two-stage
approach popularized by R-CNN, where a classifier is applied to a sparse set of
candidate object locations. In contrast, one-stage detectors that are applied
over a regular, dense sampling of possible object locations have the potential
to be faster and simpler, but have trailed the accuracy of two-stage detectors
thus far. In this paper, we investigate why this is the case. We discover that
the extreme foreground-background class imbalance encountered during training
of dense detectors is the central cause. We propose to address this class
imbalance by reshaping the standard cross entropy loss such that it
down-weights the loss assigned to well-classified examples. Our novel Focal
Loss focuses training on a sparse set of hard examples and prevents the vast
number of easy negatives from overwhelming the detector during training. To
evaluate the effectiveness of our loss, we design and train a simple dense
detector we call RetinaNet. Our results show that when trained with the focal
loss, RetinaNet is able to match the speed of previous one-stage detectors
while surpassing the accuracy of all existing state-of-the-art two-stage
detectors. Code is at: https://github.com/facebookresearch/Detectron.

YOLOv3: An Incremental Improvement <a href="http://arxiv.org/abs/1804.02767" target="_blank">http://arxiv.org/abs/1804.02767</a> （更快）
![1804.02767v1](http://www.arxiv-sanity.com/static/thumbs/1804.02767v1.pdf.jpg)
We present some updates to YOLO! We made a bunch of little design changes to
make it better. We also trained this new network that's pretty swell. It's a
little bigger than last time but more accurate. It's still fast though, don't
worry. At 320x320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but
three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3
is quite good. It achieves 57.9 mAP@50 in 51 ms on a Titan X, compared to 57.5
mAP@50 in 198 ms by RetinaNet, similar performance but 3.8x faster. As always,
all the code is online at https://pjreddie.com/yolo/

### Image Segmentation

Mask R-CNN <a href="http://arxiv.org/abs/1703.06870" target="_blank">http://arxiv.org/abs/1703.06870</a> （可分割出对象）
![1703.06870v3](http://www.arxiv-sanity.com/static/thumbs/1703.06870v3.pdf.jpg)
We present a conceptually simple, flexible, and general framework for object
instance segmentation. Our approach efficiently detects objects in an image
while simultaneously generating a high-quality segmentation mask for each
instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a
branch for predicting an object mask in parallel with the existing branch for
bounding box recognition. Mask R-CNN is simple to train and adds only a small
overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to
generalize to other tasks, e.g., allowing us to estimate human poses in the
same framework. We show top results in all three tracks of the COCO suite of
challenges, including instance segmentation, bounding-box object detection, and
person keypoint detection. Without bells and whistles, Mask R-CNN outperforms
all existing, single-model entries on every task, including the COCO 2016
challenge winners. We hope our simple and effective approach will serve as a
solid baseline and help ease future research in instance-level recognition.
Code has been made available at: https://github.com/facebookresearch/Detectron

Encoder-Decoder with Atrous Separable Convolution for Semantic Image
  Segmentation <a href="http://arxiv.org/abs/1802.02611" target="_blank">http://arxiv.org/abs/1802.02611</a> （只分割出类别）
![1802.02611v2](http://www.arxiv-sanity.com/static/thumbs/1802.02611v2.pdf.jpg)
Spatial pyramid pooling module or encode-decoder structure are used in deep
neural networks for semantic segmentation task. The former networks are able to
encode multi-scale contextual information by probing the incoming features with
filters or pooling operations at multiple rates and multiple effective
fields-of-view, while the latter networks can capture sharper object boundaries
by gradually recovering the spatial information. In this work, we propose to
combine the advantages from both methods. Specifically, our proposed model,
DeepLabv3+, extends DeepLabv3 by adding a simple yet effective decoder module
to refine the segmentation results especially along object boundaries. We
further explore the Xception model and apply the depthwise separable
convolution to both Atrous Spatial Pyramid Pooling and decoder modules,
resulting in a faster and stronger encoder-decoder network. We demonstrate the
effectiveness of the proposed model on the PASCAL VOC 2012 semantic image
segmentation dataset and achieve a performance of 89% on the test set without
any post-processing. Our paper is accompanied with a publicly available
reference implementation of the proposed models in Tensorflow at
https://github.com/tensorflow/models/tree/master/research/deeplab.

### Caption-to-Image

Photographic Text-to-Image Synthesis with a Hierarchically-nested
  Adversarial Network <a href="http://arxiv.org/abs/1802.09178" target="_blank">http://arxiv.org/abs/1802.09178</a>
![1802.09178v2](http://www.arxiv-sanity.com/static/thumbs/1802.09178v2.pdf.jpg)
This paper presents a novel method to deal with the challenging task of
generating photographic images conditioned on semantic image descriptions. Our
method introduces accompanying hierarchical-nested adversarial objectives
inside the network hierarchies, which regularize mid-level representations and
assist generator training to capture the complex image statistics. We present
an extensile single-stream generator architecture to better adapt the jointed
discriminators and push generated images up to high resolutions. We adopt a
multi-purpose adversarial loss to encourage more effective image and text
information usage in order to improve the semantic consistency and image
fidelity simultaneously. Furthermore, we introduce a new visual-semantic
similarity measure to evaluate the semantic consistency of generated images.
With extensive experimental validation on three public datasets, our method
significantly improves previous state of the arts on all datasets over
different evaluation metrics.

AttnGAN: Fine-Grained Text to Image Generation with Attentional
  Generative Adversarial Networks <a href="http://arxiv.org/abs/1711.10485" target="_blank">http://arxiv.org/abs/1711.10485</a>
![1711.10485v1](http://www.arxiv-sanity.com/static/thumbs/1711.10485v1.pdf.jpg)
In this paper, we propose an Attentional Generative Adversarial Network
(AttnGAN) that allows attention-driven, multi-stage refinement for fine-grained
text-to-image generation. With a novel attentional generative network, the
AttnGAN can synthesize fine-grained details at different subregions of the
image by paying attentions to the relevant words in the natural language
description. In addition, a deep attentional multimodal similarity model is
proposed to compute a fine-grained image-text matching loss for training the
generator. The proposed AttnGAN significantly outperforms the previous state of
the art, boosting the best reported inception score by 14.14% on the CUB
dataset and 170.25% on the more challenging COCO dataset. A detailed analysis
is also performed by visualizing the attention layers of the AttnGAN. It for
the first time shows that the layered attentional GAN is able to automatically
select the condition at the word level for generating different parts of the
image.

### Image-to-Caption / Image Q&A

Bottom-Up and Top-Down Attention for Image Captioning and Visual
  Question Answering <a href="http://arxiv.org/abs/1707.07998" target="_blank">http://arxiv.org/abs/1707.07998</a>
![1707.07998v3](http://www.arxiv-sanity.com/static/thumbs/1707.07998v3.pdf.jpg)
Top-down visual attention mechanisms have been used extensively in image
captioning and visual question answering (VQA) to enable deeper image
understanding through fine-grained analysis and even multiple steps of
reasoning. In this work, we propose a combined bottom-up and top-down attention
mechanism that enables attention to be calculated at the level of objects and
other salient image regions. This is the natural basis for attention to be
considered. Within our approach, the bottom-up mechanism (based on Faster
R-CNN) proposes image regions, each with an associated feature vector, while
the top-down mechanism determines feature weightings. Applying this approach to
image captioning, our results on the MSCOCO test server establish a new
state-of-the-art for the task, achieving CIDEr / SPICE / BLEU-4 scores of
117.9, 21.5 and 36.9, respectively. Demonstrating the broad applicability of
the method, applying the same approach to VQA we obtain first place in the 2017
VQA Challenge.

A simple neural network module for relational reasoning <a href="http://arxiv.org/abs/1706.01427" target="_blank">http://arxiv.org/abs/1706.01427</a>
![1706.01427v1](http://www.arxiv-sanity.com/static/thumbs/1706.01427v1.pdf.jpg)
Relational reasoning is a central component of generally intelligent
behavior, but has proven difficult for neural networks to learn. In this paper
we describe how to use Relation Networks (RNs) as a simple plug-and-play module
to solve problems that fundamentally hinge on relational reasoning. We tested
RN-augmented networks on three tasks: visual question answering using a
challenging dataset called CLEVR, on which we achieve state-of-the-art,
super-human performance; text-based question answering using the bAbI suite of
tasks; and complex reasoning about dynamic physical systems. Then, using a
curated dataset called Sort-of-CLEVR we show that powerful convolutional
networks do not have a general capacity to solve relational questions, but can
gain this capacity when augmented with RNs. Our work shows how a deep learning
architecture equipped with an RN module can implicitly discover and learn to
reason about entities and their relations.

### Depth Estimation

Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps
  with Accurate Object Boundaries <a href="http://arxiv.org/abs/1803.08673" target="_blank">http://arxiv.org/abs/1803.08673</a> （Review）
![1803.08673v1](http://www.arxiv-sanity.com/static/thumbs/1803.08673v1.pdf.jpg)
We revisit the problem of estimating depth of a scene from its single RGB
image. Despite the recent success of deep learning based methods, we show that
there is still room for improvement in two aspects by training a deep network
consisting of two sub-networks; a base network for providing an initial depth
estimate, and a refinement network for refining it. First, spatial resolution
of the estimated depth maps can be improved using skip connections among the
sub-networks which are trained in a sequential fashion. Second, we can improve
estimation accuracy of boundaries of objects in scenes by employing the
proposed loss functions using depth gradients. Experimental results show that
the proposed network and methods improve depth estimation performance of
baseline networks, particularly for reconstruction of small objects and
refinement of distortion of edges, and outperform the state-of-the-art methods
on benchmark datasets.

### Face-Image-to-3D-Model

Evaluation of Dense 3D Reconstruction from 2D Face Images in the Wild <a href="http://arxiv.org/abs/1803.05536" target="_blank">http://arxiv.org/abs/1803.05536</a> （Review）
![1803.05536v2](http://www.arxiv-sanity.com/static/thumbs/1803.05536v2.pdf.jpg)
This paper investigates the evaluation of dense 3D face reconstruction from a
single 2D image in the wild. To this end, we organise a competition that
provides a new benchmark dataset that contains 2000 2D facial images of 135
subjects as well as their 3D ground truth face scans. In contrast to previous
competitions or challenges, the aim of this new benchmark dataset is to
evaluate the accuracy of a 3D dense face reconstruction algorithm using real,
accurate and high-resolution 3D ground truth face scans. In addition to the
dataset, we provide a standard protocol as well as a Python script for the
evaluation. Last, we report the results obtained by three state-of-the-art 3D
face reconstruction systems on the new benchmark dataset. The competition is
organised along with the 2018 13th IEEE Conference on Automatic Face & Gesture
Recognition.

另见：

Joint 3D Face Reconstruction and Dense Alignment with Position Map
  Regression Network <a href="http://arxiv.org/abs/1803.07835" target="_blank">http://arxiv.org/abs/1803.07835</a>
![1803.07835v1](http://www.arxiv-sanity.com/static/thumbs/1803.07835v1.pdf.jpg)
We propose a straightforward method that simultaneously reconstructs the 3D
facial structure and provides dense alignment. To achieve this, we design a 2D
representation called UV position map which records the 3D shape of a complete
face in UV space, then train a simple Convolutional Neural Network to regress
it from a single 2D image. We also integrate a weight mask into the loss
function during training to improve the performance of the network. Our method
does not rely on any prior face model, and can reconstruct full facial geometry
along with semantic meaning. Meanwhile, our network is very light-weighted and
spends only 9.8ms to process an image, which is extremely faster than previous
works. Experiments on multiple challenging datasets show that our method
surpasses other state-of-the-art methods on both reconstruction and alignment
tasks by a large margin.

Extreme 3D Face Reconstruction: Seeing Through Occlusions <a href="http://arxiv.org/abs/1712.05083" target="_blank">http://arxiv.org/abs/1712.05083</a>
![1712.05083v2](http://www.arxiv-sanity.com/static/thumbs/1712.05083v2.pdf.jpg)
Existing single view, 3D face reconstruction methods can produce beautifully
detailed 3D results, but typically only for near frontal, unobstructed
viewpoints. We describe a system designed to provide detailed 3D
reconstructions of faces viewed under extreme conditions, out of plane
rotations, and occlusions. Motivated by the concept of bump mapping, we propose
a layered approach which decouples estimation of a global shape from its
mid-level details (e.g., wrinkles). We estimate a coarse 3D face shape which
acts as a foundation and then separately layer this foundation with details
represented by a bump map. We show how a deep convolutional encoder-decoder can
be used to estimate such bump maps. We further show how this approach naturally
extends to generate plausible details for occluded facial regions. We test our
approach and its components extensively, quantitatively demonstrating the
invariance of our estimated facial details. We further provide numerous
qualitative examples showing that our method produces detailed 3D face shapes
in viewing conditions where existing state of the art often break down.

Video Based Reconstruction of 3D People Models <a href="http://arxiv.org/abs/1803.04758" target="_blank">http://arxiv.org/abs/1803.04758</a>
![1803.04758v3](http://www.arxiv-sanity.com/static/thumbs/1803.04758v3.pdf.jpg)
This paper describes how to obtain accurate 3D body models and texture of
arbitrary people from a single, monocular video in which a person is moving.
Based on a parametric body model, we present a robust processing pipeline
achieving 3D model fits with 5mm accuracy also for clothed people. Our main
contribution is a method to nonrigidly deform the silhouette cones
corresponding to the dynamic human silhouettes, resulting in a visual hull in a
common reference frame that enables surface reconstruction. This enables
efficient estimation of a consensus 3D shape, texture and implanted animation
skeleton based on a large number of frames. We present evaluation results for a
number of test subjects and analyze overall performance. Requiring only a
smartphone or webcam, our method enables everyone to create their own fully
animatable digital double, e.g., for social VR applications or virtual try-on
for online fashion shopping.

Disentangling Features in 3D Face Shapes for Joint Face Reconstruction
  and Recognition <a href="http://arxiv.org/abs/1803.11366" target="_blank">http://arxiv.org/abs/1803.11366</a>
![1803.11366v1](http://www.arxiv-sanity.com/static/thumbs/1803.11366v1.pdf.jpg)
This paper proposes an encoder-decoder network to disentangle shape features
during 3D face reconstruction from single 2D images, such that the tasks of
reconstructing accurate 3D face shapes and learning discriminative shape
features for face recognition can be accomplished simultaneously. Unlike
existing 3D face reconstruction methods, our proposed method directly regresses
dense 3D face shapes from single 2D images, and tackles identity and residual
(i.e., non-identity) components in 3D face shapes explicitly and separately
based on a composite 3D face shape model with latent representations. We devise
a training process for the proposed network with a joint loss measuring both
face identification error and 3D face shape reconstruction error. To construct
training data we develop a method for fitting 3D morphable model (3DMM) to
multiple 2D images of a subject. Comprehensive experiments have been done on
MICC, BU3DFE, LFW and YTF databases. The results show that our method expands
the capacity of 3DMM for capturing discriminative shape features and facial
detail, and thus outperforms existing methods both in 3D face reconstruction
accuracy and in face recognition accuracy.

### Image-to-3D-Model【太多了】

### 还有关于人脸的各种任务，待稍后补充
