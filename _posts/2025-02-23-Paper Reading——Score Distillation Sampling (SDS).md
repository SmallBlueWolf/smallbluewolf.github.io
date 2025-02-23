>  前言：SDS（分数蒸馏采样），目前几乎是文本生成 3D 模型的核心方法，因此写一篇 blog 来专门记录文献阅读过程
- Paper Link: [https://arxiv.org/abs/2209.14988](https://arxiv.org/abs/2209.14988) （Poole B, Jain A, Barron J T, et al. Dreamfusion: Text-to-3d using 2d diffusion[J]. arXiv preprint arXiv:2209.14988, 2022.）

## 原理部分

- 文本 to 图像的研究领域背景
	- 现在通过文本生成图像的 SOTA 工作基本都是基于 **Diffusion 模型**完成的

>  这里给没有听说过 Diffusion 模型的读者简单解释一下：Diffusion 模型具有非常强大的 2D 图像生成能力，其原理可以简单理解为：我们有一组用于训练的真实图像，将它们的像素不断随机移动，使其变成噪声图像；把这个过程告诉模型，训练模型尝试执行这一过程的逆过程——也即让模型尝试将噪声图像变回正常图像；因此，我们给训练好的模型一张噪声图，它就可以慢慢将其变成我们想要的图像
>  具体内容可以查看原文献：[https://arxiv.org/abs/1503.03585](https://arxiv.org/abs/1503.03585) （Sohl-Dickstein J, Weiss E, Maheswaranathan N, et al. Deep unsupervised learning using nonequilibrium thermodynamics[C]//International conference on machine learning. pmlr, 2015: 2256-2265.）

>  文本生成 2D 图像中，Diffusion 模型已经表现出了非常卓越的性能（如 Stable Diffusion，你能在网上看到的大多数文生图模型等），但是我们是否可以将其延用到文本生成 3D 模型任务中呢？
>  答：首先，2D 图像的标注数据集目前已经非常广泛了，但是标注的 3D 数据集或者多视角的图像数据，目前仍然稀缺且获取成本高；因此，目前的研究主要都集中于如何利用现有的文本 to 2D 图像的模型能力迁移到 3D 生成任务中

- 作者列出了两种基于 Diffusion 模型生成 3D 模型的思路
	1. “Using the generated multi-view images as inputs for a few-shot 3D reconstruction method.”
		- 第一种很好理解，用一个文生图模型生成一个 3D 模型多个视角的图像，将其变成一个“多视角图像的三维重建”任务
		- 但是问题在于，我们很难保证 Diffusion 模型生成的一组多视角图像的连贯性和稳定性，仅仅用 Prompts 去控制是非常不稳定的
	2. “Using the multi-view diffusion model as a prior for Score Distillation Sampling.”
		- 第二种就是，用多视角的 Diffusion 模型作为 SDS 过程的先验，这个可能比较难理解，我们可以先来看一下文献给出的 Pipeline ![Pasted image 20250223125753.png|800](https://raw.githubusercontent.com/SmallBlueWolf/smallbluewolf.github.io/main/_static/2025-02-23-Paper%20Reading——Score%20Distillation%20Sampling%20(SDS)/Pasted%20image%2020250223125753.png|800)
			- 从左往右看，首先随机初始化一个 NeRF，由一个 MLP 来参数化表示体积密度和反射率（颜色），也即图中的 $\text{MLP}(\cdot;\theta)$
			- 在球坐标系中，随机采样一个相机参数 P（包含相机在空间中的位置和方向数据 (x,y,z,d) ）；同时，在相机周围随机采样一个点光源
			- 根据相机参数 P 和光源位置，对这个 NeRF 进行渲染，生成当前相机视角下的 2D 图像，将这个图像送入 Diffusion 模型中
			- 首先对送来的图像加噪，然后用一个 Diffusion 模型中预训练好的 U-Net 接收用户输入的文本描述，对加噪声后的图像进行预测，得到一个预测的噪声值（理想情况下，如果 NeRF 渲染的图像已经较为接近输入文本的描述，那么 U-Net 预测的噪声应该与实际添加的噪声非常接近）
			- 利用 Diffusion 模型中常用的噪声预测 loss，计算输入文本对应的噪声和实际添加噪声之间的差异 $$
L_{\text{NeRF}}=E[w(t)||UNet(\alpha_{t}x_{render}+\sigma_{t}\epsilon|t)-\epsilon||^{2}]\quad with\quad x_{render}=NeRF(camera, \theta)
$$
			- 我们的 NeRF 表示主要是由一个 $\text{MLP}(\cdot;\theta)$ 参数化体积密度和反射率构成的，我们用这个 loss 来调整 NeRF 的表示，使得场景在该随机化（随机相机参数和随机光源）采样的 2D 图像更加贴近于输入的文本描述
			- 最后，我们不断重复这一过程——在不同的随机采样视角下不断重复，每次迭代使得 NeRF 渲染的图像与文本描述的匹配度提高
- 总结一下， SDS 的目标，是最小化一个基于 KL 散度（相对熵）的概率密度蒸馏损失，通过比较在不同噪声水平下渲染图像与 Diffusion 模型预测的噪声之间的差异来指导 3D 模型参数的更新，从而将 2D Diffusion 模型的 prior 传递到 3D 生成过程中，而无需专门训练 3D Diffusion 模型

>  需要说明的是，U-Net 预测的 loss 主要反向传播给 NeRF，U-Net 作为 Diffusion 中预训练好的结构，本身它的梯度计算复杂度较高，而且在实践上来说并不是十分重要，因此 DreamFusion 将其参数冻结，因此 loss 全部用于更新 NeRF，不再更新 U-Net 的参数，在实验中仍然能够有比较好的效果

- 重点再解释一下 SDS 损失函数 $$
\mathcal{L}_{\text{SDS}} = \mathbb{E}_{t,\epsilon,c}\Bigl[ w(t)\,\Bigl\|\epsilon - \epsilon_\phi\Bigl(x_0 + \sigma(t)\,\epsilon,\; t,\; c\Bigr)\Bigr\|_2^2\Bigr]
$$ 
	-  $x_0$ ：由当前 NeRF 模型渲染得到的图像
	-  $t$ ：采样的噪声时间步（通常服从均匀分布），决定噪声级别
	- $\epsilon \sim \mathcal{N}(0, I)$：从标准正态分布中采样的噪声
	-  $x_t = x_0 + \sigma(t)\,\epsilon$ ：在  $x_0$  上加入噪声后得到的图像，其中  $\sigma(t)$  是噪声尺度
	-  $\epsilon_\phi(x_t, t, c)$ ：预训练的扩散模型（例如 U-Net）在给定噪声图像  $x_t$ 、时间步 $ t$  以及文本条件  $c$  下预测得到的噪声
	-  $w(t)$ ：权重函数，用来平衡不同噪声级别下的损失（在部分实现中可以取 1）
	-  $c$ ：文本条件，即输入的自然语言描述，通过扩散模型引入语义指导
- 直观地说，如果 NeRF 渲染得到的图像 $x_{0}$ 与文本描述很相符，那么加入噪声后，U-Net 预测的噪声 $\epsilon_\phi(\cdot)$ 应该与实际加入的噪声 $\epsilon$ 非常接近，计算二者的均方误差得到这个 loss
- 因此，我们希望对于 NeRF 在随机采样的各个视角下看到的 2D 图像都和文本描述比较相符，而我们利用 Diffusion 模型中成熟的 loss 来衡量渲染的 2D 图像和文本描述之间的匹配程度
	- 可能你会意识到，这存在一个问题——文本 prompt 通常只能描述一个物体的典型视图，并不能很好的描述所有视图——因此，我们可以考虑在不同视角下采样时，对输入文本进行对应视角下的调整
	- 例如，对于仰角大于 $60°$ 时，在文本中添加“overhead view”的 prompt 等；也可以通过前视图、侧视图、后视图的方位角的加权组合来更进一步调整 prompt


>  这里也给没有听说过 NeRF 的读者简单解释一下：NeRF (Neural Radiance Fields)，神经网络辐射场，一种利用神经网络来**表示**和生成 3D 场景的方法，它希望将 3D 场景的细节编码到一个神经网络中，从而用这个神经网络来表示这个 3D 场景——你输入该场景中一个点的坐标以及观察角度，理想情况下，它就会输出在该视角下观察时，该点的颜色和透明度；这样的话，你可以得到某一个视角下，该场景所有点看上去应该是什么样，从而得到这个视角下看到的该 3D 场景的 2D 图像。
>  具体原理如果感兴趣可以看看原文献：[https://arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934) （Mildenhall B, Srinivasan P P, Tancik M, et al. Nerf: Representing scenes as neural radiance fields for view synthesis[J]. Communications of the ACM, 2021, 65(1): 99-106.）

## 实验部分

- 在实验中，3D 场景在一台 4 核的 TPUv4 机器上进行优化，每个芯片单独渲染一个视图，并计算 U-Net 的推理任务，每个设备的 batch_size=1，优化了 15,000 次迭代，大约需要 1.5 小时完成

- 评估指标： 
	- 由于不存在真实的 3D 地面真值，作者采用了基于 CLIP 的 R-Precision 指标来衡量生成的 3D 模型在不同视角下渲染出的图像与输入文本描述之间的一致性
	- 实验结果显示，DreamFusion 在生成图像质量和 3D 几何一致性方面均优于先前的方法，如 Dream Fields 和 CLIP-Mesh 
    
- 消融实验：  
    论文通过一系列消融实验展示了各个组件（如视角增强、视角依赖文本提示、光照条件和纹理无关渲染）的重要性
    结果表明，每一项改进都对提升 3D 几何的准确性和渲染图像的真实感有显著作用


## 结论

- SDS 的优点
	- 不使用 3D 训练数据，仅使用预训练的 2D Diffusion 模型
	- SDS 方法冻结了 U-Net 参数，也即不需要对预训练的 Diffusion 模型修改
- SDS 的局限性
	- 文献中具体使用的是 $64\times 64$ 的 2D Diffusion 模型，生成的 3D 模型在细节上存在一定不足
	- SDS 具有一定的模式趋向性，可能导致生成结果多样性不足，同时在某些情况下会出现过度平滑或过饱和的问题