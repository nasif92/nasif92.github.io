# Final Exam Practice Questions — Complete Solutions

**Computer vision and deep learning | Worked solutions and revision guide**

Based on the supplied PracticeQuestions_for_FinalExam_2026 (1).docx (the internal title says 2025). All 44 listed questions are answered. Q23 and Q24 are absent from the source. True/False examples are not supplied. The explanations below are worked solutions, not an official marking key.

## Part I — Mathematics and backpropagation (Q1–Q13)

### Q1. Compute the forward pass: convolution → max pooling → ReLU

The input table is displaced in the original document's diagram; its actual entries are reproduced below. Use the deep-learning convention of cross-correlation (no kernel flip), as in PyTorch Conv2d. Assume no bias.

$$I=\begin{bmatrix}1&3&2&4&6&4\\4&8&3&1&0&2\\2&1&4&3&9&1\\4&7&2&3&9&2\end{bmatrix},\quad W=\begin{bmatrix}-1&0&1\\-1&0&1\\-1&0&1\end{bmatrix}$$

**Step 1 — Convolution.** With stride 1, no padding, and a 3 × 3 filter, the output height is 4 − 3 + 1 = 2 and width is 6 − 3 + 1 = 4.

$$J_{ij}=\sum_{a=0}^{2}\sum_{b=0}^{2} I_{i+a,j+b}W_{ab}$$

Because the middle column of W is zero, every output is the sum of the right column minus the sum of the left column in the corresponding input patch.

| Output position (1-based) | Calculation | Result |
|---|---|---:|
| J₁₁ | (2+3+4) − (1+4+2) | 2 |
| J₁₂ | (4+1+3) − (3+8+1) | −4 |
| J₁₃ | (6+0+9) − (2+3+4) | 6 |
| J₁₄ | (4+2+1) − (4+1+3) | −1 |
| J₂₁ | (3+4+2) − (4+2+4) | −1 |
| J₂₂ | (1+3+3) − (8+1+7) | −9 |
| J₂₃ | (0+9+9) − (3+4+2) | 9 |
| J₂₄ | (2+1+2) − (1+3+3) | −2 |

$$J=\begin{bmatrix}2&-4&6&-1\\-1&-9&9&-2\end{bmatrix}$$

**Step 2 — Max pooling.** Use two non-overlapping 2 × 2 windows (stride 2):

- First window: max(2, −4, −1, −9) = 2, at J₁₁.
- Second window: max(6, −1, 9, −2) = 9, at J₂₃.

$$K=[2,9]$$

**Step 3 — ReLU.** ReLU(x) = max(0,x). Both values are positive, so:

$$\boxed{O=[2,9]}$$

**Final answer:** J is 2 × 4; K and O are both 1 × 2. Save the positions of the maxima for Q2.

### Q2. Backpropagation through ReLU, max pooling and convolution

Target IO = [7,10]. Define the loss and use Q1's output:

$$L=\frac12\|O-IO\|_2^2=\frac12[(2-7)^2+(9-10)^2]=13$$

Let G_X denote the gradient of L with respect to X (same shape as X). We propagate backwards in the order O → K → J → W and I.

**Step 1 — Output gradient.** Differentiating the squared loss cancels the factor 1/2:

$$G_O=O-IO=[-5,-1]$$

**Step 2 — ReLU gradient.** The derivative is 1 for positive inputs and 0 for negative inputs (usually 0 at zero in implementations). Both K entries are positive.

$$G_K=G_O\odot\mathbf{1}_{K>0}=[-5,-1]$$

**Step 3 — Max pooling gradient.** Route each upstream gradient only to the winning element of its pooling window. The other elements receive zero.

$$G_J=\begin{bmatrix}-5&0&0&0\\0&0&-1&0\end{bmatrix}$$

**Step 4 — Filter gradient.** Each output Jᵢⱼ is a weighted sum of its 3 × 3 input patch. Differentiating with respect to a filter entry gives the corresponding input entry. Since W is shared, add contributions from all positions:

$$\frac{\partial L}{\partial W_{ab}}=\sum_{i,j}(G_J)_{ij}I_{i+a,j+b}$$

Only two patches contribute (top-left at row 1, column 1 and row 2, column 3, using 1-based indexing).

$$G_W=-5\begin{bmatrix}1&3&2\\4&8&3\\2&1&4\end{bmatrix}-1\begin{bmatrix}3&1&0\\4&3&9\\2&3&9\end{bmatrix}$$

$$\boxed{G_W=\begin{bmatrix}-8&-16&-10\\-24&-43&-24\\-12&-8&-29\end{bmatrix}}$$

**Step 5 — Input gradient.** Initialize a 4 × 6 zero matrix. For each nonzero G_J entry, scatter-add its value times W into the input region that produced that entry.

- Add −5W at rows 1–3, columns 1–3.
- Add −W at rows 2–4, columns 3–5.
- The overlapping entries must be added, not overwritten. For example, row 2, column 3 receives −5 + 1 = −4.

$$\boxed{G_I=\begin{bmatrix}5&0&-5&0&0&0\\5&0&-4&0&-1&0\\5&0&-4&0&-1&0\\0&0&1&0&-1&0\end{bmatrix}}$$

**Exam tip:** These are gradients, not updated parameters. If gradient descent with learning rate η is requested, W_new = W − ηG_W. If a shared scalar convolution bias were included, its gradient would be the sum of G_J = −6. I is normally an input, not a learnable parameter, but its gradient can be passed to earlier layers.

### Q3. Chain rule for a two-layer neural network with L2 loss

The printed equation omits σ even though the question defines sigmoid. First solve the likely intended network (hidden sigmoid), then give the literal linear-network case. Use row vectors and half squared L2 loss.

Let x have shape 1 × d, W₁ be d × h, b₁ be 1 × h, W₂ be h × c, b₂ and y be 1 × c.

**Forward pass:**

$$z_1=xW_1+b_1,\quad a_1=\sigma(z_1),\quad y_p=a_1W_2+b_2$$
$$L=\frac12\sum_{k=1}^{c}(y_{p,k}-y_k)^2$$

The sigmoid and its derivative are:

$$\sigma(z)=\frac1{1+e^{-z}},\qquad \sigma'(z)=\sigma(z)(1-\sigma(z))$$

**Backpropagate the output error:**

$$\delta_2=\frac{\partial L}{\partial y_p}=y_p-y$$

Since y_p = a₁W₂ + b₂, the output-layer derivatives are:

$$\boxed{\frac{\partial L}{\partial W_2}=a_1^T\delta_2},\qquad \boxed{\frac{\partial L}{\partial b_2}=\delta_2}$$

The hidden activation receives gradient δ₂W₂ᵀ. Multiply elementwise by the sigmoid derivative to get the preactivation gradient:

$$\delta_1=(\delta_2W_2^T)\odot a_1\odot(1-a_1)$$

Finally, because z₁ = xW₁ + b₁:

$$\boxed{\frac{\partial L}{\partial W_1}=x^T\delta_1},\qquad \boxed{\frac{\partial L}{\partial b_1}=\delta_1}$$

| Gradient | Shape |
|---|---|
| a₁ᵀδ₂ | h × c |
| δ₂ | 1 × c |
| xᵀδ₁ | d × h |
| δ₁ | 1 × h |

**Literal equation with no sigmoid:** Set a₁ = z₁ and δ₁ = δ₂W₂ᵀ; all four boxed equations remain the same. If the intended sigmoid is instead on the output, multiply the output error by y_p ⊙ (1−y_p) before propagating it. If both layers use sigmoid, include both derivatives.

**Loss convention:** If L = ||y_p − y||² instead, multiply every gradient by 2. If L is the unsquared Euclidean norm, use δ₂ = (y_p − y)/||y_p − y|| for nonzero error. For a batch, sum per-example bias gradients and sum or average all gradients according to the loss reduction.

### Q4. Compute softmax for MNIST logits

Given logits z = [−0.1, 12.0, 3.0, −3.2, −8.0, 17.0, 1.9, 0.6, 4.9, −12.0]. Softmax converts arbitrary real scores to positive probabilities summing to 1:

$$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}=\frac{e^{z_i-17}}{\sum_j e^{z_j-17}}$$

Subtracting the largest logit prevents numerical overflow without changing the result. The shifted exponential sum is approximately 1.006744728.

| Digit | Logit | Softmax probability |
|---:|---:|---:|
| 0 | −0.1 | 3.72087427 × 10⁻⁸ |
| 1 | 12.0 | 0.00669280583 |
| 2 | 3.0 | 8.25957857 × 10⁻⁷ |
| 3 | −3.2 | 1.67622418 × 10⁻⁹ |
| 4 | −8.0 | 1.37949010 × 10⁻¹¹ |
| 5 | 17.0 | 0.993300457 |
| 6 | 1.9 | 2.74937487 × 10⁻⁷ |
| 7 | 0.6 | 7.49292063 × 10⁻⁸ |
| 8 | 4.9 | 5.52226704 × 10⁻⁶ |
| 9 | −12.0 | 2.52662425 × 10⁻¹³ |

**Answer:** Predicted class is digit 5 with about 99.33% probability. Softmax preserves the ordering of logits; argmax is unchanged.

### Q5. Compute cross-entropy loss

Ground truth y = [0,1,0,0,0,0,0,0,0,0], so the correct class is digit 1, not digit 5.

$$L=-\sum_{i=0}^{9}y_i\ln p_i=-\ln(0.00669280583)\approx\boxed{5.00672209}$$

Only the true class contributes because the target is one-hot. Equivalently, use log-sum-exp directly: L = 17 − 12 + ln(1.006744728). PyTorch CrossEntropyLoss expects raw logits, not already-softmaxed probabilities.

### Q6. Minimum and maximum cross-entropy

**Answer:** For a one-hot target, the minimum is 0 and there is no finite maximum (unbounded above).

Since L = −ln(p_true), a correct class probability of 1 gives zero loss, while a probability approaching zero makes the loss approach +∞. With finite softmax logits, probabilities are strictly between 0 and 1, so zero is a limiting value. For soft labels, the minimum is the entropy of the target distribution, not necessarily zero.

### Q7. Learnable parameters in 2 × 2 max pooling

**Answer: 0.** Max pooling applies a fixed maximum operation; its kernel size and stride are hyperparameters, not trainable weights. It has no bias.

### Q8. Output dimensions of four 3 × 3 convolution filters

Input shape: (b, 10, 28, 28), using PyTorch NCHW convention (batch, channels, height, width). Each of the four filters spans all ten input channels.

For dilation 1, convolution output size along an axis is:

$$n_{out}=\left\lfloor\frac{n+2p-k}{s}\right\rfloor+1$$

Valid convolution means padding p = 0. With k = 3 and s = 1, output height and width are both 28 − 3 + 1 = 26.

**Answer:** (b, 4, 26, 26) in NCHW; equivalently (batch, height, width, depth) = (b, 26, 26, 4).

### Q9. Output dimensions of four 1 × 1 convolution filters

The spatial output is 28 − 1 + 1 = 28. There are four output channels because there are four filters, each spanning ten input channels.

**Answer:** (b, 4, 28, 28) in NCHW; or (b, 28, 28, 4) in batch-height-width-depth order.

A 1 × 1 convolution mixes channel values at each spatial location without aggregating neighboring locations.

### Q10. Learnable parameters in Q8 and Q9

For a standard, ungrouped convolution with bias, parameter count is:

$$C_{out}(C_{in}k_hk_w+1)$$

| Layer | Weights | Biases | Total |
|---|---:|---:|---:|
| Q8: 3 × 3 | 4 × 10 × 3 × 3 = 360 | 4 | **364** |
| Q9: 1 × 1 | 4 × 10 × 1 × 1 = 40 | 4 | **44** |

If bias=False, the answers are 360 and 40. Parameter count is independent of batch size and spatial activation size because weights are shared.

### Q11. Parameter count for filter size 5 × 5 × 10 × 5

The question states (in_channels, out_channels, height, width), so use C_in = 5, C_out = 5, k_h = 10, k_w = 5.

Weights = 5 × 5 × 10 × 5 = 1,250. Biases = 5.

**Answer: 1,255 with biases, or 1,250 without biases.** Note that actual PyTorch Conv2d weight layout is (out_channels, in_channels/groups, kH, kW); here swapping the first two dimensions makes no numerical difference.

### Q12. Halve spatial resolution using convolution

Assume h and w are even. Specify a standard convolution with in_channels = d, out_channels = d, kernel_size = 2, stride = 2, padding = 0, dilation = 1.

$$h_{out}=\left\lfloor\frac{h-2}{2}\right\rfloor+1=h/2$$

Likewise w_out = w/2. An alternative is kernel 3, stride 2, padding 1, giving the same output dimensions when h and w are even.

**Answer:** Conv2d(d, d, kernel_size=2, stride=2, padding=0). Unlike max pooling, this operation learns how to combine local features. For odd input sizes, exact integer halving requires specifying the rounding convention.

### Q13. Learnable parameters in average pooling

**Answer: 0.** A 2 × 2 average pooling layer computes (a+b+c+d)/4 using fixed coefficients. Stride and kernel size are hyperparameters. No weights or biases are learned.

---

## Part II — Short-answer and conceptual questions

Each question starts with a concise exam-ready response, followed by a more detailed explanation for revision.

### Q14. Why machine learning instead of rule-based AI for image classification?

**Exam answer:** Images exhibit substantial variation in lighting, orientation, scale, background and appearance, making it difficult to manually define rules for every class. Machine learning learns discriminative patterns from examples and can generalize to previously unseen images.

**Explanation:** A rule such as “a cat has pointed ears” fails for occlusions, unusual viewpoints or other animals with pointed ears. A neural network can learn a hierarchy of edges, textures, parts and semantic features. Generalization still depends on representative training data and appropriate evaluation.

### Q15. How does k-nearest neighbors (k-NN) work?

**Exam answer:** k-NN finds the k training examples closest to a query according to a distance measure, such as Euclidean distance. For classification, the predicted class is the majority label of those neighbors, optionally weighted by distance.

**Explanation:** Suppose the five closest examples have labels A, A, B, A, B; the prediction is A. Small k can be sensitive to noise; large k may smooth away meaningful boundaries. Feature scaling matters because large-scale features can dominate distances. For regression, average the neighbors' target values instead.

### Q16. Explain underfitting and overfitting

**Exam answer:** Underfitting occurs when a model fails to learn important patterns, resulting in poor training and validation performance. Overfitting occurs when a model learns training-specific details or noise, achieving strong training performance but comparatively poor validation performance.

**Explanation:** Underfitting is commonly associated with high bias; overfitting with high variance. A large train–validation gap may also be caused by distribution shift, so check the data split and preprocessing before drawing conclusions.

### Q17. How would you overcome underfitting in a CNN?

**Exam answer:** Increase model capacity by adding layers or filters, train longer with a suitable learning rate, or reduce excessive regularization. Also verify that the training pipeline, labels and preprocessing are correct.

**Explanation:** If training loss remains high, inspect optimization first: learning rate, gradients, normalization and data quality. More parameters are useful only if insufficient capacity is the cause. Strong dropout, excessive weight decay or excessive augmentation can also prevent fitting.

### Q18. How would you tackle CNN overfitting?

**Exam answer:** Use more representative training data, label-preserving augmentation, weight decay, dropout or early stopping to improve generalization. Reduce excessive model capacity or use pretrained features where appropriate.

**Explanation:** Monitor validation loss during training and select a checkpoint using validation data, not test data. Augmentations such as flips or crops help only when they preserve the class label. Prevent leakage between train and validation sets; in medical imaging, split by patient when appropriate.

### Q19. When is gradient descent preferred over closed-form linear regression?

**Exam answer:** Gradient descent is preferable when the feature dimension or dataset is very large, making the closed-form solution computationally or memory expensive. Mini-batch gradient descent also supports streaming data and iterative optimization without forming a large feature covariance matrix.

**Explanation:** For X with n examples and d features, the normal equation is θ = (XᵀX)⁻¹Xᵀy when X has full column rank. Forming XᵀX costs roughly O(nd²), and solving the dense system costs O(d³); explicit inversion is usually discouraged. QR, SVD or least-squares solvers are more numerically stable alternatives. A full gradient costs O(nd) per iteration; mini-batches reduce per-step cost but convergence depends on conditioning and learning rate.

### Q20. Why are CNNs more suitable than fully connected networks for visual recognition?

**Exam answer:** CNNs exploit spatial locality by processing neighboring pixels together and reuse filters across the image through weight sharing. This reduces parameters and provides translation-equivariant features, making learning visual patterns more efficient.

**Explanation:** A learned edge filter can respond to an edge anywhere in the image. Stacking layers increases receptive field and enables hierarchical feature extraction. Pooling or global aggregation can provide some translation tolerance, but CNNs are not automatically invariant to all translations, rotations or scales.

### Q21. Why do convolutional layers have fewer parameters than fully connected layers?

**Exam answer:** A convolutional neuron connects only to a local receptive field, and the same filter weights are shared across spatial positions. A fully connected layer instead uses independent weights for every input-output pair.

**Explanation:** Mapping a 28 × 28 single-channel input to four 26 × 26 output maps by dense connections would require 784 × 2,704 + 2,704 = 2,122,640 parameters. A 3 × 3 convolution producing the same output shape needs just 4 × (9+1) = 40 parameters. Equal shape does not imply equal representational freedom; convolution constrains the connections.

### Q22. How do you reduce activation map dimensions?

**Exam answer:** Reduce spatial height and width using pooling or a strided convolution. Reduce depth (number of channels) using a convolution with fewer output filters, commonly a 1 × 1 convolution.

**Explanation:** An input (b,64,32,32) becomes (b,64,16,16) after 2 × 2 pooling with stride 2. A subsequent 1 × 1 convolution with 16 filters gives (b,16,16,16). Spatial resolution and channel depth are different quantities.

*Q23 and Q24 are not present in the supplied question sheet.*

### Q25. Main differences between ResNet and AlexNet

**Exam answer:** AlexNet is a relatively shallow sequential CNN with five convolutional and three fully connected learnable layers in its original architecture. ResNet uses residual blocks with shortcut connections, allowing much deeper networks to be optimized more effectively; standard ResNets also use batch normalization and global average pooling.

**Explanation:** A residual block learns F(x) and outputs F(x)+x (or uses a projection when shapes differ). The shortcut provides a direct gradient route. AlexNet uses large early filters, max pooling and large dense classification layers; ResNet commonly uses smaller filters in blocks and a compact classification head. A deeper model does not necessarily have more parameters.

### Q26. Which requires more memory: DenseNet or ResNet?

**Exam answer:** DenseNet typically requires more activation memory than a comparable ResNet in a naive implementation, because each layer concatenates feature maps from all preceding layers within a dense block. ResNet generally adds shortcut features rather than concatenating all previous outputs.

**Explanation:** DenseNet computes x_l = H_l([x_0,...,x_{l−1}]); channel depth grows by the growth rate at every layer in the block. ResNet computes x_{l+1} = x_l + F_l(x_l). DenseNet can nevertheless have fewer parameters, and memory-efficient implementations, checkpointing, depth, width and batch size can change the actual comparison.

### Q27. What is gained or lost by converting fully connected layers to fully convolutional layers?

**Exam answer:** An equivalent convolutional conversion preserves the original computation on the original input size while enabling evaluation on larger spatial inputs and producing spatial score maps. It also permits shared computation for overlapping receptive fields rather than independently classifying each sliding window.

**Explanation:** A dense layer mapping a C × H × W tensor to M outputs can be replaced by M filters of size H × W spanning C channels. Later dense layers acting on 1 × 1 outputs become 1 × 1 convolutions. On a larger input, the converted network gives a grid rather than necessarily one global prediction; additional aggregation may be needed. Input sizes must still be sufficiently large and respect architectural constraints. Exact conversion loses no information on the original input.

### Q28. How does parameter count change under an equivalent FC-to-convolutional conversion?

**Exam answer:** It does not change when the weights and biases are reshaped into exactly equivalent convolutional filters.

**Explanation:** A fully connected layer with CHW input features and M outputs has M(CHW+1) parameters. A convolution with M filters of shape C × H × W has exactly the same count. Replacing the dense layer with a different smaller kernel or global average pooling can reduce parameters, but that is an architectural change rather than exact conversion.

### Q29. Why is YOLO faster than Faster R-CNN?

**Exam answer:** YOLO is a one-stage detector that directly predicts bounding boxes and class scores from dense image features. Faster R-CNN first generates region proposals and then processes proposed regions in a second detection stage, typically creating more computational overhead.

**Explanation:** Faster R-CNN shares backbone features between its stages, so it is not running a complete CNN separately for every proposal. YOLO removes the explicit proposal-and-refinement pipeline. Actual speed depends on model variant, image resolution, hardware and implementation.

### Q30. What is the role of a region proposal network (RPN)?

**Exam answer:** An RPN predicts candidate object regions from a shared feature map by estimating objectness scores and bounding-box offsets. These proposals are passed to a subsequent detection head for object classification and further bounding-box refinement.

**Explanation:** In standard Faster R-CNN, the RPN operates on anchors, predicts foreground versus background and adjusts anchor coordinates. High-scoring proposals are filtered, commonly by non-maximum suppression, before RoI feature extraction. An RPN is class-agnostic in its usual formulation.

### Q31. Why use a fully convolutional network for semantic segmentation?

**Exam answer:** Semantic segmentation requires a class prediction for every pixel rather than one class label for an entire image. A fully convolutional architecture retains spatial correspondence and can use upsampling to produce dense class-score maps aligned with the input.

**Explanation:** A classifier returns C class scores per image; a segmentation network typically returns C × H × W logits per image. Pixelwise softmax and argmax yield a class mask. Downsampling alone reduces resolution, so a decoder or suitable upsampling mechanism is needed.

### Q32. Role of transposed convolution in semantic segmentation

**Exam answer:** Transposed convolution is a learnable upsampling operation that increases the spatial resolution of low-resolution feature maps. It helps the decoder recover dense pixel-level predictions from high-level semantic representations.

**Explanation:** For dilation 1, output size is (n−1)s − 2p + k + output_padding. A 2 × 2 transposed convolution with stride 2 and zero padding doubles the size. It is the transpose of the linear convolution operator, not generally its inverse. Uneven overlap may produce checkerboard artifacts; interpolation followed by convolution is an alternative.

### Q33. What is a skip connection?

**Exam answer:** A skip connection passes features from an earlier layer directly to a later layer, bypassing one or more intermediate transformations. It improves gradient flow or preserves useful earlier features, depending on the architecture.

**Explanation:** ResNet uses elementwise addition x+F(x). U-Net commonly concatenates encoder features with corresponding decoder features to recover spatial detail. Addition needs matching shapes (or a projection); concatenation increases the channel count when joining channels.

### Q34. Why is a reconstruction-only autoencoder unsupervised?

**Exam answer:** An autoencoder uses the input itself as the reconstruction target and does not require externally supplied class labels or annotations. It learns a representation by minimizing the discrepancy between the original input and its reconstruction.

**Explanation:** With encoder E and decoder D, optimize ||x−D(E(x))||² or another reconstruction criterion. This is traditionally called unsupervised representation learning and can also be described as self-supervised because the target is derived from the input.

### Q35. Why does an autoencoder try to implement the identity function?

**Exam answer:** The training objective encourages its reconstructed output to equal its input, so the encoder-decoder composition approximates the identity mapping on the training distribution. A bottleneck or other constraint forces it to learn useful representations instead of simply copying every input.

**Explanation:** D(E(x)) ≈ x does not mean E(x)=x. The latent code may be much smaller, sparse, noisy or otherwise constrained. An unconstrained high-capacity autoencoder may learn a trivial identity with little useful representation learning.

### Q36. Why is a variational autoencoder (VAE) a generative model?

**Exam answer:** A VAE learns a probabilistic latent representation and a decoder that maps latent samples into the data space. Sampling from the learned model's latent prior and decoding allows it to generate new examples resembling the training distribution.

**Explanation:** The encoder predicts parameters of q(z|x), often mean μ and diagonal variance σ². Use z=μ+σ⊙ε with ε∼N(0,I) for differentiable sampling. The typical negative ELBO combines reconstruction loss with KL(q(z|x)||p(z)), encouraging compatibility with a prior such as N(0,I). At generation time, sample z from the prior, not from an existing image's encoder.

### Q37. Why can diffusion models be superior to GANs for image generation?

**Exam answer:** Diffusion models often provide strong sample quality and diversity with a stable denoising training objective, avoiding the adversarial optimization instability and mode collapse associated with GANs. Their main disadvantage is typically slower generation because sampling involves multiple denoising steps.

**Explanation:** GANs train two competing networks, so training can become imbalanced. Diffusion learns to reverse a gradual corruption process, often using a noise-prediction or equivalent objective. Superiority is not universal: GANs can be much faster at inference and can perform very well for particular tasks.

### Q38. Differences between flow matching and diffusion for image generation

**Exam answer:** Diffusion models learn to reverse a progressive noise corruption process, often by estimating noise or a score function. Flow matching directly trains a time-dependent velocity field that transports a simple source distribution into the data distribution, typically using an ODE at inference.

**Explanation:** A standard diffusion forward marginal can be written x_t = α_t x_data + σ_t ε, with a schedule controlling noise. A common denoising objective is E||ε−ε_θ(x_t,t)||². Reverse sampling can use a stochastic differential equation or a deterministic probability-flow ODE.

For a simple linear flow-matching path from noise x₀ to data x₁:

$$x_t=(1-t)x_0+tx_1,\quad u_t=x_1-x_0$$
$$L_{FM}=\mathbb E\|v_\theta(x_t,t)-u_t\|^2$$

At inference, solve dx_t/dt = v_θ(x_t,t) from t=0 to 1. The conditional path is straight in this example, but the learned marginal trajectories need not be. Other flow paths are possible, and flow matching can use diffusion-like probability paths. Both are related continuous-time generative approaches, not entirely disjoint families.

| Aspect | Typical diffusion | Typical flow matching |
|---|---|---|
| Training target | Noise, score, or equivalent parameterization | Velocity along a prescribed probability path |
| Generation | Reverse SDE or ODE | ODE integration |
| Path | Noise-schedule-based corruption | Chosen interpolation, e.g. linear |
| Speed | Depends on sampler and steps | Can use efficient paths and solvers; not always faster |

### Q39. Why do diffusion and flow matching use time-embedded architectures?

**Exam answer:** The correct denoising operation or velocity depends on the current noise level or position along the generation trajectory. Time embeddings inform a shared network which stage it is processing so that it can learn different transformations at different times.

**Explanation:** A nearly pure-noise image needs large-scale structural recovery; a nearly clean image needs fine detail adjustments. A network receiving x_t alone does not explicitly know the noise schedule stage. Time t is often embedded using sinusoidal features or a learned embedding and injected into network blocks by addition, scale/shift or conditioning. Time here is generation-process time, not video time or training epoch.

### Q40. Why use an autoencoder's latent space in diffusion or flow matching?

**Exam answer:** An autoencoder compresses images into a smaller latent representation, allowing diffusion or flow matching to operate at lower spatial resolution and computational cost. The decoder then converts generated latent samples back to image space.

**Explanation:** Train or obtain an encoder E and decoder D; encode images, train the generative process on z=E(x), then generate z and decode D(z). This reduces memory and compute but introduces a reconstruction bottleneck; information lost by the encoder cannot reliably be recovered by the decoder.

### Q41. Working principles of GANs

**Exam answer:** A generative adversarial network consists of a generator that transforms random noise into synthetic samples and a discriminator that distinguishes real data from generated data. Through adversarial training, the generator learns to produce increasingly realistic samples that fool the discriminator.

**Explanation:** The original minimax objective is:

$$\min_G\max_D\;\mathbb E_{x\sim p_{data}}[\log D(x)]+\mathbb E_{z\sim p_z}[\log(1-D(G(z)))]$$

The discriminator maximizes its classification objective, while the generator aims to make generated examples look real. In practice the generator often uses the non-saturating loss −E log D(G(z)). At inference only the generator is required. Mode collapse happens when many noise inputs produce similar outputs, reducing diversity.

### Q42. Which is more powerful: GANs or VAEs?

**Exam answer:** If image sharpness and perceptual realism are the main criteria, GANs have traditionally had an advantage over basic VAEs because adversarial training encourages realistic samples. VAEs offer a structured probabilistic latent space and more straightforward training, so they may be preferable for representation learning and latent-space inference; neither is universally more powerful.

**Explanation:** Basic pixel-reconstruction objectives can average plausible outputs, yielding blur. GANs can generate sharp samples but may collapse modes or be unstable. Modern VAE variants can substantially improve image quality. State your criterion rather than claiming an absolute ranking.

### Q43. Vanishing and exploding gradients

**Exam answer:** Vanishing gradients occur when repeated chain-rule multiplications make gradients extremely small, preventing earlier layers from learning effectively. Exploding gradients occur when these multiplications produce very large gradients, causing unstable updates or numerical overflow.

**Explanation:** In a scalar illustration, multiplying by 0.5 through 20 layers gives about 9.54 × 10⁻⁷; multiplying by 2 gives 1,048,576. Real networks involve products of Jacobian matrices, not just scalar derivatives. Sigmoid saturation can exacerbate vanishing gradients. Suitable initialization, residual connections, normalization and non-saturating activations help; gradient clipping is useful for exploding gradients.

### Q44. Why scale the dot product in a vision transformer?

**Exam answer:** The magnitude of query-key dot products tends to grow with the key dimension, which can push softmax into saturation and produce very small gradients. Dividing by √d_k keeps attention logits at a more stable scale and improves optimization.

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Explanation:** If independent query and key components have zero mean and unit variance, the sum of d_k products has variance d_k. Dividing by √d_k gives variance approximately 1. For d_k=64, the scale divisor is 8. The divisor is based on the dimension of each attention head's keys, not the number of patches.

### Q45. Advantages of ViT over a CNN

**Exam answer:** A standard Vision Transformer uses self-attention to model relationships between distant image patches directly and to adapt feature interactions to the input. ViTs also tend to scale well with large datasets and pretraining, and their token-based structure is convenient for multimodal learning.

**Explanation:** A conventional convolution starts with local spatial neighborhoods and expands receptive field through depth; global self-attention can connect any two tokens in one layer. However, global attention costs O(N²) in token interactions, and ViTs often need substantial pretraining or suitable inductive biases. CNNs can be more efficient and data-efficient in small-data or high-resolution settings. Positional information is needed to encode spatial arrangement in standard ViTs.

### Q46. What is a Swin Transformer and how does it differ from a CNN?

**Exam answer:** A Swin Transformer is a hierarchical vision transformer that computes self-attention within local windows and shifts the windows in alternating blocks to allow cross-window communication. Unlike a CNN's fixed learned convolution kernels, Swin uses input-dependent attention weights, while still building multiscale spatial feature maps through patch merging.

**Explanation:** Split an image into patch tokens, apply window multi-head self-attention, then shifted-window attention in the following block. Patch merging reduces spatial resolution and typically increases channel dimension between stages. With a fixed window size, attention complexity grows approximately linearly with the number of tokens rather than quadratically as in global attention. This makes Swin useful for dense prediction, including detection and segmentation. It shares CNN-like locality and hierarchy but uses attention for feature interaction.

---

## Part III — Quick revision sheet

| Concept | Essential fact |
|---|---|
| Convolution output | floor((n + 2p − dilation × (k−1) − 1)/s) + 1 |
| Standard convolution parameters | C_out × (C_in × kH × kW + 1), with bias |
| Pooling parameters | Zero |
| Half squared L2 gradient | Prediction minus target |
| Max pooling backward | Gradient goes to the selected maximum |
| ReLU backward | Gradient passes for positive preactivation |
| Sigmoid derivative | σ(z)(1−σ(z)) |
| Softmax | Exponentiate shifted logits and normalize |
| One-hot cross-entropy | −ln(probability of true class) |
| FC → equivalent conv | Same parameters; supports spatial score maps |
| ResNet vs DenseNet | Addition vs concatenation |
| YOLO vs Faster R-CNN | One-stage vs two-stage detection |
| Autoencoder vs VAE | Reconstruction vs probabilistic latent model |
| GAN vs diffusion | Adversarial objective vs denoising objective |
| Flow matching | Learn a time-conditioned velocity field |
| ViT vs Swin | Global attention vs local shifted-window hierarchy |

**Common mistakes:** Confusing channels with spatial dimensions; forgetting bias parameters; applying softmax twice; routing max-pooling gradients to all four elements; forgetting that shared filter gradients add across spatial positions; treating transposed convolution as an inverse; assuming more parameters always means better performance; and claiming that one generative model is universally superior.

**Source:** Supplied practice examination, pages 1–4. The question numbering and topics follow that source; calculations and conceptual explanations are independently worked out. The referenced Week 5 notebook and lecture materials were not supplied, so course-specific conventions beyond the question sheet are not assumed.
