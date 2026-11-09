# Mathematical foundations and viva preparation

These examples explain algorithms, not measured AgriVision results. TensorFlow
handles differentiation and optimization; our code defines inputs, architecture,
training protocol and validation checks.

## 1. Digital images

\(I \in \{0,\ldots,255\}^{H\times W\times3}\). Here H is height, W width,
and the final index selects red/green/blue. An image is a grid of channel values.
Our shared pipeline converts to RGB and resizes to 224×224, supplying float32
values in [0,255]. Example: pixel (255,0,0) is red in RGB.

**Viva:** “An RGB image is a height-by-width-by-three tensor. The backbone includes
its own preprocessing, so we do not divide by 255 again.”

## 2. Convolution

\(Y_{i,j,o}=b_o+\sum_{u,v,c}K_{u,v,c,o}X_{i+u,j+v,c}\), shown for unit stride
and valid cross-correlation. X is input, K a learned kernel, b bias, o output
channel, and u,v,c index the local spatial patch/input channels. Deep-learning
“convolution” usually implements cross-correlation without flipping K.

Both backbones learn local patterns such as edges, texture and spot boundaries.
Example: summing a 2×2 patch [1,2;3,4] with an all-one kernel gives 10 before bias.
**Viva:** “Shared filters slide over the image, learning spatially local features
with fewer parameters than fully connecting every pixel.”

## 3. Feature extraction

\(F_\theta(I)=h_L(h_{L-1}(\ldots h_1(I))))\). Each h is a layer transformation
and theta denotes backbone parameters. Early stages tend to capture local
patterns; deeper stages combine them into more task-relevant representations.
Our frozen ImageNet backbone maps a leaf tensor to feature maps before pooling.
Example: edges combine into a texture representation, which the new head can use.
**Viva:** “We reuse pretrained visual features and learn a new disease-class head;
fine-tuning then adapts selected features to our dataset.”

## 4. ReLU and other activations

\(\operatorname{ReLU}(x)=\max(0,x)\). x is a scalar activation; the function
retains positive values. Example: ReLU(-2)=0 and ReLU(3)=3.
Grad-CAM uses ReLU to retain positively contributing regions. The application
backbones also use their designed activations, including swish/hard-swish; we
do not claim every activation in EfficientNet/MobileNetV3 is ReLU.
**Viva:** “Nonlinearities let stacked layers represent nonlinear patterns.
Grad-CAM's final ReLU keeps positive class contributions.”

## 5. Pooling

Global average pooling: \(g_k=(HW)^{-1}\sum_i\sum_j A_{i,j,k}\), where A is a
feature tensor and k identifies its channel. It turns each spatial feature map
into one number. Our head uses this before dropout/logits. Example: averaging
[1,2;3,4] gives 2.5. Local max pooling instead takes the maximum over a window.
**Viva:** “Global averaging reduces head parameters and summarizes the spatial
feature maps; it differs from flattening every location into separate features.”

## 6. Softmax

\(p_c=\exp(z_c-m)/\sum_j\exp(z_j-m)\), with m=max_j z_j for numerical stability.
z are logits; p_c is a normalized class score. For logits [0,0], scores are
[0.5,0.5]. Our float32 softmax produces probabilities for ranking the top three.
**Viva:** “Scores sum to one across known classes, but a high score is not proof
of correctness or a calibrated estimate on unfamiliar images.”

## 7. Cross-entropy

\(\mathcal L=-N^{-1}\sum_{n=1}^N\log p_{n,y_n}\). N is batch size, y_n the
integer true label and p_{n,y_n} its predicted score. Sparse cross-entropy avoids
explicit one-hot labels. Example: a correct-class score of 0.8 yields loss
-ln(0.8)≈0.223; 0.2 yields ≈1.609. Class weights multiply sample losses during
training only. **Viva:** “The loss penalizes assigning low probability to the
correct class and is differentiable for gradient-based learning.”

## 8. Backpropagation

\(\partial\mathcal L/\partial w=(\partial\mathcal L/\partial h)
(\partial h/\partial w)\), generalized by the chain rule through all layers.
w is a trainable weight and h an intermediate value. GradientTape and Keras
compute gradients; frozen weights are excluded from updates. For h=wx,
L=(h-y)^2, w=1,x=2,y=0, dL/dw=8.
**Viva:** “Backpropagation efficiently computes how each trainable weight changes
the loss. The optimizer uses those gradients; the two are different operations.”

## 9. Adam

\(m_t=\beta_1m_{t-1}+(1-\beta_1)g_t\),
\(v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2\),
\(\hat m_t=m_t/(1-\beta_1^t),\quad\hat v_t=v_t/(1-\beta_2^t)\),
\(\theta_t=\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)\).
g_t is gradient, m/v moving moments, beta decay rates, eta learning rate and epsilon
a numerical stabilizer. Adam uses these to update the trainable head/backbone tail.
For a first scalar gradient 2 and zero moments, bias-corrected moments are 2 and
4, so the first step is approximately eta in the negative direction.
**Viva:** “Adam combines momentum and adaptive scaling. We use a smaller learning
rate when adapting pretrained weights to reduce destructive changes.”

## 10. Transfer learning

Stage 1: \(\min_\phi\mathcal L(h_\phi(F_{\theta_0}(I)),y)\).
Stage 2: \(\min_{\phi,\theta_S}\mathcal L\), where theta_0 is ImageNet backbone,
phi the new head and S the selected trainable tail. Example: replace the ImageNet
head with C dataset classes, train it, then adapt the last 30 layers while keeping
BatchNorm fixed. C comes from actual dataset folders, never a guessed constant.
**Viva:** “ImageNet supplies an initialization, not a ready-made disease model.
Our final predictions require our own completed plant-disease training.”

## 11. Augmentation

\(\min_\theta\mathbb E_{(I,y),T}[\mathcal L(f_\theta(T(I)),y)]\).
T is a random approximately label-preserving transformation. Training samples
vary while labels remain fixed. Example: a horizontal flip changes orientation
but generally preserves a leaf disease label. Strong color changes could destroy
symptom meaning. **Viva:** “We use mild transformations only on training data.
Augmented copies must not leak into validation/test; more transforms do not create
independent biological samples.”

## 12. Precision

\(P=TP/(TP+FP)\). For a chosen class, TP is correctly predicted members and FP
other classes wrongly assigned to it. Example: TP=8,FP=2 gives precision 0.8.
Macro precision averages class precision equally. The evaluation fixes the full
class order and sets undefined divisions to zero.
**Viva:** “Precision answers: among predictions of this class, how many were right?”

## 13. Recall

\(R=TP/(TP+FN)\), where FN is actual class members missed by the classifier.
Example: TP=8,FN=4 gives recall 2/3. Macro recall treats rare and common classes
equally. **Viva:** “Recall answers: among actual examples of this class, how many
did we find? Low recall can reveal an overlooked disease class.”

## 14. F1 and accuracy

\(F1=2PR/(P+R)\). With P=.8 and R=2/3, F1≈.727.
\(F1_{macro}=C^{-1}\sum_cF1_c\);
\(F1_{weighted}=N^{-1}\sum_cn_cF1_c\), where n_c is class support and C the class
count. Accuracy is \(N^{-1}\sum_n\mathbf1[\hat y_n=y_n]\).
**Viva:** “Macro F1 gives each class equal importance; weighted F1 emphasizes
frequent classes. Overall accuracy can hide poor minority-class performance.”

## 15. Grad-CAM

\(\alpha_k^c=Z^{-1}\sum_i\sum_j\partial z_c/\partial A_{i,j}^k\),
\(L_{GradCAM}^c=\operatorname{ReLU}(\sum_k\alpha_k^cA^k)\).
A^k is a convolutional feature map, z_c the selected pre-softmax score and Z the
number of spatial locations. Gradients estimate how changes in features influence
that score; their spatial means weight the maps. Example: if alpha=[2,-1] and
feature values at a pixel are [3,4], the positive map value is ReLU(6-4)=2.
Our implementation handles each nested backbone and discovers its own target
convolution layer. **Viva:** “Grad-CAM visualizes coarse regions contributing
positively to a class score. It does not prove the diagnosis or causality.”

## Additional likely viva questions

**Why EfficientNetB0?** It is a practical pretrained CNN baseline from a family
that jointly scales depth, width and resolution. We use B0 rather than training
the whole family. Its performance on our crops must be measured.

**Why MobileNetV3Large?** It uses mobile-oriented building blocks such as
depthwise operations, inverted residual designs, squeeze-and-excitation and
hardware-aware design. It is our efficiency comparison; being designed for
mobile use does not guarantee lower measured latency on every GPU/runtime.

**How is the comparison fair?** Same frozen split, preprocessing contract,
training protocol and metrics; report hardware/policy and parameter counts.
Tail-layer counts do not equate compute budgets, so disclose that limitation.

**Why freeze BatchNorm?** Updating its running statistics on a small new dataset
can destabilize pretrained behavior. We keep it in inference mode while allowing
selected non-BatchNorm layers to receive gradients during fine-tuning.

**How do you prevent leakage?** Remove exact decoded copies and allocate near-match
connected groups to one split. Review hash matches. We cannot guarantee plant/site
separation or catch every near duplicate without richer metadata and manual review.

**What is confidence-based uncertainty?** A heuristic warning when the largest
softmax score is below a threshold. It is not Bayesian uncertainty, calibrated
confidence or a reliable out-of-distribution detector.

**Can it diagnose any plant?** No. It chooses among the dataset's supported labels,
and may confidently misclassify unfamiliar crops, non-leaves or field imagery.

**What demonstrates generalization?** Held-out internal performance estimates
performance on that dataset distribution. Separate compatible-class external field
evaluation would be stronger evidence, but has not been performed here.

**Why float32 output with mixed precision?** Faster lower-precision intermediate
computation on compatible GPUs can coexist with numerically stable float32 logits
and softmax. Actual GPU performance still requires execution evidence.

**What happens after a Colab disconnect?** Drive retains the last committed
full-model checkpoint, optimizer and epoch/stage metadata. The current partial
epoch repeats; callback patience and augmentation order restart.

**Where do disease facts come from?** Static reviewed university-extension
records in disease_info.json. The model does not generate treatment advice.

**What is your accuracy?** “I can report it only after real evaluation. The
application reads the actual saved metrics, or says evaluation is unavailable.”
