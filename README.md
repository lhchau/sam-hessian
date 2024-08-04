# Sharpness-Aware Minimization 

## 2024/07/04

- In the paper "Why SAM is robust to label noise", the authors mentioned that SAM's ability to find flat minima was reweighting the gradient, up-weighting gradients corresponding to low loss points rather high loss points. We can intuitively interpret them as *"Since the loss decreases faster for clean points over noisy points, SAM preferentially up-weights the gradients of clean points."* or *"Since the loss decreases faster for useful patterns over noisy details, SAM preferentially up-weights the gradients of useful patterns."*

- Previous research, we decompose SAM's gradients into four checkpoints based on the ratio of perturbed gradient over original gradient $ratio = \frac{ \nabla L(w_t + \rho g_t / ||g_t||)}{g_t}$:
  - **ckpt1**: $ratio > 1$.
  - **ckpt2**: $0 < ratio < 1$.
  - **ckpt3**: $ratio < -1$.
  - **ckpt4**: $-1 < ratio < 0$.

- In this research, we aim to explore which checkpoint is responsible for learning useful patterns and which checkpoint slows down noisy fitting. 
  - The technique, we used to verify this study, involved increasing the gradient of each checkpoint subsequently and comparing those with the original SAM.
  - Our hypotheses:
    - **noisy fitting**: Training accuracy increases rapidly, whereas validation accuracy increases slowly.
    - **Learn usefull pattern**: Training accuracy increases slowly, whereas validation accuracy increases quickly.

### SAM Efficiency - SAME

- We implemented the new optimization as follow:

$$
\begin{align*}
  g_{t+0.5} &= \nabla L(w_t + \rho \frac{ \nabla L(w_t) }{ || \nabla L(w_t) || }) \\
  w_{t+1} &= w_t - \eta (g_{t+0.5} + g_{t+0.5} * mask(ckpt))
\end{align*}
$$

- The results showed that:
  - SAMECHECKPOINT1: 
    - In the early stage, training accuracy is **less** than the original SAM, validation accuracy is quite **similar** to original SAM -> **Learn useful pattern**
    - In the later stage, training accuracy is **greater** than the original SAM,  validation accuracy is **greater** than original SAM -> **Learn useful pattern**
    - In addition, higher condition number is, higher flatter, higher training accuracy and slower gradient norm are. In contrast, the increasing perturbation radius also leads to the declining of gradient norm but the training accuracy is very low.
  - SAMECHECKPOINT2:
    - In the early state, training accuracy is **greater** than the original SAM, validation accuracy is quite **similar** to original SAM -> **Noisy learning**
    - In the later stage, training accuracy is **greater** than the original SAM,  validation accuracy is **greater** than original SAM -> **Learn useful pattern**
  - SAMCHECKPOINT3:
    - In the early state, training accuracy is **less** than the original SAM, validation accuracy is quite **similar** to original SAM -> **Learn useful pattern**
    - In the later stage, training accuracy is **less** than the original SAM,  validation accuracy is **less** than original SAM -> **Underfitting**
  - SAMCHECKPOINT4:
    - In the early state, training accuracy is **less** than the original SAM, validation accuracy is quite **similar** to original SAM -> **Learn useful pattern**
    - In the later stage, training accuracy is **less** than the original SAM,  validation accuracy is **higher** than original SAM -> **Learning useful pattern**

### CustomSAME

- From our findings, we create a new optimizer, utilizing the advantage of each checkpoint
  - Increase grad checkpoint1
  - Maintain grad checkpoint2
  - Reduce grad checkpoint3
  - Increase grad checkpoint4

- Early stage:
  - Paras corresponding to learn useful pattern, have large gradients, those weights will be large, those variances will be small.
  - Paras corresponding to learn noisy, have small gradients, those weights will be large, those variances will be large.
- Later stage:
  - Paras corresponding to learn useful pattern, have small gradients, those weights will be small, those variances will be small.
  - Paras corresponding to learn noisy, have large gradients, those weights will be large, those variances will be large.


### God sign

1. The generalization effect of SAM diminished when more data?
2. The gradient norm of perturbed gradient over current gradient at the later stage is more greater than the earlier stage. 
3. Checkpoint3, checkpoint4 increase during later stage while checkpoint1 decrease. In contrast, ASAM do experience the increase of checkpoint1.

## 2024/07/12

- [x] Run SAMMAGNITUDE and SAMDIRECTION on different architectures:
  - [x] Resnet18
  - [x] Resnet34
  - [x] Resnet50
  - [x] Wrn28-10 
  - Chu y luc report thi doi ten SAMMAGNITUDE <-> SAMDIRECTION

1. Magnitude cua SAM quyet dinh toi accuracy. Insights:
- Chi su dung cac dao ham cua SAM ma cung dau voi SGD => Test acc cao nhung test loss cao (Nguoc lai SAM thi dam bao duoc test acc cao va test loss thap).
- Chi su dung cac dao ham cua SGD ma cung dau voi SAM => Test acc cao hon so voi SGD binh thuong nhung van ko cao bang thi nghiem tren.
- Magnitude cua SAM nhung Direction cua SGD => Test acc cao
- Direction cua SAM nhung Magnitude cua SGD => Ko cao
2. Direction cua SAM quyet dinh toi test loss !!!

Do tac dong tai sao SAM lai co direction nguoc huong voi SGD: 
- Thi nghiem SAMANANATOMY, bo het ratio < 0, tai buoc di len thu 2, nhung ratio < 0, van duy tri tai buoc 3, tham chi con cao hon. 

### Project1: Hypothesis
- SAM ko train duoc voi rho cao vi co mot so grad outlier khien cho buoc di len qua xa tai mot so parameters => Ko on dinh.
- Project1: Lam sao kiem tra duoc gia thuyet nay co that hay ko?
- CLAMPSAM 

### Project2: Ve Lien he giua Direction va Magnitude
- perturbation radius cao => ratio < 0 tang
- cap nhat parameters chi dua tren cac ratio > 0
- Ideal: 
  - Vi khi rho cao qua thi keo theo mot so direction thay doi nhieu => Giam tac dong cua 1 so parameters lai => Khien nhieu ratio > 0 hon
  - Dieu nay se giup giao thoa giua Direction va Magnitude???


## 2024/08/04

**Problem Setup**

Model. We consider binary classification. The sign of the model’s output $f: R^n \to R$ maps inputs $x \in X$ to discrete labels $t \in \{-1, 1\}$. We will study two kinds of models – a linear model and a
2-layer deep linear network (DLN) in this work. We do not include the bias term.

- **Linear**: $f(w, x) = \langle w, x \rangle$
- **DLN**: $f(v, W, x) = \langle v, Wx \rangle$

**Lemma 3.1 (Preferential up-weighting of low loss points)** Consider the following function:
 
$$
f(z) = \frac{ \sigma(-z + C) }{ \sigma(-z) } = \frac{ 1 + exp(z) }{ 1 + exp(z-C)} = 1 + \frac{exp(z) - exp(z-C)}{1 + exp(z - C)}
$$

This function is stricly increasing if $C > 0$.

We can interpret 1-SAM as a gradient reweighting scheme, where the weight for example $x_i$ is set to 

$$
\frac{ || \nabla l(w + \epsilon_i, x_i, t_i) ||}{ || \nabla l(w, x_i, t_i) || } = \frac{ \sigma( -t_i \langle w, x_i \rangle + \rho || x_i ||)}{ \sigma( -t_i \langle w, x_i \rangle )}
$$

Choose $z = t_i \langle w, x_i \rangle$ and $C = \rho || x_i ||_2$

**Objective**: We consider the logistic loss $l(w, x, t) = -log(\sigma (t . f(w, x)))$ for sigmoid function $\sigma(z) = \frac{1}{1 + exp(-z)}$. Given $n$ training points $[(x_i, t_i)]^n_{i=1}$ sampled from the data distribution $D$, our training objectives is 

$$
\min_w L(w)
$$

where $L(w) = \frac{1}{n} \sum^n_{i=1} l(w, x_i, t_i)$

By chain rule, we can write the sample-wise gradient with respect to the logistic $l(w, x, t)$ as 

$$
-\nabla_w l(x, t) = t . \sigma(-t f(w, x)) \nabla f(w, x)
$$

SAM perturbation:

$$
\begin{align*}
-\nabla_w l(x, t) &= t . \sigma(-t f(w + \rho \frac{ - t . \sigma(-t f(w, x)) \nabla f(w, x) } { || t . \sigma(-t f(w, x)) \nabla f(w, x) || } , x)) \\ &\nabla f(w + \rho \frac{ - t . \sigma(-t f(w, x)) \nabla f(w, x) } { || t . \sigma(-t f(w, x)) \nabla f(w, x) || }, x) 
\end{align*}
$$

**Explicit reweighting does not fully explain SAM's gains**

We first note that the upweighting of low-loss points can be observed even in multi-class classification with neural networks. Recall the general form of the gradient for multi-class cross-entropy loss is 

$$
\nabla l(x, t) = \langle \sigma (f(w, x)) - e_t, \nabla f(w, x) \rangle 
$$

where $\sigma$ is the softmax function and $e_t$ is the one hot encoding of the label $t$.

A similar analysis of SAM under label noise in linear models was conducted by Andriushchenko, however they attribute the label noise robustness in neural networks to logit scaling. We claim the opposite: that the *direction* or the network Jacobian of SAM's update becomes much more important.

**Analysis**

Motivated by this, we study the effect of perturbing the Jacobian in a simple 2-layer DLN. In the linear case, the Jacobian term was constant, but in the non-linear case the Jacobian is also sensitive to perturbation. In particular, for 2-layer DLNs, J-SAM regularizes the norm of intermediate activations and last layer weights, 

**Proposition 4.1** For binary classification in a 2-layer deep linear network $f(v, W, x) = \langle v, Wx \rangle$, J-SAM approximately reduces to SGD with L2 norm penalty on the intermediate activations and last layer weights.

**Proof**: We write the form of the J-SAM update for the first layer $W$ of the deep linear network:

$$
\begin{align*}
  - \nabla_{W + \epsilon^1} l(w + \epsilon, x, t) &= \sigma (-t f(w,x))(tv - \frac{\rho}{J}z)x^T \\
  &= - \nabla_W l(w, x, t) - \frac{ \rho \sigma (-t f(w, x)) }{J} z x^T
\end{align*}
$$

where $z = Wx$ is the intermediate activation and $J = || \nabla f(x) || = \sqrt{ ||z||^2 + ||x||^2 ||v||^2}$ is a normalization factor. In the second layer, the gradient with respect to $v$ is

$$
\begin{align*}
  - \nabla_{v + \epsilon^2} l(w + \epsilon, x, t) &= \sigma (-t f(w,x))(tz - \frac{\rho ||x||^2 }{J}v) \\
  &= -\nabla_v l(w, x, t) - \frac{ \rho \sigma (-t f(w, x)) ||x||^2 }{J} v
\end{align*}
$$

From these equations, note that SAM adds an activation norm regularization to the first layer $zx^T = \nabla_W \frac{1}{2} ||z||^2_2$ scaled by some scalar dependent on $\rho$, $f(w, x)$, and $J$. Similarly, note that SAM adds a weight norm penalty to the second layer weights $v = \nabla_v \frac{1}{2} ||v||^2$ also multiplied by some scalar. The normalization factor J scales the regularization be closer to the norm than the squared norm.