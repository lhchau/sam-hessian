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