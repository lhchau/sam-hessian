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
    - **Slow down the noisy fitting**: Training accuracy increases rapidly, whereas validation accuracy increases slowly.
    - **Learn usefull pattern**: Training accuracy increases slowly, whereas validation accuracy increases quickly.

### SAM Efficiency - SAME

- We implemented the new optimization as follow:

$$
\begin{align*}
  g_{t+0.5} &= \nabla L(w_t + \rho \frac{ \nabla L(w_t) }{ || \nabla L(w_t) || }) \\
  w_{t+1} &= w_t - \eta (g_{t+0.5} + g_{t+0.5} * mask(ckpt))
\end{align*}
$$