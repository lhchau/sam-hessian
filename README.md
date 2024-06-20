# Sharpness-Aware Minimization Hessian

## Analysis
The original SAM update:
$$
w_{t+1} = w_t - \eta \nabla L(w_t + \rho \frac{ \nabla L(w_t) }{ || \nabla L(w_t) || })
$$
This formula aims to minimize the following loss:
$$
L(w_t + \epsilon)
$$
We use third order Taylor approximation for the above loss:
$$
\begin{align*}
  L(w_t + \epsilon) &= L(w_t) + \epsilon \nabla L(w_t) + \frac{\epsilon \nabla^2L(w_t) \epsilon}{2} \\
  &= L(w_t) + \epsilon \nabla L(w_t) + \frac{Tr(\nabla^2(w_t))}{2}
\end{align*}
$$
Take derivative both side:
$$
\begin{align*}
  \nabla L(w_t + \epsilon) &= \nabla L(w_t) + \nabla[ \epsilon \nabla L(w_t) ] + \frac{\nabla [ Tr(\nabla^2(w_t)) ]}{2}
\end{align*}
$$
Assume $\epsilon = \rho \frac{ \nabla L(w_t) }{ || \nabla L(w_t) ||}$
$$
\begin{align*}
  \nabla L(w_t + \epsilon) &= \nabla L(w_t) + \nabla[ \rho \frac{ \nabla L(w_t) }{ || \nabla L(w_t) ||} \nabla L(w_t) ] + \frac{\nabla [ Tr(\nabla^2(w_t)) ]}{2} \\
  &=  \nabla L(w_t) + \nabla[ \rho ||\nabla L(w_t)|| ] + \frac{\nabla [ Tr(\nabla^2(w_t)) ]}{2} \\
  &= \nabla L(w_t) + \rho \frac{ \nabla L(w_t) }{ || \nabla L(w_t) || } + \frac{\nabla [ Tr(\nabla^2(w_t)) ]}{2}
\end{align*}
$$

Approximation of gradient of gradient norm:
$$
\begin{align*}
  &\nabla L(w_t + v) = \nabla L(w_t) + \nabla^2 L(w_t) v \\
  &\nabla^2 L(w_t) v = \nabla || \nabla L(w_t) || = \nabla L(w_t + v) - \nabla L(w_t) 
\end{align*}
$$

## Intuition from Update Rule
Considering the SAM update rule, we denote \( D \) as the gradient computed on the full batch, and \( B \) as the gradient computed on a mini-batch:

$$
w_{t+1} = w_t - \eta \nabla_{B} L(w_t + \rho \frac{\nabla_{B} L(w_t)}{||\nabla_{B} L(w_t)||})
$$

We focus on the gradient and use a first-order Taylor approximation. For convenience in analysis, and without loss of generality, we eliminate the gradient normalization in the denominator:

$$
\begin{align*}
\eta \nabla_{B} L(w_t + \rho \nabla_{B} L(w_t)) &= \eta [\nabla_{B} L(w_t) + \rho \nabla_{B}^2 L(w_t) \nabla_{B} L(w_t)] \\
&= \eta (I + \rho \nabla_{B}^2 L(w_t)) \nabla_{B} L(w_t) \\
&= \eta (I + \rho \nabla_{B}^2 L(w_t)) (\nabla_{D} L(w_t) + \epsilon)
\end{align*}
$$

The gradient magnitude is rescaled with weight $I + \rho \nabla_{B}^2 L(w_t)$. Comparing to IRE whose update rule as follow:
$$
\begin{align*}
w_{t+1} = w_t - \eta( \nabla_{B} L(w_t) + k P_t \nabla_{B} L(w_t))
\end{align*}
$$
where $k$ is control hyperparameter, $P_t$ is projection which retain the top flat coordinates. We can interpret this formula as **accelerating convergence in flat direction** and maintain the sharp direction as SGD.

IRE approach is quite different with intuition derived from SAM update where the acceleration happens on both sharp and flat direction, and even that the sharp direction is more accelerated than flat.

## Extended version of SGDHESS
From the analysis in the paper **How SGD find flat minima**, they proved that the noise variance in SGD aligns well with matrix Gauss-Newton $G$, which is an linearized approximation of Hessian $H$.

We propose the variant of SGD as follow:
$$
\begin{align*}
  &m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \\
  &\hat{\epsilon_t} = g_t - m_t \ (\text{Noise extraction}) \\
  &\epsilon_t = \beta_1 \epsilon_{t-1} + (1-\beta_1) \hat{\epsilon_t} \\
  &v_t = \beta_2 \epsilon_{t-1} + (1-\beta_1) (\epsilon_t - \hat{\epsilon_t})^2 \\
  &w_{t+1} = w_t - \eta g_t(I + \rho * v_t)
\end{align*}
$$

## Research Question
**Why AdaHessian cannot find flat minima while the multiplicative version can?**

**AdaHessian**:
$$
\begin{align*}
&m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \\
&h_t = \beta_2 h_{t-1} + (1-\beta_2) h_t^2 \\
&w_{t+1} = w_t - \eta \frac{m_t}{\sqrt{h_t} + \epsilon}
\end{align*}
$$