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
  \nabla L(w_t + \epsilon) &= \nabla L(w_t) + \nabla[ \rho \frac{ \nabla L(w_t) }{ || \nabla L(w_t) ||} \nabla L(w_t) ] + \frac{\nabla [ Tr(\nabla^2(w_t)) ]}{2}
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