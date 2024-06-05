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
We choose $\epsilon = \rho \frac{ \text{diag}(H(w)) \nabla L(w_t) }{ || \text{diag}(H(w)) \nabla L(w_t) ||}$