---
marp: true
author: Jairo Sousa
size: 16:9
style: |
  :root {
    font-size: 20px;
    margin: 0px;
    padding: 35px;
  }
math: mathjax
---

# Modelo

- Modelo $\phi^4$ com condição de fronteira dirichlet 
$$
\phi(x=0)=H.\tag{1}
$$

- As soluções BPS que interpola a condição de fronteira é
$$
\varphi ( x) =\mu (\tanh( \chi -x))^{\nu }\tag{2}
$$
para
$$
\chi =\mu \tanh^{-1}\left( H^{\nu }\right) 
\quad \text{e} \quad
\nu =\text{sgn}( 1-|H|) \tag{3}
$$

<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>
![width:800px center](solutions.svg)

---

# Overview

![width:1000px center](initial_plots.svg)

---

# Perturbações

- Considerando pequenas perturbações em torno da solução na fronteira
$$
\phi =\varphi +e^{i\omega t} \psi,\tag{4}
$$
substituindo na equação de movimento 
$$
\partial _{t}^{2} \phi -\partial _{x}^{2} \phi +V'( \phi ) =0\tag{5}
$$
e linearizando, obtemos um problema de autovalor
$$
\left[ -\partial _{x}^{2} +U( x)\right] \psi ( x) =\omega ^{2} \psi ( x),\tag{6}
$$
para
$$
U(x)= V''(\varphi) =6\varphi ^{2}(x) -2\tag{7}
$$
na semilinha negativa $x \leq 0$.

---

- Analisando apenas as perturbações para as soluções que interpolam $|H|\leq 1$,
$$
U( x) =6\tanh^{2}\left(\tanh^{-1} H-x\right) -2\tag{8}
$$

![width:900px center](schro-potential.svg)

---

## Solução numérica

- Domínio discreto com $N_p=2048$ pontos igualmente espaçados por $h=L/(N_p-1)$, onde $L=20$
$$
x_k = kh - L,\quad
0 \leq k \leq N_p-1\tag{9}
$$
$$
U_k = U(x_k),\quad \psi_k = \psi(x_k)\tag{10}
$$

- Utilizando o método de diferenças finitas de segunda ordem,
$$
D^{2} \psi _{k} =\frac{\psi _{k-1} -2\psi _{k} +\psi _{k+1}}{h^{2}}\tag{11}
$$
podemos aproximar o operador diferencial em forma de uma matriz $(N-1)\times(N-1)$
$$
-D^{2} +U_{i} =-\frac{1}{h^{2}}\begin{pmatrix}
-2 & 1 &  & 0\\
1 & -2 & 1 & \\
 & \ddots  & \ddots  & \ddots \\
0 &  & 1 & -2
\end{pmatrix} +\begin{pmatrix}
U_{1} &  & 0 & 0\\
 & U_{2} &  & 0\\
0 &  & \ddots  & \\
0 & 0 &  & U_{N_{p} -2}
\end{pmatrix}\tag{12}
$$
e resolver o problema de autovalor na aproximação matricial.

---

## Densidade espectral

- Sendo $\tilde{\omega}_0$ a menor frequência encontrada na análise numérica do problema de autovalor

![width:570px](eigenvalues.svg) ![width:530px](spectrum-map-v=0.25.svg)

---

# Ocsilon-fronteira ($H=-0.5$, $v\approx 0.13$)

![height:230px center](static-oscilon_1.svg) 
![width:600px](static-oscilon-decay_1.svg) ![width:600px](static-oscilon-fft_1.svg)