# Quantum Error Forms

(A writeup of the details with proofs is in my 2025 PhD thesis.)

### Hermitian form $\lambda$
Let $\mathcal{H}_n = (\mathbb{C}^2)^{\otimes n}$ be the state space of $n$ qubits.  Let $M \le \mathcal{H}_n$ be a quantum code. 

Let $\mathcal{E} \le \mathrm{End}_\mathbb{C}(\mathcal{H}_n)$ be an error space, and suppose $M$ corrects $\mathcal{E}$.  

Then there exists a Hermitian form $\lambda : \mathcal{E} \times \mathcal{E} \to \mathbb{C}$ defined by 

$$
\lambda(E, F) = \braket{v|E^\dagger F|v}
$$

where $\ket{v}$ is any unit length vector in $M$. Once an ordered basis for $\mathcal{E}$ is chosen, we can write a Hermitian matrix $\widehat{\lambda}$ corresponding to $\lambda$. We are focusing initially on calculations related to the 9-qubit Shor code, which we call $M_\text{Shor}$, but where convenient we support generality for other computations.  Along these lines, we use the Pauli basis for $`\mathcal{E}_n := \mathrm{End}_\mathbb{C}(\mathcal{H}_n)`$, with the base-4 indexing scheme that orders it:

$$
\begin{align*}
0 &\mapsto I\\
1 &\mapsto X\\
2 &\mapsto Y\\
3 &\mapsto Z\\
s \in 0 \ldots (2^n)^2 - 1 &\mapsto [s]_4 \ \ \text{(base 4 string)} \mapsto W_0 \otimes \cdots \otimes W_{n-1} \ \ W_i \in \{I, X, Y, Z\}
\end{align*}
$$

e.g. for $n = 9$,

$$
\begin{align*}
0 & \mapsto 0 0 0 \ 0 0 0 \ 0 0 0  \mapsto I \otimes I \otimes I \otimes \ I \otimes I \otimes I \otimes I \otimes I \otimes I\\
1 & \mapsto 000 \ 000 \ 001 \mapsto X \otimes I \otimes I \otimes \ I \otimes I \otimes I \otimes I \otimes I \otimes I\\
5 & \mapsto 000 \ 000 \ 011 \mapsto X \otimes X \otimes I \otimes \ I \otimes I \otimes I \otimes I \otimes I \otimes I\\
9 & \mapsto 000 \ 000 \ 021 \mapsto Y \otimes X \otimes I \otimes \ I \otimes I \otimes I \otimes I \otimes I \otimes I\\
262143 & \mapsto 333 \ 333 \ 333 \mapsto Z \otimes Z \otimes Z \otimes \ Z \otimes Z \otimes Z \otimes Z \otimes Z \otimes Z.
\end{align*}
$$

As is typical in the literature, these strings are often abbreviated $XII \ III \ III$, for example, and we use the notation $X_j$ for the string with I in every position except $j$ where there is an $X$. Call this ordered basis $\mathcal{P}_n$.  For the subspace of errors affecting at most $t$ qubits nontrivially, which we call $\mathcal{E}_t$, we use a subset of $`\mathcal{P}_n$ called $\mathcal{P}_{n,t}`$, with the ordering inherited from $`\mathcal{P}_n`$. Continuing the example,

$$
\mathcal{P}_{9,1} = \{ I^{\otimes 9}, X_1, Y_1, Z_1, X_2, Y_2, Z_2, \ldots, X_9, Y_9, Z_9\},
$$

and $`\dim \mathcal{E}_1 = |\mathcal{P_{9,1}}| = 28`$.

The first task is to compute the matrix $\widehat{\lambda}$ for Hermitian form $\lambda$ induced by $M_\text{Shor}$, in the $\mathcal{P}_{9,1}$ basis.  This is initiated by the command `make lambdaHat` which eventually writes file `lambdaHat.txt`.

```
mkdir worksets/000-demo
cd worksets/000-demo
make -f ../Makefile lambdaHat
...
make -f ../Makefile rank-lambdaHat
...
[2025-11-08 21:20:52] ======================================================================
[2025-11-08 21:20:52] RESULT: rank = 22
[2025-11-08 21:20:52] ======================================================================
```

### Error forms $B_{\lambda, E, F}$

The notation $`(E, F)^*B`$ means the _mixed pullback_ of $B$ by $E$ and $F$, defined by $`(E, F)^*B(u, v) = B(Eu, Fv)`$. Define

$$
    B_{\lambda, E, F} = (E, F)^*B - \lambda(E, F) \cdot B.
$$

```
cd worksets/000-demo
make -f ../Makefile Bhat-on-M
...
2025-10-31 23:53:54] ======================================================================
[2025-10-31 23:53:54] All pairs processed!
[2025-10-31 23:53:54]
[2025-10-31 23:53:54] Total pairs checked: 784
[2025-10-31 23:53:54] Violations found: 0
...
make -f ../Makefile Bhat-hermitian
...
[2025-11-02 00:12:49] ======================================================================
[2025-11-02 00:12:49] All pairs processed!
[2025-11-02 00:12:49]
[2025-11-02 00:12:49] Total pairs checked: 784
[2025-11-02 00:12:49] Hermitian matrices: 730
[2025-11-02 00:12:49] Non-Hermitian matrices: 54
...
```



### Total error form $D_{\lambda, \mathcal{E}}$

The _total error form_ $D_{\mathcal{E}, \lambda}$ is 

```math
D_{\mathcal{E}, \lambda} = \sum_{E, F \in \mathcal{P}_{9,1}} B_{\lambda, E, F}.
```

Abbreviating $D := D_{\mathcal{E}, \lambda}$, we compute the matrix $\widehat{D}$ corresponding to $D$ with respect to the computational basis of $\mathcal{H}_n$.

```
cd worksets/000-demo
make -f ../Makefile Dhat
...
[2025-11-08 23:55:43] Writing D̂ to output/Dhat.txt...
[2025-11-08 23:55:43] (Using sparse format: only non-zero entries)
[2025-11-08 23:55:43]
[2025-11-08 23:55:44] Matrix written successfully!
[2025-11-08 23:55:44] Non-zero entries: 23552 / 262144
[2025-11-08 23:55:44] Sparsity: 91.02%
...

make -f ../Makefile signature-Dhat
========================================
Computing signature of D̂ (numerical)...
========================================
...
[2025-11-09 06:37:46] ======================================================================
[2025-11-09 06:37:46] RESULTS:
[2025-11-09 06:37:46]   Signature: (92, 420, 0)
[2025-11-09 06:37:46]   p = 92 (positive eigenvalues)
[2025-11-09 06:37:46]   q = 420 (negative eigenvalues)
[2025-11-09 06:37:46]   r = 0 (zero eigenvalues / nullity)
[2025-11-09 06:37:46]   Rank = p + q = 512
[2025-11-09 06:37:46] ======================================================================
```


**Isotropic Extension Algorithm**

Assume $M$ corrects $\mathcal{E}$ and we have bases $M = \{v_i\}$ and $B = \{E_i\}$. 

(1) Start with $u = v_1$, which is isotropic with respect to total error form $D = D_{\mathcal{E}, \lambda}$. FollowingA \[Grove Prop. 10.8\] there exists $w \in \mathcal{H}_n$ with $D(u, w) = 1$. (Since $D$ is nondegenerate.)

 Solve the linear equation $$u^\dagger \hat{D} w = 1$$ for $w$.  Use `SymPy linsolve()`

(2) Then $D(w, w) \in \mathbb{C}$ is a number (actually it's real).  Take $b = -\frac{1}{2} D(w,w)$, and let $v = bu + w$. Then $(u, v)$ is a hyperbolic pair.  We already have $D(u,u) = 0$.  And
$$
\begin{align*} D(v, v) &= D(bu + w, bu+w) = D(bu, bu) + D(bu, w) + D(w, bu) + D(w, w) \\
&= b \bar{b}D(u,u) + \bar{b}D(u,w) + b D(w,u) + D(w,w)\\
&= 0 + \bar{b} + b + D(w,w)\\
&= -\tfrac{1}{2} D(w,w) + -\tfrac{1}{2} D(w,w) + D(w,w) \\
&= 0,
\end{align*} 
$$
so $v$ is isotropic.  And
$$
D(u, v) = D(u, bu + w) = D(u, bu) + D(u, w) = bD(u, u) + 1 = 1.
$$

(3) For each $i,j$, check whether $u+v$ is isotropic with respect to error form $B_{\lambda, E_i, E_j}$. 
If so, let $N = M \oplus \langle u+v \rangle$.
