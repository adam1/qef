# Quantum Error Forms

(A writeup of the details with proofs is in my 2025 PhD thesis.)

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

The first task is to compute the matrix $\widehat{\lambda}$ for Hermitian form $\lambda$ induced by $M_\text{Shor}$, in the $\mathcal{P}_{9,1}$ basis.  This is initiated by the command `make lambdaHat` which eventually writes file `lambdaHat.h5`.

