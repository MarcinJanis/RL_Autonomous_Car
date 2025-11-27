

1) Udate policy

$$
L = \mathbb{E}\Big[
    \min\big(
        r_t(\theta) A_t,\;
        \operatorname{clip}(r_t(\theta),\, 1 - \epsilon,\, 1 + \epsilon)\, A_t
    \big)
\Big]
$$
