

1) Udate policy

$L = E( min( r_t(\theta) A_t, clip( r_t( \theta ) A_t, 1 - \epsilon, 1 + \epsilon) A_t ) )$

where: 

$ r_t( \theta ) = \frac{\pi ( a_t | s_t)} {\pi_old( a_t | s_t)} $
$ a_t - action taken (propability) $
$ s_t - state actual $
