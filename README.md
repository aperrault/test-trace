# test-trace

Author: Andrew Perrault

If you use our code, please cite us:
Designing Efficient Contact Tracing Through Risk-Based Quarantining
Andrew Perrault, Marie Charpignon, Jonathan Gruber, Milind Tambe, Maimuna S. Majumder

Dependencies:
Python 3
numpy
matplotlib
scipy
numba

Reproducing figures from the paper:

Table 4: python -i runs.py
run_suite(num_index_cases=100000, trace_delay=1, cases_contacted=0.8, trace_false_negative=0.2, base_reduction=0.5, test_delay=1, dropout=0.05)

Figure 3: python -i runs.py
run_suite_delay(num_index_cases=100000, trace_delay=1, cases_contacted=0.8, trace_false_negative=0.2, base_reduction=0.5)
