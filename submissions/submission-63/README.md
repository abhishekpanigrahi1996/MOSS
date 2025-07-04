# **Optimizing Explanations: Nuances Matter When Evaluation Metrics Become Loss Functions**

*Authors: Jonas B. Raedler, Hiwot Belay Tadesse, Weiwei Pan, Finale Doshi-Velez*

---
This is the notebook that accompanies our paper, published at the *Methods and Opportunities at Small Scale (MOSS)* workshop, held at ICML 2025 (Vancouver, Canada).

**Please Read:**

In the interest of time, this notebook does not run our experiments on the entire considered data. Instead, we choose to only run them for dimensions 2 and 4, and only for the perturbation region $u=5$. We use all the functions that were considered in the paper. If interested, people can adjust variables to generate results for other data.

Additional Note: in the paper, we used the MOSEK solver, which requires a license (they provide free academic licenses: https://www.mosek.com/). We provide an alternative solver here (`solver="SCS"`) that is freely available through the `cvxpy` library, which leads to very similar results (we note, though, that some minor differences may occur).