Dear Editors,

Please consider our letter, “Friction-Budgeted Stabilization of RNTI-Level Containment in O-RAN,” for the call on Zero Trust Architecture and Adaptive Access Control for Next-Generation Networks.

The paper addresses a narrow operational gap between risk scoring and adaptive enforcement. It formulates a three-action RNTI-level controller—ALLOW, RESTRICT, or ISOLATE—under an explicit benign-friction budget, security and delay safeguards, and a minimum stability gain. The novelty is the constrained formulation, matched-budget protocol, and resulting security–stability frontier, not EWMA or hysteresis. Evaluation uses a public OpenIreland trace, chronological label-blind trace-block holdout, one-epoch actuation lag, unseen numeric RNTIs, per-attack outcomes, and paired block bootstrap.

The main held-out result is a 57.4% reduction in action transitions and a 4.48 percentage-point reduction in malicious ALLOW time relative to a budget-matched stateless rule, with unchanged median capped delay. The paper also reports the counterevidence needed to interpret this result: hard-isolation exposure worsens, Slowloris coverage and delay worsen materially, and the held-out friction interval crosses the nominal budget. Under gradient-boosted scores the framework selects EWMA and passes all four controller gates, but held-out friction exceeds budget; a separate asymmetric-family diagnostic fails its coverage gate. Feasibility also depends on the empirically inferred timestamp scale.

Accordingly, the manuscript does not claim durable identity, authentication efficacy, causal attack prevention, or deployed QoS benefit. Its contribution is an auditable method and bounded offline result for stabilizing policy actions at the current RNTI attachment.

Sincerely,

[Corresponding author]
