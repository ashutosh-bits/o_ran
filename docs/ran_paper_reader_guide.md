# Beginner's Guide to the RNTI-Containment O-RAN Paper

This guide is a self-contained path from basic cellular networking to the exact theory, experiment, and claims in **“Friction-Budgeted Stabilization of RNTI-Level Containment in O-RAN.”** It is written for a reader who does not yet know radio access networks (RANs).

The paper itself is only four pages, so it necessarily compresses many ideas. This guide expands those ideas, explains the notation, connects every major claim to the implementation, and marks the line between what the experiment demonstrates and what it does not.

## Contents

1. [The paper in one minute](#1-the-paper-in-one-minute)
2. [How to use this guide](#2-how-to-use-this-guide)
3. [Cellular networking from first principles](#3-cellular-networking-from-first-principles)
4. [The RAN protocol stack](#4-the-ran-protocol-stack)
5. [From conventional RAN to O-RAN](#5-from-conventional-ran-to-o-ran)
6. [RNTI: the central enforcement handle](#6-rnti-the-central-enforcement-handle)
7. [KPI glossary for this dataset](#7-kpi-glossary-for-this-dataset)
8. [Dataset, traffic, and threat model](#8-dataset-traffic-and-threat-model)
9. [Zero trust and the paper's exact mapping](#9-zero-trust-and-the-papers-exact-mapping)
10. [From raw rows to causal decision epochs](#10-from-raw-rows-to-causal-decision-epochs)
11. [Risk-scoring theory](#11-risk-scoring-theory)
12. [Score-to-action controller theory](#12-score-to-action-controller-theory)
13. [Operational metrics](#13-operational-metrics)
14. [Friction-budgeted policy selection](#14-friction-budgeted-policy-selection)
15. [Leakage-resistant evaluation protocol](#15-leakage-resistant-evaluation-protocol)
16. [Statistical inference](#16-statistical-inference)
17. [Reading the frozen results](#17-reading-the-frozen-results)
18. [What the paper proves—and does not prove](#18-what-the-paper-provesand-does-not-prove)
19. [Code and artifact walkthrough](#19-code-and-artifact-walkthrough)
20. [From an offline intent to a real O-RAN implementation](#20-from-an-offline-intent-to-a-real-o-ran-implementation)
21. [Reviewer and presentation questions](#21-reviewer-and-presentation-questions)
22. [Common misconceptions](#22-common-misconceptions)
23. [Glossary](#23-glossary)
24. [Primary reading list](#24-primary-reading-list)
25. [Final readiness checklist](#25-final-readiness-checklist)

## 1. The paper in one minute

A connected mobile radio context can be addressed over the radio interface using a temporary identifier from the **RNTI** family. This dataset exports a numeric `mac_rnti`, although its exact RNTI subtype and authoritative lifecycle are not documented. The RAN telemetry contains radio- and MAC-layer measurements such as signal quality, selected modulation/coding, throughput, buffers, and successful or unsuccessful counters.

The study converts those measurements into a risk score once per observed RNTI-second. A policy then maps the score history to one of three network-level intents:

- `ALLOW`: preserve normal service.
- `RESTRICT`: request reduced service or a constrained/quarantine treatment for the observed lease.
- `ISOLATE`: request the strongest containment treatment for that observed lease.

A direct score threshold reacts quickly, but noisy scores can make the action switch repeatedly. A stable controller can reduce switching, but excessive persistence can admit attacks longer or fail to cover short attacks. The paper therefore does **not** ask merely, “Which classifier has the highest accuracy?” It asks:

> Among causal score-to-action policies that satisfy a declared benign-friction budget and security/delay safeguards, which policy minimizes action churn?

On the frozen chronological test replay, the selected policy reduced action transitions by **57.4%** and malicious `ALLOW` time by **4.48 percentage points** relative to a tuning-budget-matched stateless threshold. It did not improve everything: malicious time short of full `ISOLATE` increased, mean delay worsened, and Slowloris was a clear failure mode.

That qualified trade-off—not EWMA or hysteresis by itself—is the paper's contribution.

## 2. How to use this guide

There are two reasonable study tracks.

### Fast track: about 10 focused hours

1. Read Sections 3–7 for cellular, RAN, O-RAN, RNTI, and KPI foundations.
2. Read Sections 8–13 for the dataset, model, controller, metrics, and experimental protocol.
3. Read Sections 14–18 alongside the manuscript's Results and Discussion.
4. Complete the five exercises in Section 19.
5. Use the reviewer questions in Section 21 to test yourself.

### Deep track: about two weeks

| Day | Topic | Concrete outcome |
|---|---|---|
| 1 | Cellular end-to-end path | Draw UE → RAN → core → data network from memory. |
| 2 | Radio intuition and protocol stack | Explain PHY versus MAC and control versus user plane. |
| 3 | gNB/CU/DU/RU and O-RAN | Place near-RT RIC, xApp, E2, A1, O1, and SMO correctly. |
| 4 | RNTIs and KPI glossary | Explain why an RNTI is useful for enforcement but not identity. |
| 5 | Dataset and attacks | Explain what every label means and why Slowloris is difficult. |
| 6 | Chronology, blocks, leases, epochs | Reconstruct the raw-row-to-decision pipeline. |
| 7 | Logistic risk scoring | Derive the logistic equation and explain train-only preprocessing. |
| 8 | Calibration and detector metrics | Distinguish ranking, calibration, and action utility. |
| 9 | Sequential state machines | Hand-simulate `ALLOW`/`RESTRICT`/`ISOLATE`. |
| 10 | Friction-constrained selection | Explain the feasible set and matched reference without notes. |
| 11 | Time-weighted and episode metrics | Calculate friction, exposure, churn, coverage, and delay by hand. |
| 12 | Leakage-resistant evaluation | Defend chronological blocks, unseen RNTIs, and one-epoch lag. |
| 13 | Bootstrap inference and sensitivity | Explain the resampling unit and every important failure analysis. |
| 14 | Paper defense | Answer all questions in Section 21 in your own words. |

You are ready to present the paper when you can explain the entire chain below without describing the work as “an ML attack classifier”:

```text
radio/MAC observations
        ↓ causal 1-second aggregation
calibrated attack-risk score
        ↓ sequential policy selected under a benign-friction budget
ALLOW / RESTRICT / ISOLATE intent
        ↓ one-epoch delayed offline replay
security × friction × stability × delay evaluation
```

## 3. Cellular networking from first principles

### 3.1 The four parts of a mobile connection

A simplified cellular system has four parts:

```text
User equipment          Radio access network       Mobile core        Data network
(phone/modem/IoT)       (cell site/base station)   (control + data)   (Internet/apps)
       UE          ⇄          RAN             ⇄        Core       ⇄      Service
```

- The **UE** is the radio endpoint: phone, modem, sensor, vehicle unit, or test handset.
- The **RAN** establishes and maintains the radio link, schedules shared spectrum, adapts transmission parameters, and carries user/control traffic between the UE and core.
- The **core network** authenticates subscribers, manages mobility/session context, applies policy, and routes user-plane packets.
- The **data network** is the application side: Internet, enterprise network, edge service, voice platform, and so on.

The RAN is therefore not “the whole mobile network.” It is the radio-facing access segment.

### 3.2 Why radio is different from Ethernet

Radio resources are shared and the channel changes continually. Signal blockage, distance, interference, movement, multipath, transmit power, and competing users all affect what can be sent. The scheduler must repeatedly decide:

- which UE transmits or receives;
- on which time-frequency resources;
- at what modulation and coding rate;
- with what retransmission or power-control response; and
- how queued traffic should be prioritized.

These decisions generate the KPIs used in the paper. A risk model can observe a pattern in them, but it must not automatically interpret every unusual radio condition as malicious. A train entering a tunnel and an attack can both disturb telemetry for very different reasons.

### 3.3 Uplink, downlink, user plane, and control plane

- **Uplink (UL)** goes from UE to network.
- **Downlink (DL)** goes from network to UE.
- The **user plane** carries application traffic.
- The **control plane** carries signaling used to establish, configure, move, and release connections.

The distinction matters because an RNTI is a radio-side handle and the paper's KPIs are mostly radio/MAC observations. The study does not observe a complete subscriber-authentication or application-authorization transaction.

## 4. The RAN protocol stack

Think of each layer as solving a different part of reliable shared-radio delivery. Exact details differ between LTE and 5G NR, but the following conceptual map is enough to understand this paper.

| Layer | Main job | Examples relevant to this study |
|---|---|---|
| PHY | Apply coding/modulation and convert bits to/from radio waveforms; perform radio measurements | SINR, RSSI, samples, applied modulation/coding |
| MAC | Schedule shared resources, multiplex traffic, report buffers, coordinate fast retransmission behavior | CQI/MCS use, bit rate, OK/NOK counts, BSR, TTI, buffer |
| RLC | Segment/reassemble data and optionally retransmit missing pieces | Not directly modeled here |
| PDCP | Sequence handling, header compression, ciphering/integrity functions depending on plane | Not directly modeled here |
| SDAP (5G user plane) | Map QoS flows to radio bearers | Relevant to a future real `RESTRICT` implementation, not observed here |
| RRC | Configure radio resources, connection state, measurements, and mobility | Context for attachment/lifecycle; not an input here |

### 4.1 PHY intuition

Scheduler/link-adaptation logic selects how aggressively to transmit using CQI and other state; the PHY applies the selected coding/modulation and performs radio processing/measurements. A clean link can support a denser modulation and higher coding rate, moving more bits per resource. A noisy link needs a more robust scheme. Important quantities include:

- **SINR**: desired signal power relative to interference plus noise. Higher usually supports more aggressive transmission.
- **RSSI**: received total signal power indicator. Stronger is not always cleaner because RSSI also includes interference/noise.
- **MCS**: modulation-and-coding scheme index. It summarizes a chosen trade-off between robustness and spectral efficiency.
- **CQI**: UE-reported channel-quality indicator used to help choose a suitable downlink transmission mode/MCS.

Do not assume a universal linear relationship or unit from a CSV column name. Values depend on radio generation, implementation, collector, and configuration.

### 4.2 MAC intuition

The MAC layer allocates time-frequency opportunities and reacts quickly to demand and delivery success. Important quantities include:

- **bit rate**: recent delivered/offered UL or DL rate;
- **OK/NOK counters**: successful versus unsuccessful transmission-related outcomes in the collector;
- **BSR (Buffer Status Report)**: indication of pending uplink data at a UE;
- **DL buffer**: queued downlink data waiting for service;
- **TTI count**: transmission-time scheduling activity in the measurement interval;
- **PHR (Power Headroom Report)**: information about remaining UE transmit-power headroom.

High traffic, growing buffers, or many failures may be informative, but none is intrinsically an attack. The classifier learns associations in the controlled dataset; it does not discover a universal law of radio security.

### 4.3 Scheduling, resource blocks, and retransmission

LTE and 5G NR divide radio capacity across time and frequency. OFDM represents the channel with many narrow subcarriers; a scheduler allocates groups of time-frequency resources to UEs. In NR, a physical resource block spans 12 subcarriers, while slot duration depends on numerology. A 10 ms radio frame contains 1 ms subframes and a numerology-dependent number of slots.

This explains two important scale differences:

- PHY/MAC scheduling decisions occur far more frequently than the paper's one-second policy decision.
- A one-second KPI vector summarizes many transmissions; it is not one native radio transaction.

Fast link recovery often uses HARQ: the receiver acknowledges success/failure and the sender can retransmit with combined decoding information. Higher-layer RLC may also provide retransmission depending on its mode. The CSV's `OK`/`NOK` names should not automatically be equated to one exact standardized HARQ counter without collector documentation.

### 4.4 Bearers and QoS

Application packets are carried through logical radio bearers rather than being scheduled as anonymous Internet packets. In 5G, QoS flows from the core side are mapped to data radio bearers; SDAP and lower layers help preserve that treatment over radio.

A real `RESTRICT` implementation might change bearer/QoS/scheduling treatment. This paper cannot claim such an effect because the trace contains neither bearer-control actions nor post-action QoS measurements.

## 5. From conventional RAN to O-RAN

### 5.1 Functional disaggregation

A disaggregated base station can be understood as three processing regions:

```text
          higher-layer processing      time-sensitive processing       radio front end
Core ⇄        O-CU / CU      --F1--        O-DU / DU      --Open Fronthaul-- O-RU ⇄ UE
```

- **CU (Central Unit)** is a logical node hosting RRC, SDAP, and PDCP functions and can be split into control-plane and user-plane components.
- **DU (Distributed Unit)** is a logical node hosting RLC, MAC, High-PHY, and latency-sensitive scheduling-related functions.
- **O-RU (O-RAN Radio Unit)** is a physical node terminating Open Fronthaul and hosting Low-PHY/RF functions near the antennas.

O-CU, O-DU, and the RICs are logical nodes/functions; O-RU is explicitly a physical node. Nodes/functions can be bundled or colocated in several deployment arrangements, but that does not make the O-RU a virtual logical function.

### 5.2 Why “Open” matters

O-RAN aims to make interfaces and control functions more open, interoperable, programmable, and disaggregated. This creates opportunities for optimization and security applications, but also introduces new interfaces, software supply-chain concerns, and control components that must themselves be secured.

### 5.3 The RIC hierarchy

The RAN Intelligent Controller separates policy/optimization by timescale:

```text
Service Management and Orchestration (SMO)
  └── Non-RT RIC + rApps: policy, analytics, model lifecycle; generally > 1 s
                    │
                   A1
                    ↓
Near-RT RIC + xApps: near-real-time control; roughly 10 ms–1 s
                    │
                   E2
                    ↓
              O-CU / O-DU / E2 nodes

SMO ── O1 ── managed O-RAN functions
SMO ── O2 ── O-Cloud infrastructure
```

- An **rApp** commonly runs in the non-real-time management/analytics environment.
- An **xApp** runs on the near-RT RIC and can consume telemetry or request control actions.
- **E2** connects the near-RT RIC to E2 nodes such as O-CU/O-DU functions.
- **A1** carries policy/guidance and related information between non-RT and near-RT control.
- **O1** supports management, configuration, fault, and performance functions.
- **O2** connects the SMO to O-Cloud infrastructure management/deployment services.
- **Open Fronthaul** connects O-DU and O-RU in the relevant split.

O-RAN E2 service models include measurement exposure such as **E2SM-KPM** and control capabilities such as **E2SM-RC**. That provides a plausible deployment mapping for a KPI-driven controller.

However, the public trace in this project was collected by the authors' testbed instrumentation/custom RIC path. The CSV is **not proven to be a standardized E2SM-KPM report**, and the paper does not deploy its actions through E2SM-RC. In the manuscript, `RESTRICT` and `ISOLATE` are policy intents evaluated by offline replay.

### 5.4 E2, E2AP, and service models

These terms describe different layers:

- **E2** is the logical interface between a near-RT RIC and an E2 node.
- **E2AP** supplies generic procedures such as service discovery, subscription, indication/report delivery, and control signaling.
- An **E2 service model (E2SM)** defines the domain-specific information and behavior for one exposed RAN function.
- **E2SM-KPM** defines performance-measurement reporting semantics.
- **E2SM-RC** defines RAN-control-related semantics for supported functions.

An E2 connection does not imply that every KPI or action exists. The E2 node advertises supported functions, and an xApp must use the procedures/service model actually implemented. O2, separately, connects the SMO toward O-Cloud infrastructure; it is not the telemetry/control path used in this paper's conceptual loop.

### 5.5 LTE, NR, and the dataset's radio generation

- **LTE** is the 4G radio system; its base station is an eNB and its RAN is E-UTRAN.
- A **gNB** provides NR access within NG-RAN. NG-RAN may also contain an ng-eNB providing E-UTRA toward the 5GC, so NG-RAN is not synonymous with NR-only access.
- **5GC** is the 5G Core.
- **O-RAN** is an open/disaggregated/programmatic RAN architecture and can relate to LTE or NR.
- “Open RAN” is a broader industry concept; “O-RAN” usually refers more specifically to the O-RAN Alliance architecture/specifications.

O-RAN also does not imply that all implementation code is open source.

### 5.6 Do not overclaim the radio generation

The source work describes an OpenIreland/srsRAN environment and a 4G/5G-capable software/testbed stack. Several field names are LTE-like; for example, `phy_ul_turbo_iters` evokes LTE turbo decoding, whereas NR shared data channels use LDPC coding. The inspected public sources do not conclusively establish that every row in this CSV is a production 5G NR trace. Use 5G/NG-RAN architecture to understand the CFP and deployment context, but describe the evidence itself as **O-RAN-style RAN telemetry from the documented testbed**, not proof of production 5G NR performance.

Authoritative background: [3GPP/ETSI NG-RAN architecture (TS 38.401)](https://www.etsi.org/deliver/etsi_ts/138400_138499/138401/18.07.00_60/ts_138401v180700p.pdf), [O-RAN architecture (ETSI TS 103 982)](https://www.etsi.org/deliver/etsi_ts/103900_103999/103982/08.00.00_60/ts_103982v080000p.pdf), [O-RAN E2 general aspects (ETSI TS 104 038)](https://www.etsi.org/deliver/etsi_ts/104000_104099/104038/04.01.00_60/ts_104038v040100p.pdf), [E2AP procedures (ETSI TS 104 039)](https://www.etsi.org/deliver/etsi_ts/104000_104099/104039/04.00.00_60/ts_104039v040000p.pdf), and [the E2 service-model framework/common elements (ETSI TS 104 040)](https://www.etsi.org/deliver/etsi_ts/104000_104099/104040/04.00.00_60/ts_104040v040000p.pdf).

## 6. RNTI: the central enforcement handle

### 6.1 What an RNTI is

An **RNTI (Radio Network Temporary Identifier)** is a short radio-interface identifier used to address or distinguish radio-related procedures for a UE/context. There are several RNTI types for different purposes. A C-RNTI, for example, is associated with a connected UE context in a cell; temporary forms also participate in access procedures. The exact type represented by a collector field should not be assumed without metadata.

The key word is **temporary**.

### 6.2 What an RNTI is not

An RNTI is not automatically:

- a person;
- a subscriber account;
- an IMSI/SUPI or other durable subscriber identifier;
- a permanent device identity;
- globally unique;
- stable across detach, handover, reassignment, or time.

The same numeric value can be reused, and a UE's radio identifier can change. Therefore “unseen RNTI” in this paper means only that a numeric RNTI value did not occur in earlier partitions. It does **not** mean a cryptographically new user or device.

### 6.3 Why it is still useful

The RAN needs an addressable radio context on which a network action can operate. Within each inferred lease, this study treats the observed numeric `mac_rnti` as an ephemeral grouping and requested-action handle; the CSV does not verify its allocation, validity, release, handover, or attachment lifecycle. A real deployment key would need at least E2-node/cell context, RNTI, and lifecycle generation because the number is not network-global. This supports statements such as:

> “The offline controller requests a restrictive treatment for this currently observed RNTI lease.”

It does not support:

> “The system denied Alice access to confidential resource X.”

The project therefore creates an **RNTI lease**: a label-blind, inferred lifecycle scoped to one observable trace block and reset after 30 seconds of RNTI inactivity. This prevents controller state from leaking across obviously separated appearances of a reused numeric value. It remains an analytical proxy, not a logged allocation/release lifecycle.

For standards context, see [3GPP/ETSI NR overall description (TS 38.300)](https://www.etsi.org/deliver/etsi_ts/138300_138399/138300/18.09.00_60/ts_138300v180900p.pdf).

## 7. KPI glossary for this dataset

The final logistic model uses 25 dynamic KPI columns. They are easier to remember in functional groups.

### 7.1 Link quality and adaptation

| Columns | Working interpretation |
|---|---|
| `mac_dl_cqi` | Downlink channel-quality indication exposed by the collector |
| `mac_dl_mcs`, `phy_dl_mcs`, `phy_ul_mcs` | Selected modulation/coding indices |
| `phy_ul_pusch_sinr`, `phy_ul_pucch_sinr` | Uplink shared/control-channel SINR |
| `phy_ul_pusch_rssi`, `phy_ul_pucch_rssi` | Received power indicators for UL shared/control channels |
| `phy_ul_pucch_ni` | Control-channel noise/interference indicator |
| `mac_dl_cqi_offset`, `mac_ul_snr_offset` | Collector/control offsets associated with link adaptation |

`PUSCH` is the Physical Uplink Shared Channel, normally used for uplink user data and some signaling. `PUCCH` is the Physical Uplink Control Channel, used for uplink control information.

### 7.2 Traffic, queues, and scheduling activity

| Columns | Working interpretation |
|---|---|
| `mac_dl_brate`, `mac_ul_brate` | Downlink/uplink bit-rate measurements |
| `mac_ul_bsr` | Uplink buffer-status information |
| `mac_dl_buffer` | Downlink queued data |
| `mac_nof_tti` | Number of transmission-time intervals represented |
| `phy_ul_n_samples`, `phy_ul_n_samples_pucch`, `phy_dl_n_samples` | Counts of collector samples/observations in the interval |

### 7.3 Reliability and processing response

| Columns | Working interpretation |
|---|---|
| `mac_dl_ok`, `mac_dl_nok`, `mac_ul_ok`, `mac_ul_nok` | Successful/unsuccessful delivery-related counters as exported |
| `phy_ul_turbo_iters` | Iterative decoder effort in the source stack |

### 7.4 Power headroom

| Column | Working interpretation |
|---|---|
| `mac_phr` | UE power-headroom information: how much uplink transmit-power margin remains under the applicable reporting model |

The guide intentionally says “working interpretation.” The public CSV does not supply authoritative per-column units and collector semantics for every field. Never convert these names into precise physical claims that the source does not document.

### 7.5 Excluded fields and why

The model excludes:

- `mac_rnti` and duplicate `ue_ident`: identity/scenario leakage and memorization risk;
- `id_ue`: source-side UE/scenario context excluded to prevent scenario/entity leakage; the public data do not establish it as a durable subscriber identity;
- `timestamp` and derived times: capture/label schedule leakage;
- `mob_pattern`: scenario metadata, reserved for audit/stratification;
- `label` and derived targets: ground truth, never a predictor;
- constants and known export artifacts;
- `samples_in_epoch`: excluded by the locked protocol so report density does not become an easy capture signature.

This is an explicit allowlist: a newly appearing numeric column is not silently admitted as a feature.

Do not confuse the derived `samples_in_epoch` count with permitted source KPI fields such as `phy_ul_n_samples`, `phy_ul_n_samples_pucch`, and `phy_dl_n_samples`. The former counts raw CSV reports placed in a one-second analytical bin; the latter are collector-exported numeric KPIs whose precise semantics remain source-specific.

## 8. Dataset, traffic, and threat model

### 8.1 Source

The study uses the public OpenIreland dataset **“RAN Performance measurements for security threats”**. The local immutable source is:

```text
/nobackup/ashukuma/o_ran/dtst.csv
```

Its frozen SHA-256 is:

```text
4cc8498466eb7cb258412721ae94a2460f04a7da1235ac07e8e9cd20e15a76a7
```

The source contains 3,175,140 semicolon-delimited rows and 47 columns. The dataset is released under CC BY 4.0. See the [official data record](https://data.mendeley.com/datasets/t2rzh9y4mp/1) and its [Computer Networks data paper](https://doi.org/10.1016/j.comnet.2024.110710).

### 8.2 Labels

Four labels represent ordinary/application traffic:

| Label | Meaning in the experiment |
|---|---|
| `Web Browsing` | Interactive web traffic |
| `SIPP` | SIP/VoIP-style generated traffic |
| `youtube` | Video-streaming traffic |
| `iot` | IoT-style traffic |

Four labels represent attacks:

| Label | High-level behavior | Why KPIs may reveal it |
|---|---|---|
| `portscan` | Probe many ports/services | Repeated connections and changed traffic pattern |
| `ddos-ripper-C` | High-rate connection/request flooding | Sustained load, rate, queue, and delivery changes |
| `dos-hulk-C` | High-rate HTTP/SYN/GET-style flooding | Strong throughput and scheduling signature |
| `slowloris-C` | Hold many server connections open with small, slow partial requests | Low-rate behavior can resemble benign traffic |

The source authors' early-detection paper describes the testbed and attack generators in more detail: [Xavier et al., IEEE ICC 2023 author preprint](https://arxiv.org/pdf/2302.01864).

### 8.3 Why Slowloris matters

Port scans and flooding attacks often create conspicuous bursts or sustained high traffic. Slowloris is deliberately low and slow: it consumes server connection state while sending little data. A model built from radio/MAC load indicators can therefore rank it poorly or inconsistently.

This is not a minor footnote. It tests whether the proposed controller succeeds only on attacks that strongly perturb RAN KPIs. The final results confirm that boundary: the controller helps on the three high-rate/probing attacks but worsens Slowloris coverage and delay.

### 8.4 Threat model

The defensible threat model is narrow:

1. An observed numeric RNTI lease is assumed to represent an already attached radio context; the CSV does not log authoritative attachment lifecycle.
2. Its observed traffic becomes one of the dataset's attack classes.
3. A trusted telemetry path supplies radio/MAC KPIs.
4. A trusted score-to-action controller can request `ALLOW`, `RESTRICT`, or `ISOLATE` treatment for that observed lease.
5. The study evaluates those requested states by replaying a fixed trace.

Out of scope are telemetry forgery, RIC compromise, durable identity theft, credential authentication, per-resource authorization, attacker adaptation to enforcement, scheduler realization, and causal QoS effects.

These labels are controlled UE-originated application/network-service traffic classes observed indirectly through RAN KPIs. They are not radio jamming, rogue-base-station attacks, signaling storms, E2/A1 attacks, or RIC compromise. The `DDoS` generator/class name also does not establish a field-scale distributed botnet.

### 8.5 Controlled labels are not natural incidents

The trace was generated in controlled experiments. A benign-to-attack label change can be used as a replayed attack onset, but it is not evidence that a real subscriber was naturally compromised at that instant. Use the phrase **controlled attack-class onset in recorded telemetry**, not “observed account takeover.”

## 9. Zero trust and the paper's exact mapping

### 9.1 Zero-trust principle

NIST zero-trust architecture rejects implicit trust based only on network location and continually evaluates access using policy and available context. An ongoing session can be continued, limited, or revoked. See [NIST SP 800-207](https://csrc.nist.gov/pubs/sp/800/207/final) and the [NIST reference implementation architecture](https://pages.nist.gov/zero-trust-architecture/VolumeB/architecture.html).

Three conceptual functions are useful:

- **Policy Information Point (PIP)**: supplies evidence/context.
- **Policy Decision Point (PDP)**: evaluates policy and chooses an outcome.
- **Policy Enforcement Point (PEP)**: applies the outcome to traffic/access.

In the NIST logical model, the PDP comprises the Policy Engine and Policy Administrator, while the PEP is separate. This replay implements no PEP.

### 9.2 Mapping used in this study

| Zero-trust concept | Study proxy |
|---|---|
| PIP evidence | RAN KPI vector and calibrated risk stream |
| Policy-engine/PDP-like logic | Sequential controller |
| Policy-administration/output proxy | Requested `ALLOW`, `RESTRICT`, `ISOLATE` intent for the observed RNTI lease |
| PEP | Absent from this replay; prospective O-CU/O-DU/bearer/scheduler/filter enforcement |
| Continuous-evaluation proxy | Repeated decisions on nonempty observed one-second bins |
| Operational cost | Benign RNTI-time placed in `RESTRICT` or `ISOLATE` |

This mapping makes the work relevant to adaptive control and continual verification in next-generation networks.

### 9.3 Where the mapping ends

The dataset contains no reliable durable principal, credential, MFA event, requested application resource, resource sensitivity, policy rule, authorization grant, or enforced outcome. Accordingly:

- `risk_score` is not an authentication confidence score;
- an RNTI is not a user identity;
- `RESTRICT` is not a measured scheduler slice or rate limit;
- `ISOLATE` is not a logged radio release or packet drop;
- benign friction is not measured user experience;
- lower replayed exposure is not proven attack prevention.

This paper is a **network-level continuous-containment study inspired by zero-trust control**, not a complete zero-trust identity system. LANL authentication data would be a better primary source for genuine identity/adaptive-authentication claims.

## 10. From raw rows to causal decision epochs

The hardest part of a trustworthy telemetry experiment is often not the model; it is defining time and independent units without looking at labels.

### 10.1 Five data units

| Unit | Definition in this project | Count |
|---|---|---:|
| Raw report | One source KPI record | 3,175,140 |
| Causal epoch | One observed 1-second bin for one RNTI lease | 525,271 |
| RNTI lease | Inferred, block-scoped lifecycle of a numeric RNTI | 5,892 |
| Trace block | Label-blind time block used for splitting/inference | 273 |
| Split | Complete blocks assigned to one experimental role | 4 |

### 10.2 Schema and integrity audit

Before modeling, code verifies:

- exactly the expected 47 columns and order;
- known, non-null labels;
- finite timestamps;
- zero exact duplicate source rows;
- expected constant fields;
- `mac_rnti == ue_ident` on every row; and
- chronology statistics, for which the frozen audit records one raw source-order timestamp reversal.

The duplicate identifier column is evidence of an export/schema relationship, not independent identity evidence.

### 10.3 Timestamp ambiguity

The data article calls the raw timestamp “microseconds” but does not specify the encoding or epoch. The actual magnitude creates an inconsistency:

- raw / `100000` maps to 24–27 October 2022 and gives a median within-stream interval near 89.6 ms;
- raw / `1000000` maps to April 1975 and gives a median near 8.96 ms.

The related testbed work reports roughly 100 ms collection cadence, and the 2022 calendar is plausible. The primary pipeline therefore uses raw / `100000`, but calls it an **empirically inferred time scale**, not confirmed Unix microseconds. Under raw /`1000000`, preparation, model fitting/scoring, and tune-only policy feasibility are rerun; that branch stops fail-closed before held-out controller replay because it yields no feasible primary proposal.

### 10.4 Chronological correction

The source has one backward timestamp jump in row order; its cause is not author-confirmed. The code retains the raw timestamp, adds scaled event time, and stably sorts by event time plus original row index. It does not assume file row order is time order.

### 10.5 Label-blind trace blocks

A new trace block starts when either:

- the global observed gap exceeds 300 seconds; or
- the UTC-aligned 900-second time bin changes.

No label, RNTI, UE field, mobility value, or model score creates a block boundary. Blocks are useful grouping surrogates, but the dataset does not provide trustworthy capture/campaign IDs. Therefore say **trace-block-held-out**, not “independent-capture-held-out.”

### 10.6 RNTI leases

Within a block, a numeric RNTI begins a new inferred lease after more than 30 seconds of inactivity. A lease also necessarily ends at a block boundary. Its ID is structurally similar to:

```text
trace_block_id : numeric_rnti : within-block sequence
```

The lease rule uses time and RNTI only; it never resets when a benign label changes to an attack label. A label-based reset would tell the controller when the attack begins.

### 10.7 One-second causal aggregation

For each `(trace block, RNTI lease, numeric RNTI, one-second bin)`:

- KPI features use the last observation available in the bin;
- `decision_time_s` is the right boundary of the bin;
- the number of raw reports is retained for audit but excluded from the model;
- all labels observed in the bin are retained as a list for later evaluation;
- an epoch is an attack epoch if any attack label appears;
- empty seconds are not fabricated or silently labeled benign.

Consequently, “RNTI-time” in this study means **observed entity-time represented by epochs**, not continuous attachment occupancy including unobserved gaps.

Some source fields may be gauges, interval deltas, or monotonic counters. Last-in-bin aggregation does not turn a cumulative counter into a rate and does not handle an undocumented wrap/reset. `OK`/`NOK`, TTI, and source sample-count values are therefore treated only as exported numeric features—not standardized packet-loss or HARQ rates—pending authoritative collector semantics.

### 10.8 Chronological partitions

Whole blocks are assigned by time. A block crossing a cut is assigned wholly to the later split.

| Split | Epochs | Permitted purpose |
|---|---:|---|
| Train | 246,193 | Fit imputation, scaling, and base risk model |
| Calibration | 93,989 | Fit probability calibration |
| Controller tune | 83,528 | Match budgets and select controller structure/thresholds |
| Test | 101,561 | Frozen controller-policy replay and inference |

This separation prevents the same observations from serving incompatible roles.

## 11. Risk-scoring theory

The risk scorer is deliberately conventional because it is not the paper's main novelty.

### 11.1 Logistic regression

Let the standardized 25-KPI vector for lease $i$, epoch $t$ be $x_{it}\in\mathbb{R}^{25}$. Logistic regression forms a linear log-odds score:

\[
f(x_{it})=\beta_0+\beta^\top x_{it}.
\]

The sigmoid maps it to a number between zero and one:

\[
\sigma(z)=\frac{1}{1+e^{-z}}, \qquad p_0(x)=\sigma(f(x)).
\]

The parameters minimize weighted binary cross-entropy plus L2 regularization:

\[
\min_{\beta_0,\beta}
\sum_j w_j\left[-y_j\log p_j-(1-y_j)\log(1-p_j)\right]
+\frac{\lambda}{2}\lVert\beta\rVert_2^2.
\]

- Cross-entropy penalizes confident wrong predictions strongly.
- Class-balanced training weights prevent the larger class from dominating the fitted boundary.
- L2 regularization discourages extreme coefficients and reduces variance.
- `C=1` in the implementation is the inverse regularization-strength convention used by scikit-learn.

### 11.2 Train-only preprocessing

Missing-value medians and standardization parameters are fit exactly once on `train`.

For feature $k$:

\[
z_{jk}=\frac{x_{jk}-\mu_k^{\text{train}}}{\sigma_k^{\text{train}}}.
\]

Using calibration, tuning, or test values to compute the median/mean/standard deviation would leak later-distribution information into the model.

### 11.3 Platt calibration

Class weighting and regularization can make a model useful for ranking without making its output probability scale trustworthy. A separate one-dimensional logistic calibrator is fit on the calibration partition:

\[
s(x)=\sigma(\gamma_0+\gamma_1 f(x)).
\]

This is Platt scaling. It changes score scale and spacing; with a positive slope it preserves ranking. The resulting `risk_score` is the only continuous evidence supplied to the controller.

Even after calibration, a value like 0.955 should not be marketed universally as “95.5% certainty.” Calibration is conditional on this trace, sampling scheme, labels, and time period. Score calibration can drift.

### 11.4 AUROC, AUPRC, and Brier score

**AUROC** measures ranking across every possible threshold. One interpretation is:

\[
\Pr(s_{\text{attack}}>s_{\text{benign}})
+\tfrac12\Pr(s_{\text{attack}}=s_{\text{benign}}).
\]

**AUPRC** summarizes precision/recall trade-offs and is often more informative when attacks are rare.

**Brier score** is mean squared probability error:

\[
\operatorname{Brier}=\frac{1}{n}\sum_j(s_j-y_j)^2.
\]

These measure ranking/calibration, not operational policy quality. A high AUROC does not guarantee:

- a feasible 1% benign-friction threshold;
- few action transitions;
- fast episode containment;
- full isolation;
- Slowloris performance; or
- transfer to another day/network.

The primary logistic score has test AUROC 0.9513. The HGB sensitivity has higher AUROC, 0.9748, yet its selected controller exceeds the held-out friction budget. That is a concrete demonstration that classifier ranking is not the paper's final objective.

### 11.5 Alternative nonlinear scorer

The sensitivity model is a shallow histogram gradient-boosted tree ensemble (depth 3, at most 7 leaves, learning rate 0.05, 150 iterations, L2 regularization). It uses the same allowed features and partition discipline. Its job is to test whether the **selection/evaluation framework** survives a different risk stream, not to start a model leaderboard.

## 12. Score-to-action controller theory

### 12.1 Detection versus sequential control

The system has two distinct functions:

\[
x_{it}\xrightarrow{\text{risk model}}s_{it}
\xrightarrow{\text{sequential controller}}b_{it}
\xrightarrow{\text{one-epoch lag}}a_{i,t+1}.
\]

- $s_{it}\in[0,1]$ is calibrated evidence.
- $b_{it}$ is the decision computed after observing epoch $t$.
- $a_{i,t+1}$ is the effective action in the next observed epoch.

The controller API receives score, time, numeric RNTI, and lease ID. It does not accept a label.

### 12.2 Stateless reference

The locked stateless comparator independently thresholds every causal decision epoch:

\[
b_{it}=\begin{cases}
\text{ISOLATE},&s_{it}\ge h,\\
\text{ALLOW},&s_{it}<h.
\end{cases}
\]

Its restrict and isolate thresholds are equal, so it jumps directly between `ALLOW` and `ISOLATE`. This makes malicious `ALLOW` and malicious not-`ISOLATE` identical for the comparator.

A score sequence near $h$, such as $(0.94,0.96,0.94,0.96)$, makes the action alternate repeatedly. This is policy churn or threshold chatter.

### 12.3 EWMA

An exponentially weighted moving average uses elapsed time:

\[
m_t=m_{t-1}+\alpha_t(s_t-m_{t-1}),
\qquad
\alpha_t=1-e^{-\Delta t/\tau}.
\]

The time constant $\tau$ controls memory. Larger $\tau$ suppresses brief spikes more strongly but delays response and recovery. For one-second spacing and $\tau=2$ s, $\alpha\approx0.393$.

### 12.4 N-report persistence

An N-report controller requires a transition condition to hold for N consecutive reports. It behaves like switch debouncing:

- transient one-report excursions are ignored;
- genuine changes incur at least N−1 reports of added evidence delay;
- missing scores break the consecutive-evidence sequence.

Short attacks can end before the entry condition persists long enough.

### 12.5 Hysteresis

Hysteresis uses different entry and recovery thresholds:

\[
h_{\downarrow}<h_{\uparrow}.
\]

Enter a state at or above $h_{\uparrow}$, but leave it only below $h_{\downarrow}$. Between the thresholds, retain the current state. This dead band prevents oscillation around a single threshold.

Hysteresis is a standard control primitive. Merely applying it would not be a sufficient research contribution.

### 12.6 Three-state asymmetric controller

Let:

- $(r_\uparrow,r_\downarrow)$: `RESTRICT` entry/recovery thresholds;
- $(q_\uparrow,q_\downarrow)$: `ISOLATE` entry/recovery thresholds.

They satisfy the ordered relationships required by the implementation:

\[
r_\downarrow\le r_\uparrow\le q_\uparrow,
\qquad
r_\downarrow\le q_\downarrow\le q_\uparrow.
\]

Ignoring report persistence for one moment, the state machine is:

```text
ALLOW
  ├─ score ≥ isolate_enter  ───────────────→ ISOLATE
  └─ score ≥ restrict_enter ───────────────→ RESTRICT

RESTRICT
  ├─ score ≥ isolate_enter  ───────────────→ ISOLATE
  └─ score < restrict_exit  ───────────────→ ALLOW

ISOLATE
  └─ score < isolate_exit   ───────────────→ RESTRICT
```

Recovery is staged: `ISOLATE → RESTRICT → ALLOW`. Escalation can jump directly from `ALLOW` to `ISOLATE`.

The locked primary policy is:

| Parameter | Value |
|---|---:|
| `restrict_enter` | 0.9550303 |
| `restrict_exit` | 0.9050303 |
| `isolate_enter` | 0.9968761 |
| `isolate_exit` | 0.9468761 |
| Entry reports | 1 |
| Recovery reports | 2 |
| EWMA | none |
| Minimum state holds | 0 s |

Thus the chosen mechanism escalates immediately but requires two confirming reports for **each** downward state transition. Because recovery is staged, a full `ISOLATE → RESTRICT → ALLOW` recovery normally needs two reports below `isolate_exit`, then a fresh two-report sequence below `restrict_exit`; the computed transitions are also subject to the one-epoch action lag. It expresses the policy preference “respond quickly; restore service cautiously.”

### 12.7 Hand simulation

Use a simplified restrict-only controller with entry 0.95, exit 0.90, and two-report recovery:

| Epoch | Score | Computed decision | Effective action | Explanation |
|---:|---:|---|---|---|
| 0 | .20 | ALLOW | ALLOW | New lease starts ALLOW |
| 1 | .96 | RESTRICT | ALLOW | Decision cannot act retroactively |
| 2 | .89 | RESTRICT | RESTRICT | First low recovery report |
| 3 | .96 | RESTRICT | RESTRICT | Low sequence breaks |
| 4 | .89 | RESTRICT | RESTRICT | First low report again |
| 5 | .88 | ALLOW | RESTRICT | Second low report permits recovery decision |
| 6 | .20 | ALLOW | ALLOW | Recovery becomes effective |

A stateless threshold would switch on almost every crossing. The sequential controller is stable because it retains the restrictive state across small/short dips. The cost is extra benign restrictive time and potentially slower recovery.

### 12.8 Why one-epoch lag is essential

KPIs for epoch $t$ are known only after the epoch's observations exist. Applying the decision to the same epoch would use future-complete information to claim retroactive protection. The implementation enforces:

\[
a_{i0}=\text{ALLOW},\qquad a_{i,t+1}=b_{it}.
\]

This prevents look-ahead bias and gives onset delay a meaningful lower bound. Controller state resets at a lease boundary, never at a ground-truth label transition.

## 13. Operational metrics

Let $d_{it}$ be the observed duration of an epoch, $y_{it}=1$ indicate an attack, and encode actions as `ALLOW=0`, `RESTRICT=1`, `ISOLATE=2`.

In the frozen primary representation, every retained bin has `epoch_seconds=1.0`, including a partially occupied edge bin. Thus the published ratios weight occupied one-second RNTI bins—not raw-report count and not continuously logged attachment time. Empty seconds contribute nothing.

### 13.1 Benign friction

\[
F(\pi)=
\frac{\sum d_{it}(1-y_{it})\mathbf{1}[a_{it}\ge1]}
{\sum d_{it}(1-y_{it})}.
\]

This is benign observed RNTI-time in `RESTRICT` or `ISOLATE`. Lower is better.

It is a policy occupancy proxy—not actual user complaints, QoE loss, throughput reduction, or financial cost. The primary formulation weights `RESTRICT` and `ISOLATE` equally, so benign isolate time is also reported separately.

### 13.2 Malicious `ALLOW` exposure

\[
E_A(\pi)=
\frac{\sum d_{it}y_{it}\mathbf{1}[a_{it}=0]}
{\sum d_{it}y_{it}}.
\]

This measures attack-labeled time left fully allowed. Lower is better.

### 13.3 Malicious not-`ISOLATE` exposure

\[
E_{NI}(\pi)=
\frac{\sum d_{it}y_{it}\mathbf{1}[a_{it}<2]}
{\sum d_{it}y_{it}}.
\]

It counts both `ALLOW` and `RESTRICT` as short of full isolation. Necessarily:

\[
E_{NI}\ge E_A.
\]

The difference $E_{NI}-E_A$ is malicious time in `RESTRICT`. Reporting both prevents a three-state controller from claiming universal success merely by moving attack time from `ALLOW` to `RESTRICT`.

### 13.4 Policy churn

\[
C(\pi)=
\frac{60\sum\mathbf{1}[a_{it}\ne a_{i,t-1}]}
{\sum d_{it}}.
\]

This is action transitions per observed RNTI-minute. Ten simultaneously observed RNTIs for one minute contribute ten RNTI-minutes. Escalations and recoveries both count; transitions across lease resets do not.

### 13.5 Episode coverage

An attack episode is covered if it reaches at least `RESTRICT`:

\[
K_e=\mathbf{1}[\exists t\in e:a_{it}\ge1],
\qquad
\operatorname{Coverage}=\frac{1}{N}\sum_{e=1}^{N}K_e.
\]

Headline coverage includes episodes whose RNTI is already at least `RESTRICT` at onset, possibly due to pre-attack risk. Such episodes are covered and receive delay zero. Time exposure and episode coverage answer different questions: one long missed episode can dominate time exposure, while many short missed episodes can dominate episode coverage. The saved onset strata should be inspected alongside the headline summary.

### 13.6 Delay

For a covered episode:

\[
D_e=T_{\text{first effective RESTRICT}}-T_{\text{attack onset}}.
\]

If already contained at onset, delay is zero. In capped summaries, a missed episode receives:

\[
D_e^{\text{cap}}=\min(\text{observed episode span},30\text{ s}).
\]

This finite miss penalty makes summaries computable, but it must be stated: a missed attack is not truly “detected after 30 seconds.” A short missed episode can receive a small capped value because its observed span is short. The median can remain small while a minority of long/slow attacks become much worse, so coverage, onset strata, mean capped delay, and per-attack results remain necessary safeguards.

### 13.7 Toy calculation

Suppose a replay contains:

- 100 benign RNTI-seconds, of which 1 is restricted;
- 50 malicious RNTI-seconds: 10 `ALLOW`, 20 `RESTRICT`, 20 `ISOLATE`;
- 6 action transitions over 150 observed RNTI-seconds.

Then:

\[
F=1/100=1\%,
\]

\[
E_A=10/50=20\%,
\]

\[
E_{NI}=(10+20)/50=60\%,
\]

\[
C=60(6)/150=2.4\text{ transitions/RNTI-minute}.
\]

This example shows why low `ALLOW` exposure need not mean strong full isolation.

## 14. Friction-budgeted policy selection

### 14.1 Why ordinary threshold comparison is unfair

Controller outcomes depend on aggressiveness. A low threshold will usually contain more attack time but disrupt more benign time. Comparing controller A at 0.1% friction with controller B at 5% friction confounds mechanism with operating cost.

The paper first calibrates every structure to a common benign-friction budget $B$, then compares security and stability.

### 14.2 Matched stateless reference

For each budget, the tuning data determine an exact stateless threshold from below. The threshold is the most aggressive feasible score boundary whose benign friction does not exceed $B$, respecting score ties and indivisible one-second epochs.

Each temporal controller structure is independently calibrated to the same nominal budget. A candidate must use at least 95% of $B$, apart from the mass of one indivisible benign epoch. This prevents a nearly always-`ALLOW` controller from appearing stable simply because it does nothing.

At the primary $B=1\%$ tuning point:

- proposed friction = 0.998797%;
- stateless friction = 0.998797%;
- proposed threshold structure and stateless threshold are therefore exactly matched in realized tuning friction.

Test friction is not rematched. The frozen policies are carried forward, so distribution shift is visible rather than repaired after the fact.

### 14.3 Feasible set

Let $\pi_B^0$ be the budget-matched stateless reference. The idealized budget constraint is:

\[
F(\pi)\le B,
\]

The implementation accounts for one-second discreteness. If $r$ is the maximum positive benign-epoch mass divided by benign entity-time, strict matching accepts:

\[
0.95B-r\le F(\pi)\le B+r.
\]

The stateless reference and threshold calibration approach $B$ from below; the atomic window prevents an unavoidable tied/indivisible epoch from causing an artificial rejection. Candidate policy $\pi$ must then satisfy these tune-only safeguards:

\[
E_A(\pi)-E_A(\pi_B^0)\le0.02,
\]

\[
E_{NI}(\pi)-E_{NI}(\pi_B^0)\le0.05,
\]

\[
\operatorname{Coverage}(\pi)-\operatorname{Coverage}(\pi_B^0)\ge-0.05,
\]

\[
\operatorname{MedianDelay}(\pi)-\operatorname{MedianDelay}(\pi_B^0)\le1\text{ s},
\]

\[
\frac{C(\pi)}{C(\pi_B^0)}\le0.75.
\]

The final inequality requires at least 25% churn reduction. The 1% budget, 2/5-percentage-point noninferiority margins, 1-second delay allowance, and 25% churn target are this study's declared design tolerances—not O-RAN, NIST, or operator standards. A deployment would require application-specific cost and safety justification.

The selected policy is:

\[
\pi_B^*=\arg\min_{\pi\in\Pi_B} C(\pi),
\]

where $\Pi_B$ is the set satisfying all requirements. More completely, the selection order is budget match, security noninferiority, hard-isolation safeguard, episode-coverage safeguard, capped-delay safeguard, closest budget stratum, and then minimum churn.

### 14.4 Candidate search space

The deterministic grid contains 120 structural templates:

- 1 stateless structure;
- 3 N-report structures;
- 4 EWMA structures;
- 4 symmetric-hysteresis structures;
- 108 asymmetric sequential structures.

They are calibrated at 0.1%, 0.5%, 1%, 2%, and 5% budgets: 600 structure-budget tasks. Initial calibration declares a structure infeasible when even its highest permitted threshold has irreducible friction above the budget. Score ties, atomic resolution, or severe budget undershoot are handled later by exact-reference and strict budget matching.

### 14.5 What is genuinely novel

EWMA, debounce, state machines, and hysteresis are established tools. The paper's novelty is the combination of:

1. an explicit benign-friction resource constraint;
2. a security/coverage/delay envelope;
3. exact budget-matched comparison;
4. minimum-churn selection within that feasible region;
5. causal label-blind replay with chronological group separation; and
6. an operational frontier that reports where no feasible or transferable policy exists.

It reframes the question from “Which smoother gives the best F1?” to:

> “Which causal action policy is stable enough while remaining inside a declared security and benign-intervention envelope?”

### 14.6 The frontier is multidimensional

Each policy is a point in:

\[
(F,E_A,E_{NI},C,\operatorname{Coverage},\operatorname{Delay}).
\]

There is rarely one universally best point. More friction may buy containment, while more smoothing may buy stability but cost delay. A candidate is Pareto-dominated if another candidate is no worse on all relevant outcomes and better on at least one.

“No feasible policy at this budget” is a result. Constraints must not be relaxed after test outcomes are seen.

## 15. Leakage-resistant evaluation protocol

### 15.1 Why random row splitting is invalid

Network telemetry is strongly autocorrelated. Adjacent rows share the same generator, RNTI, radio environment, scenario, capture configuration, and traffic phase. With random 80/20 rows, near-identical observations and the same temporary identifiers appear in both training and testing. A model can appear to generalize while learning a capture schedule or scenario fingerprint.

This is a standard security-ML danger; see [Arp et al., “Dos and Don'ts of Machine Learning in Computer Security”](https://www.usenix.org/conference/usenixsecurity22/presentation/arp).

### 15.2 Four-way separation

The four partitions solve different statistical problems:

| Partition | What may be learned | What must not be learned |
|---|---|---|
| Train | Feature preprocessing and classifier coefficients | Probability calibration or action thresholds |
| Calibration | Mapping raw model output to risk scale | Controller structure/thresholds |
| Controller tune | Budget thresholds and controller selection | Held-out outcomes |
| Test | Nothing; one frozen replay and inference | No retuning, recalibration, or candidate replacement |

The full trace and preliminary go/no-go behavior were inspected before the protocol was frozen, and risk diagnostics were produced for every partition. Therefore the test day is accurately described as a **frozen controller-policy outcome holdout**, not a pristine never-inspected dataset.

### 15.3 Whole-block chronological assignment

All epochs in a label-blind trace block remain in one partition. This prevents a block, lease, or controller state from crossing a split. It does not guarantee a large temporal gap between neighboring blocks/splits or eliminate dependence between blocks.

Chronology asks a realistic question: can a fixed model and policy selected earlier transfer to later telemetry? It is harder and more informative than random interpolation within the same experimental sequence.

Excluding RNTI, `id_ue`, mobility, and raw time blocks direct metadata leakage, but it cannot remove scenario-correlated patterns already present inside allowed KPIs. The evidence remains one controlled testbed, and the viable held-out day is effectively car-only.

### 15.4 Explicit controller information firewall

The action-generating controller call receives exactly:

- `risk_score`;
- `decision_time_s`;
- `mac_rnti` as the current subject key; and
- `rnti_lease_id` as the lifecycle key.

The replay table additionally carries evaluation-row ID, block ID, and epoch duration for joining/accounting; none enters controller state. The controller outputs actions, and only afterward are target labels and descriptive context joined by the unique evaluation-row key. Labels cannot influence state, persistence, or reset logic.

### 15.5 One-epoch actuation lag

The decision from epoch $t$ is shifted within the same lease to become effective at $t+1$. Every lease begins with effective `ALLOW`. No state is shifted across a lease boundary.

This avoids a common optimistic error: letting features measured during an attack second retroactively contain that same second.

### 15.6 Unseen numeric RNTIs

Let:

\[
\mathcal{U}=\{r:r\text{ occurs in test but nowhere in an earlier split}\}.
\]

The test data contain 744 numeric RNTIs: 470 seen earlier and 274 unseen. Because RNTI is already excluded from model features and serves only as a state key, this is not a direct ID-memorization test. It measures distributional transfer to numerically novel temporary handles; it does not establish subscriber/device identity novelty or generalization.

### 15.7 Attack episodes

Primary episodes are built per RNTI lease and attack stream:

- for the all-attack stream, an observed benign epoch ends the episode;
- for a type-specific stream, any observed epoch lacking that attack type ends it, including an epoch containing another attack type;
- missing telemetry gaps up to the frozen 30-second lease timeout may be bridged;
- labels do not reset controller state;
- mixed attack epochs use multi-hot membership and can simultaneously continue more than one type-specific stream.

Therefore per-attack episode counts are not additive. A strict zero-gap construction is preserved as a sensitivity.

The security margins were locked in advance, but the primary 30-second missing-telemetry episode alignment was documented in [`protocol_amendment_v1a_episode_alignment.json`](../configs/protocol_amendment_v1a_episode_alignment.json) after locked action replay and before bootstrap inference. It aligned evaluation with the tuning episode semantics, changed no actions or time-weighted outcomes, and retained strict zero-gap episodes as a mandatory sensitivity.

## 16. Statistical inference

### 16.1 Why not use rows as independent samples

There are 101,561 held-out epochs but only 56 primary trace blocks. Treating every epoch as independent would make confidence intervals artificially narrow because observations within a block are related.

### 16.2 Paired trace-block bootstrap

For each of 5,000 bootstrap replicates:

1. Sample 56 held-out trace-block IDs with replacement.
2. Include each selected block in its entirety, with multiplicity.
3. Use the **same block draw** for the proposed and reference policies.
4. Sum metric numerators and denominators across sampled blocks.
5. Form ratios only after aggregation.
6. Record the proposed-reference contrast.

For exposure, if block $b$ contributes numerator $N_{\pi b}$, denominator $D_b$, and bootstrap multiplicity $M_b^{(k)}$:

\[
E_{\pi}^{*(k)}=
\frac{\sum_bM_b^{(k)}N_{\pi b}}
{\sum_bM_b^{(k)}D_b}.
\]

The paired contrast is:

\[
\Delta_E^{*(k)}=E_P^{*(k)}-E_R^{*(k)}.
\]

Pairing is important because both controllers see the same telemetry. It preserves the covariance between their outcomes and typically reduces the variance of the policy contrast; it does not remove block-to-block variation.

For median delay, target episode identity/onset blocks are shared across policies, and entire onset blocks are resampled. Blocks without an onset remain in the sampling universe, avoiding conditioning only on attack-containing blocks.

### 16.3 Confidence intervals

The study uses percentile bootstrap intervals with seed 1729:

- two-sided 95% interval: 2.5th to 97.5th percentiles;
- one-sided upper 95% endpoint: 95th percentile;
- one-sided lower 95% endpoint: 5th percentile.

The paired block bootstrap respects more dependence than row resampling, but assumes the chosen blocks are approximately exchangeable sampling units. Deterministic time blocks from one testbed may not satisfy that assumption. Trace blocks are analytical surrogates, not verified independent campaigns. A coarser one-hour grouping reduces the test set to 14 clusters and tests sensitivity to the clustering choice; fewer/larger clusters are not automatically conservative.

### 16.4 Noninferiority gates

The frozen held-out declaration requires all four gates to pass.

For malicious `ALLOW` difference $\Delta_E=E_A^P-E_A^R$, lower is better. With a +2 percentage-point margin:

\[
U_{0.95}(\Delta_E)\le0.02.
\]

For coverage difference $\Delta_K=K^P-K^R$, higher is better. With a −5-point margin:

\[
L_{0.95}(\Delta_K)\ge-0.05.
\]

For median capped delay difference $\Delta_D$, lower is better. With a +1-second margin:

\[
U_{0.95}(\Delta_D)\le1\text{ s}.
\]

For transition-rate ratio $R_C=C^P/C^R$, a 25% reduction requires:

\[
U_{0.95}(R_C)\le0.75.
\]

This is a conjunctive claim: one failed gate defeats the overall success declaration. “Noninferior” does not mean identical; it means the data rule out degradation worse than the predeclared margin at the selected endpoint.

### 16.5 Friction uncertainty is a separate question

The pairwise gates compare proposed with reference. Under the chosen block-resampling model and source population, support for a below-1% benign-friction claim would require a one-policy upper bound:

\[
U_{0.95}(F_P)\le0.01.
\]

The held-out point is 0.821%, but the one-sided upper bound is 1.069%. Therefore the point estimate meets 1%, while the experiment does **not** certify the bound even for that narrower resampling interpretation. A passing bound still would not be a universal guarantee across sites, days, or deployments.

### 16.6 Why mean delay and not-isolated exposure remain mandatory

Mean capped delay and $E_{NI}$ were defined as descriptive safeguards rather than the four formal gates. That does not make them disposable. They reveal behavior hidden by median delay and `ALLOW` exposure. A principled paper reports them even when they weaken the story.

## 17. Reading the frozen results

### 17.1 Aggregate held-out comparison

| Metric | Proposed | Stateless | Contrast |
|---|---:|---:|---:|
| Benign friction | 0.821% | 0.643% | +0.179 pp |
| Benign `ISOLATE` time | 0.253% | 0.643% | −0.389 pp |
| Malicious `ALLOW` time | 22.330% | 26.812% | **−4.482 pp** |
| Malicious not-`ISOLATE` time | 29.571% | 26.812% | **+2.759 pp** |
| Churn | 2.179 | 5.114 | **57.39% lower** |
| Episode coverage | 82.979% | 85.771% | −2.793 pp |
| Median capped delay | 1.000 s | 1.000 s | 0 s |
| Mean capped delay | 5.703 s | 4.152 s | **+1.552 s** |

Churn is shown in transitions per observed RNTI-minute.

### 17.2 Formal gate results

| Gate | Point estimate and 95% interval | Relevant one-sided endpoint | Result |
|---|---|---:|---|
| `ALLOW` exposure difference | −4.482 pp [−5.095, −3.887] | upper −3.986 pp | Pass |
| Coverage difference | −2.793 pp [−3.964, −1.693] | lower −3.778 pp | Pass |
| Median capped delay difference | 0 s [0, 0] | upper 0 s | Pass |
| Churn ratio | 0.426 [0.404, 0.452] | upper 0.447 | Pass |

All four frozen pairwise gates pass. Their margins were predeclared; as disclosed in Section 15.7, the primary episode alignment was documented after action replay but before bootstrap inference, with zero-gap episodes retained as sensitivity.

### 17.3 Correct verbal interpretation

The stateful policy retains `RESTRICT`/`ISOLATE` across transient score declines. That reduces repeated crossings and moves some malicious time from `ALLOW` into `RESTRICT`. Therefore:

- it is much more stable;
- it leaves less attack time fully allowed;
- it does **not** isolate attacks more completely in every sense;
- it misses or delays some episodes; and
- its worse mean delay shows a harmful tail despite an unchanged median.

The correct sentence is:

> “At the frozen chronological operating point, the proposed policy reduced malicious fully allowed time and action churn while remaining within predeclared exposure, coverage, and median-delay margins.”

The incorrect sentence is:

> “The proposed method detects and blocks attacks faster and more accurately.”

### 17.4 Per-attack results

| Attack | Proposed/reference `ALLOW` exposure | Exposure difference | Coverage proposed/reference | Mean delay proposed/reference |
|---|---:|---:|---:|---:|
| Port scan | 2.67% / 10.65% | −7.98 pp | 92.34% / 87.10% | 1.42 / 1.73 s |
| DDoS/Ripper | 2.10% / 10.35% | −8.26 pp | 89.02% / 89.02% | 1.54 / 1.51 s |
| DoS/Hulk | 1.80% / 5.44% | −3.64 pp | 91.10% / 86.99% | 1.02 / 1.00 s |
| **Slowloris** | **78.02% / 76.79%** | **+1.23 pp** | **66.91% / 73.90%** | **13.35 / 9.14 s** |

Slowloris coverage difference is −6.99 pp and mean-delay difference is +4.21 s. Its exposure interval crosses zero, while coverage and delay clearly worsen. The aggregate result must never hide this.

These per-attack intervals are mandatory descriptive heterogeneity analyses, not a multiplicity-adjusted family of four independent confirmatory discoveries. Slowloris is a documented failure boundary; the other rows support bounded attack-specific descriptions.

### 17.5 Seen and unseen RNTIs

For 274 unseen numeric RNTIs:

- `ALLOW` exposure difference remains favorable at −4.18 pp;
- churn ratio remains favorable at about 0.446;
- proposed friction rises to 1.289%, with one-sided upper bound 1.727%.

For 470 seen numeric values, exposure/churn also improve and friction is lower. Because RNTI was never a predictor, the comparison measures distributional transfer to numerically novel state keys rather than direct ID memorization. The benign budget does not transfer to the unseen-value stratum.

### 17.6 Alternative scorer

With HGB risk scores, strict tune-only selection chooses a **5-second EWMA**, not the asymmetric family. On held-out replay:

- `ALLOW` exposure difference = −4.60 pp;
- coverage difference = −2.93 pp;
- churn ratio = 0.245;
- median-delay difference = 0 s;
- all four pairwise gates pass;
- benign friction = 1.186%, upper bound 1.890%.

This supports the formulation/protocol more than the particular state machine. It also shows that a higher-AUROC scorer does not guarantee budget transfer.

The separately examined asymmetric HGB candidate fails the coverage gate; do not conflate it with the HGB-selected EWMA result.

### 17.7 Timestamp sensitivity

At raw /`1000000`:

- only 64,294 one-second epochs and 29 blocks remain under the fixed rules;
- test AUROC falls to 0.930;
- no 1%-budget asymmetric proposal is feasible;
- the matched EWMA alternative reduces churn by only 14.5%, below the locked 25% criterion;
- no held-out controller claim is made for this branch.

The /`100000` scale is much better supported by epoch/cadence evidence, but this sensitivity proves that confirmed timestamp provenance remains publication-critical.

### 17.8 Lease and episode sensitivity

Changing the label-blind lease inactivity timeout from 5 to 300 seconds preserves the favorable direction of exposure and churn, although episode counts and coverage vary. Strict zero-gap episode construction produces many more short episodes and lower absolute coverage; the proposed-reference difference stays within the frozen margin and both median delays remain one second.

### 17.9 Runtime result

On one pinned Xeon 6246R logical CPU in an in-memory Python benchmark:

- batch logistic scoring: about 0.284 μs/report;
- proposed sequential update: about 3.283 μs/report;
- combined descriptive estimate: about 3.566 μs/report.

These demonstrate low algorithmic compute overhead in this harness. They exclude file I/O, deserialization, telemetry transport, RIC scheduling, xApp framework overhead, actuation, and tail latency. They are not a deployed near-RT RIC benchmark.

## 18. What the paper proves—and does not prove

### 18.1 Supported conclusions

The evidence supports these carefully bounded statements:

- Public RAN KPI risk scores can create substantial action churn under direct thresholding.
- A tune-only friction/security-constrained search can select a sequential policy with a better held-out security–stability trade-off.
- On the frozen logistic test replay, the selected policy reduces fully allowed attack time and churn at matched tuning friction while passing four predefined pairwise gates.
- Favorable exposure and churn contrasts persist for unseen numeric RNTIs; lower `ALLOW` exposure persists for port scan, DDoS/Ripper, and DoS/Hulk. Other outcomes do not uniformly improve.
- Results remain directionally stable across several lease/episode/group choices.
- Slowloris, unseen-RNTI friction, timestamp scaling, and model-dependent mechanism choice expose meaningful limits.

### 18.2 Unsupported conclusions

The evidence does not establish:

- authentication or MFA efficacy;
- durable user/device identity generalization;
- authorization to a named resource;
- causal attack blocking/prevention;
- scheduler or bearer enforcement;
- measured QoS/user harm from `RESTRICT` or `ISOLATE`;
- attacker adaptation after a real action;
- production 5G/6G scale or near-RT latency;
- mobility-induced churn or mobility generalization;
- zero-day attack detection;
- standardized E2SM-KPM collection or E2SM-RC actuation;
- attack-agnostic superiority.

### 18.3 Six distinct failure types

Keep these separate when diagnosing a result:

1. **Detection failure:** the scorer does not separate an attack class well enough for useful action.
2. **Controller failure:** smoothing/persistence delays otherwise actionable evidence. Slowloris's poor policy outcome is consistent with weaker score separation and/or adverse controller dynamics; this study does not causally decompose those mechanisms.
3. **Budget-transfer failure:** a tune-time 1% point becomes more costly later or in a new stratum.
4. **Mechanism instability:** another scorer selects a different policy family.
5. **Provenance failure:** time encoding or grouping uncertainty changes feasibility.
6. **Semantic/causal overclaim:** replayed RNTI actions are described as identity access control or real prevention.

High AUROC addresses only part of the first category.

## 19. Code and artifact walkthrough

Run commands from:

```text
/nobackup/ashukuma/xr/o_ran_publication
```

### 19.1 End-to-end pipeline map

```text
dtst.csv
  ↓ schema, ontology, duplicate, identifier, chronology audit
corrected chronological reports
  ↓ label-blind block and RNTI-lease segmentation
causal 1-second epochs + four whole-block partitions
  ↓ train-only imputation/scaling/logistic fit
raw scores
  ↓ separate Platt calibration
calibrated score stream
  ↓ tune-only budget matching and strict selection
locked controller specifications
  ↓ test-only causal replay with one-epoch lag
action, episode, stratum, and block contribution artifacts
  ↓ paired block bootstrap and sensitivities
tables, figures, readiness report, and manuscript
```

### 19.2 Module map

| Stage | Main module | What to learn from it |
|---|---|---|
| Source validation/chronology | [`data.py`](../src/oran/data.py) | Schema contract, block/lease/epoch construction |
| Alternate grouping audit | [`capture_audit.py`](../src/oran/capture_audit.py) | Why groups are label-blind surrogates |
| Split manifest | [`manifest.py`](../src/oran/manifest.py) | Complete-block chronological assignment |
| Pipeline orchestration | [`experiment.py`](../src/oran/experiment.py) | Preparation, fitting, score persistence |
| Risk model | [`model.py`](../src/oran/model.py) | Feature firewall, train-only preprocessing, calibration |
| Controller | [`controller.py`](../src/oran/controller.py) | Stateful update logic without labels |
| Metrics | [`metrics.py`](../src/oran/metrics.py) | Sample-and-hold durations and time-weighted outcomes |
| Policy templates/search | [`policy_search.py`](../src/oran/policy_search.py), [`matched_search.py`](../src/oran/matched_search.py) | Structure grid and threshold calibration |
| Strict selection | [`selection.py`](../src/oran/selection.py), [`strict_selection.py`](../src/oran/strict_selection.py) | Exact stateless reference and feasibility gates |
| Locked replay | [`evaluation.py`](../src/oran/evaluation.py), [`confirmatory.py`](../src/oran/confirmatory.py) | Input projection, one-epoch lag, label join afterward |
| Uncertainty | [`bootstrap.py`](../src/oran/bootstrap.py), [`inference.py`](../src/oran/inference.py) | Paired block resampling and gate decisions |
| Sensitivities | [`sensitivity.py`](../src/oran/sensitivity.py) | Lease-timeout replay |
| Runtime | [`benchmark.py`](../src/oran/benchmark.py) | Correctness-gated offline timing |
| Publication outputs | [`reporting.py`](../src/oran/reporting.py) | Deterministic tables/figures from locked artifacts |
| Integrity audit | [`repro_audit.py`](../src/oran/repro_audit.py) | 20 fail-closed artifact/protocol checks |

### 19.3 Authoritative artifacts

| Question | Artifact |
|---|---|
| What exact data/splits were used? | [`data_summary_v1.json`](../artifacts/audits/data_summary_v1.json), [`split_manifest_v1.json`](../artifacts/manifests/split_manifest_v1.json) |
| What is the epoch table? | [`epochs_1s_v1.parquet`](../artifacts/epochs/epochs_1s_v1.parquet) |
| What did the risk model score? | [`scores_logistic_seed1729_v1.parquet`](../artifacts/results/scores_logistic_seed1729_v1.parquet) |
| What are the frozen scientific choices? | [`study_protocol_v1_locked.json`](../configs/study_protocol_v1_locked.json) |
| What policies were replayed? | [`candidate_lock_v2.json`](../artifacts/confirmatory/candidate_lock_v2.json) |
| What was every effective action? | [`action_trace_v2.parquet`](../artifacts/confirmatory/action_trace_v2.parquet) |
| What are aggregate held-out points? | [`aggregate_metrics_v2.parquet`](../artifacts/confirmatory/aggregate_metrics_v2.parquet) |
| What are episode outcomes? | [`attack_episodes_v2.parquet`](../artifacts/confirmatory/attack_episodes_v2.parquet) |
| What are formal intervals/gates and inference-assembled sensitivities? | [`inference_report_v3.json`](../artifacts/confirmatory/inference_report_v3.json) |
| What are the lease-timeout sensitivity outputs? | [`lease_timeout_action_metrics_v1.parquet`](../artifacts/sensitivities/lease_timeout_action_metrics_v1.parquet), [`lease_timeout_episode_metrics_v1.parquet`](../artifacts/sensitivities/lease_timeout_episode_metrics_v1.parquet), [`lease_timeout_manifest_v1.json`](../artifacts/sensitivities/lease_timeout_manifest_v1.json) |
| What publication tables/figures were derived? | [`tables`](../reports/tables), [`figures`](../reports/figures), [`result_digest.json`](../reports/result_digest.json) |
| Did integrity checks pass? | [`reproducibility_audit.json`](../reports/reproducibility_audit.json) |
| What is the complete readiness verdict? | [`publication_readiness_report.md`](../reports/publication_readiness_report.md) |
| What is the current paper? | [`main.pdf`](../manuscript/main.pdf), [`main.tex`](../manuscript/main.tex) |

`controller_matched_selected_tune_v1.parquet` is an intermediate summary. The strict-selection table and locked protocol are authoritative.

### 19.4 Recommended code-reading order

Read the tests with the implementation; they provide small executable examples:

1. `tests/test_data.py`
2. `tests/test_model.py`
3. `tests/test_controller.py`
4. `tests/test_selection.py`
5. `tests/test_evaluation.py`
6. `tests/test_bootstrap.py`
7. `tests/test_repro_audit.py`

### 19.5 Hands-on exercise 1: inspect causal epochs

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import polars as pl

e = pl.read_parquet("artifacts/epochs/epochs_1s_v1.parquet")
print(e.shape)
print(e.group_by("split").len().sort("split"))
print(e.select(
    "trace_block_id", "rnti_lease_id", "mac_rnti",
    "epoch_start_s", "decision_time_s", "samples_in_epoch",
    "labels_in_epoch", "is_attack_epoch"
).head(12))
PY
```

Answer:

- Why can `samples_in_epoch` exceed one?
- Why is it excluded from the model?
- Why are empty seconds absent?
- Why is `decision_time_s` the right edge rather than first report time?

### 19.6 Hands-on exercise 2: verify the feature firewall

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from oran.experiment import PRIMARY_FEATURES
from oran.model import FORBIDDEN_FEATURE_COLUMNS

print("feature count:", len(PRIMARY_FEATURES))
print(*PRIMARY_FEATURES, sep="\n")
print("forbidden overlap:", set(PRIMARY_FEATURES) & FORBIDDEN_FEATURE_COLUMNS)
PY
```

Expected overlap: an empty set.

### 19.7 Hands-on exercise 3: follow one real lease

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import polars as pl

a = pl.read_parquet("artifacts/confirmatory/action_trace_v2.parquet")
p = a.filter(pl.col("candidate") == "proposed-template-047-B0.01")
lease = (
    p.group_by("rnti_lease_id")
     .agg(pl.col("effective_transitioned").sum().alias("transitions"))
     .sort("transitions", descending=True)
     ["rnti_lease_id"][0]
)
print(
    p.filter(pl.col("rnti_lease_id") == lease)
     .select(
         "decision_time_s", "risk_score", "evidence_score",
         "decision_state", "effective_state",
         "effective_transitioned", "is_attack_epoch", "attack_types"
     )
     .head(80)
)
PY
```

Find the one-epoch difference between `decision_state` and `effective_state`. Observe that label changes do not reset the controller.

### 19.8 Hands-on exercise 4: inspect strict-selection failures

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import polars as pl

s = pl.read_parquet(
    "artifacts/results/controller_strict_selection_tune_v1.parquet"
)
print(
    s.filter(pl.col("friction_budget") == 0.01)
     .select(
         "family", "status", "candidate", "benign_friction",
         "malicious_allow", "malicious_not_isolated",
         "episode_coverage", "median_capped_delay_s",
         "transitions_per_minute", "transition_reduction"
     )
)
PY
```

Explain why the N-report baseline loses too much coverage and why symmetric hysteresis does not clear the required churn reduction.

### 19.9 Hands-on exercise 5: reproduce a smaller bootstrap

```bash
PYTHONPATH=src .venv/bin/python -m oran.inference \
  --confirmatory-root artifacts/confirmatory \
  --cluster-mapping artifacts/block_cluster_candidates.csv \
  --output-root /tmp/oran_inference_demo \
  --hgb-root artifacts/hgb_sensitivity \
  --alternate-timebase-root artifacts/timebase_1e6 \
  --replicates 500 \
  --seed 1729
```

The point estimates should match the frozen report. Interval endpoints may differ because this exercise uses 500 rather than 5,000 replicates. It writes only to `/tmp` and does not change frozen inference.

### 19.10 Focused tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_data.py \
  tests/test_controller.py \
  tests/test_selection.py \
  tests/test_evaluation.py \
  tests/test_bootstrap.py
```

The full frozen suite currently contains 96 passing tests. Tests verify behavior; they do not substitute for replaying the 794 MB source pipeline.

### 19.11 Full reproduction

The complete, ordered commands for environment creation, source hashing, preparation, model fitting, matched search, strict selection, both confirmatory episode definitions, HGB, alternate timebase, inference, lease sensitivity, benchmark, reporting, integrity audit, tests, and manuscript compilation are maintained in the project [README regeneration section](../README.md#exact-regeneration). Use those commands rather than reconstructing a workflow from individual modules.

Treat the locked protocol JSON as a scientific input. It must not be regenerated or silently edited after viewing held-out outcomes.

## 20. From an offline intent to a real O-RAN implementation

The manuscript uses abstract actions because the trace contains no action execution. A future deployment would need to define them concretely.

### 20.1 Possible action meanings

Depending on supported node functions and operator policy:

- `ALLOW` might preserve the current bearer/scheduling treatment.
- `RESTRICT` might request rate/resource limitation, lower scheduling priority, a quarantine QoS treatment, or traffic steering/filtering.
- `ISOLATE` might stop scheduling, block matching traffic, suspend/release a bearer, or release an attached radio context.

There is no universal standards command literally named “RESTRICT RNTI” or “ISOLATE RNTI.” The implementation must map policy semantics to capabilities that the target E2 node actually exposes.

Bearer/RRC release removes only the current context; it is not durable denial. A UE may reconnect and receive another RNTI, and stopping scheduling can itself provoke radio-link failure/reconnection. Durable exclusion requires a higher-level admission/identity policy or another enforcement point.

### 20.2 Minimum deployment architecture

```text
E2 node / telemetry source
      │ reports + RNTI/context
      ▼
Near-RT RIC xApp
  ├─ feature state / calibration monitoring
  ├─ risk scorer
  ├─ per-RNTI lifecycle-aware sequential policy
  ├─ conflict/safety guard
  └─ action request + acknowledgement tracking
      │
      ▼ supported control interface/service model
O-CU/O-DU enforcement function
      │
      ├─ actual scheduling/bearer/filter action
      └─ measured QoS, recovery, reconnect, and attack response
```

The deployment would also need:

1. authoritative RNTI allocation/change/release events rather than a timeout proxy;
2. correlation to trusted subscriber/device/session context where policy requires identity;
3. supported E2 service-model/function discovery;
4. action acknowledgement, failure, timeout, and retry semantics;
5. arbitration with other xApps and operator policy;
6. safe fallback on missing/corrupt telemetry;
7. score/calibration drift monitoring;
8. action-specific QoS and business cost measurement;
9. attacker-response and reconnect-loop handling; and
10. prospective evaluation on independent days/sites.

### 20.3 Timescale reality

O-RAN commonly associates near-RT control with roughly 10 ms to 1 s and non-RT control with periods above 1 s. The study makes decisions only for nonempty observed one-second bins and adds a one-epoch effective-action lag. Its score cadence is near the nominal boundary, but it demonstrates neither a guaranteed periodic one-second loop nor end-to-end near-RT compliance. It is near-RT-motivated; deployment classification requires end-to-end measurement.

In 5G NR, OFDM numerology changes slot duration; a 10 ms radio frame contains 1 ms subframes and a numerology-dependent number of slots, while a resource block spans 12 subcarriers. Therefore, never describe a one-second paper epoch as one PHY scheduling interval. It summarizes many radio events.

Likewise, the measured 3.3 μs Python controller update is only the computation after data are in memory. End-to-end telemetry-to-enforcement latency must include collection, encoding, transport, platform scheduling, conflict handling, node processing, and acknowledgement.

## 21. Reviewer and presentation questions

Use these as an oral examination. A strong answer is short, specific, and bounded.

### Q1. What is the paper's single research question?

Can a causal score-to-action policy selected under a fixed benign-friction budget reduce RNTI-level action churn without exceeding predeclared containment and delay degradation margins on chronological held-out RAN telemetry?

### Q2. Why is this not just another IDS classifier paper?

The logistic classifier is a fixed nuisance component. The primary objects are sequential actions, benign action occupancy, malicious exposure, full-isolation safeguard, action churn, episode coverage, delay, exact budget matching, and chronological group-held-out inference.

### Q3. Why is an RNTI a defensible unit at all?

Within an inferred lease, it is an addressable temporary radio-context grouping on which a RAN treatment could be requested. The CSV does not verify the underlying lifecycle, so it is useful for lease-level replay but not durable identity or resource authorization.

### Q4. What exactly is a lease?

An analytical lifecycle proxy: one numeric RNTI within a label-blind trace block, split after more than 30 seconds of inactivity. It prevents state from crossing obvious reuse, but is not an authoritative allocation/release record.

### Q5. What prevents label leakage into the controller?

The feature allowlist rejects target/context fields; model preprocessing fits on train only; block/lease segmentation is label-blind; replay explicitly projects to score/time/RNTI/lease; the controller API has no label parameter; targets are joined only after actions are generated.

### Q6. Why four splits instead of train/test?

The base classifier, probability calibration, controller selection, and final policy evaluation are separate learning problems. Reusing data across them biases thresholds and uncertainty.

### Q7. Why not use a random split?

Adjacent telemetry and reused RNTIs/scenarios are highly correlated. A random split tests interpolation among nearly repeated capture conditions and can expose metadata fingerprints, not forward temporal transfer.

### Q8. Why match friction?

Aggressiveness changes both security and disruption. Matching benign intervention time makes stability/security comparisons operationally meaningful rather than comparing different cost levels.

### Q9. Why insist on minimum budget utilization?

A policy that remains almost always `ALLOW` would have low churn simply because it rarely acts. Requiring about 95% use prevents that degenerate solution.

### Q10. Is hysteresis the novelty?

No. Hysteresis and EWMA are standard. Novelty lies in constrained policy selection, matched-budget reference, causal/grouped protocol, and empirical security–friction–stability frontier with explicit failures.

### Q11. Why does the proposed controller reduce `ALLOW` exposure but worsen not-`ISOLATE` exposure?

It uses a genuine middle state. Persistence retains `RESTRICT` across risk dips, reducing full `ALLOW`; its higher isolation boundary means some time that the binary reference isolates remains only restricted.

### Q12. How can median delay be unchanged while mean delay worsens?

Most episodes are reached quickly, preserving the median, while a minority—especially Slowloris—are delayed or missed and receive larger capped values, increasing the mean/tail.

### Q13. Why is Slowloris difficult?

It holds application connections with low-rate traffic, so radio/MAC load patterns may resemble benign activity. The data show the failure; the physical explanation is plausible but not causally proven by this analysis.

### Q14. What does “unseen RNTI” demonstrate?

It measures whether favorable exposure/churn contrasts transfer to numerically novel temporary state keys. RNTI was excluded from model inputs, so this is not a direct ID-memorization test. It does not demonstrate new-subscriber or new-device identity generalization, and its friction overrun shows incomplete transfer.

### Q15. Why bootstrap blocks rather than epochs?

Epochs within a trace block are correlated. Resampling complete blocks preserves within-block structure better and avoids pretending 101,561 epochs are independent experiments.

### Q16. Why is the bootstrap paired?

Both policies act on the same block. Using the same sampled blocks preserves their covariance and usually estimates the contrast more precisely; it does not remove block-to-block variation.

### Q17. Why does point friction below 1% not prove the budget holds?

The point is one observed test realization. Its one-sided block-bootstrap upper bound is 1.069%, above 1%, so uncertainty does not support the bound even under the chosen block/source-population model—much less across other sites or days.

### Q18. What does the HGB sensitivity mean?

The framework can select a different temporal mechanism and still pass the pairwise gates; it does not prove the asymmetric FSM is universal. HGB's friction overrun also shows that better AUROC does not ensure cost calibration transfers.

### Q19. Is this a deployed zero-trust architecture?

No. It is zero-trust-inspired continuous, graduated network containment in offline replay. It lacks durable identity, protected-resource policy, actual enforcement, and causal outcome measurement.

### Q20. What single next experiment would strengthen it most?

A prospective xApp/RIC testbed replay with authoritative RNTI lifecycle events and concretely implemented `RESTRICT`/`ISOLATE`, measuring action acknowledgement, attack traffic after intervention, and benign QoS. If the goal becomes genuine identity/adaptive access, use authentication/session data such as LANL as primary and treat OpenIreland as complementary RAN evidence.

## 22. Common misconceptions

1. **O-RAN is not synonymous with 5G NR.** It is an open/disaggregated RAN architecture and can relate to LTE and NR.
2. **Open RAN does not mean all code is open source.** The key idea is open/interoperable functional architecture and interfaces.
3. **CU/DU and the RICs are logical nodes/functions, but O-RU is a physical node.** Functions/nodes may be bundled or colocated; that does not make O-RU virtual.
4. **Near-RT RIC does not replace slot/symbol-level DU scheduling.** Its control loop is slower and operates through exposed functions.
5. **E2, E2AP, E2SM-KPM, and E2SM-RC are different things.** E2 is the interface, E2AP provides generic procedures, and service models define measurement/control semantics.
6. **The CSV is not proven to be standard KPM encoding.** Its authors used a custom testbed/RIC collection path.
7. **A KPI is not attack ground truth.** Traffic and channel conditions jointly shape KPIs.
8. **CQI, MCS, SINR, and RSSI are not interchangeable.** They reflect different measurements/decisions.
9. **High RSSI does not necessarily imply a clean link.** It can include interference.
10. **A TTI is not universally one millisecond across all radio systems/configurations.** Avoid LTE shorthand when discussing NR.
11. **RNTI is not IMSI/SUPI, a person, or a permanent device.** It is a temporary radio identifier family.
12. **Unseen numeric RNTI is not unseen identity.** Values can be temporary and reused.
13. **High AUROC is not a stable access policy.** It ignores threshold cost and action dynamics.
14. **Friction is not churn.** One continuous 60-second restriction and sixty one-second pulses can have similar friction but radically different churn.
15. **Lower malicious `ALLOW` is not lower not-`ISOLATE`.** `RESTRICT` is a middle state.
16. **Equal median delay is not equal delay distribution.** Mean/tail and missed episodes matter.
17. **A 0.821% observed point is not a guaranteed 1% bound.** Its uncertainty interval crosses 1%.
18. **A trace block is not a known independent capture/campaign.** It is a label-blind temporal surrogate.
19. **Mobility diversity in the full CSV does not imply mobility generalization.** The viable held-out test is effectively car.
20. **Offline replay is not causal enforcement evidence.** Real actions would change subsequent traffic, telemetry, and attacker behavior.

## 23. Glossary

| Term | Plain-language meaning in this guide |
|---|---|
| 5GC | 5G Core network |
| A1 | Interface carrying policy/guidance/enrichment between non-RT and near-RT RIC functions |
| ALLOW | Preserve normal treatment in the abstract replay |
| AUPRC | Area under the precision–recall curve |
| AUROC | Area under the receiver-operating characteristic curve |
| BSR | Buffer Status Report for pending uplink data |
| Calibration | Align score values with observed outcome frequency/scale on separate data |
| C-RNTI | Cell-scoped temporary identifier associated with a connected UE radio context |
| CQI | Channel Quality Indicator used for link adaptation |
| CU / O-CU | Central Unit / O-RAN Central Unit |
| Downlink | Network to UE direction |
| DU / O-DU | Distributed Unit / O-RAN Distributed Unit |
| E2 | Logical interface between near-RT RIC and E2 nodes |
| E2AP | Generic E2 application protocol procedures |
| E2SM-KPM | E2 service model for performance measurements |
| E2SM-RC | E2 service model for RAN control |
| Epoch | One observed one-second RNTI-lease decision bin in this project |
| Friction | Fraction of benign observed RNTI-time in `RESTRICT` or `ISOLATE` |
| gNB | 5G NR base station logical node |
| HARQ | Fast hybrid retransmission mechanism at PHY/MAC boundary |
| HGB | Histogram gradient-boosting classifier used as sensitivity |
| Hysteresis | Different entry and recovery thresholds to prevent chatter |
| ISOLATE | Strongest abstract containment intent in the replay |
| Lease | Label-blind inferred lifecycle for one numeric RNTI in a trace block |
| MAC | Medium Access Control layer for scheduling/multiplexing and related reports |
| MCS | Modulation and Coding Scheme index |
| Near-RT RIC | Separate O-RAN control function hosting xApps, roughly 10 ms–1 s category |
| NI | Noise/interference indicator |
| Non-RT RIC | Policy/analytics/model function within the SMO in the >1 s category, hosting rApps |
| O1 | Management interface for managed O-RAN functions |
| O2 | SMO interface toward O-Cloud infrastructure |
| O-RAN | O-RAN Alliance architecture/specification ecosystem |
| Open RAN | Broader concept of open/disaggregated/interoperable RAN |
| PDCP | Packet Data Convergence Protocol |
| PEP | Policy Enforcement Point in zero-trust terminology; absent from this replay |
| PHR | Power Headroom Report |
| PHY | Physical radio layer |
| PIP | Policy Information Point in zero-trust terminology |
| Platt scaling | Logistic calibration of a model's raw score |
| PDP | Policy Decision Point in zero-trust terminology |
| PUCCH | Physical Uplink Control Channel |
| PUSCH | Physical Uplink Shared Channel |
| RAN | Radio Access Network |
| RESTRICT | Intermediate abstract reduced-service/quarantine intent |
| RIC | RAN Intelligent Controller |
| RLC | Radio Link Control layer |
| RNTI | Radio Network Temporary Identifier family |
| RSSI | Received Signal Strength Indicator; includes more than desired signal |
| RU / O-RU | Radio Unit / O-RAN physical Radio Unit terminating Open Fronthaul |
| SDAP | Service Data Adaptation Protocol in the 5G user plane |
| SINR | Signal-to-Interference-plus-Noise Ratio |
| SMO | Service Management and Orchestration framework/function |
| TTI | Transmission-time scheduling interval/statistic; precise meaning is system-specific |
| UE | User Equipment |
| Uplink | UE to network direction |
| xApp | Application running on a near-RT RIC |
| rApp | Application associated with non-RT RIC/SMO functions |
| ZTA | Zero-Trust Architecture |

## 24. Primary reading list

Read standards selectively; they are references, not novels.

### Cellular and NG-RAN

1. [3GPP/ETSI TS 38.201: NR physical-layer general description](https://www.etsi.org/deliver/etsi_ts/138200_138299/138201/18.00.00_60/ts_138201v180000p.pdf)—short physical-layer orientation.
2. [3GPP/ETSI TS 38.300: NR and NG-RAN overall description](https://www.etsi.org/deliver/etsi_ts/138300_138399/138300/18.09.00_60/ts_138300v180900p.pdf)—architecture, protocols, radio identities.
3. [3GPP/ETSI TS 38.401: NG-RAN architecture](https://www.etsi.org/deliver/etsi_ts/138400_138499/138401/18.07.00_60/ts_138401v180700p.pdf)—CU/DU and NG-RAN interfaces.
4. [3GPP/ETSI TS 38.321: NR MAC](https://www.etsi.org/deliver/etsi_ts/138300_138399/138321/18.05.00_60/ts_138321v180500p.pdf)—RNTI/MAC, BSR, PHR, scheduling-related procedures.

### O-RAN

5. [ETSI TS 103 982: O-RAN architecture](https://www.etsi.org/deliver/etsi_ts/103900_103999/103982/08.00.00_60/ts_103982v080000p.pdf)—SMO, RICs, O-CU/O-DU/O-RU, interfaces.
6. [ETSI TS 104 038: E2 general aspects](https://www.etsi.org/deliver/etsi_ts/104000_104099/104038/04.01.00_60/ts_104038v040100p.pdf)—E2 architecture, principles, and roles.
7. [ETSI TS 104 039: E2AP](https://www.etsi.org/deliver/etsi_ts/104000_104099/104039/04.00.00_60/ts_104039v040000p.pdf)—generic subscription/indication/control protocol.
8. [ETSI TS 104 040: E2 service-model framework](https://www.etsi.org/deliver/etsi_ts/104000_104099/104040/04.00.00_60/ts_104040v040000p.pdf)—service-model framework, common elements, and service-model list/context.

### Dataset and closest experimental context

9. [Official OpenIreland dataset record](https://data.mendeley.com/datasets/t2rzh9y4mp/1)—source, file, license, DOI.
10. [Dataset description paper](https://doi.org/10.1016/j.comnet.2024.110710)—collection design and fields.
11. [Xavier et al., early O-RAN attack detection](https://arxiv.org/pdf/2302.01864)—testbed/controller path, KPI/attack context.

### Zero trust and methodology

12. [NIST SP 800-207: Zero Trust Architecture](https://csrc.nist.gov/pubs/sp/800/207/final)—principles and architecture.
13. [NIST SP 1800-35 implementation architecture](https://pages.nist.gov/zero-trust-architecture/VolumeB/architecture.html)—continue/limit/revoke session treatment.
14. [Arp et al., security-ML methodology](https://www.usenix.org/conference/usenixsecurity22/presentation/arp)—leakage, realism, and evaluation pitfalls.

### Project-specific final reading

15. [`manuscript/main.pdf`](../manuscript/main.pdf)—read after Sections 1–18 of this guide.
16. [`reports/publication_readiness_report.md`](../reports/publication_readiness_report.md)—read for every caveat and publication condition.
17. [`configs/study_protocol_v1_locked.json`](../configs/study_protocol_v1_locked.json)—read to distinguish predeclared choices from post-hoc interpretation.

## 25. Final readiness checklist

You are ready to explain or coauthor the paper when you can do all of the following without notes:

- draw UE → RAN → core and explain uplink/downlink plus control/user planes;
- distinguish PHY, MAC, RLC, PDCP, SDAP, and RRC;
- place CU, DU, physical O-RU, SMO, both RICs, xApps/rApps, A1, E2, O1, and O2, and distinguish logical from physical nodes;
- explain why E2 is not the same as KPM or RAN Control;
- explain CQI, MCS, SINR, RSSI, BSR, PHR, PUSCH, and PUCCH in plain language;
- explain why RNTI is useful yet not identity;
- reconstruct raw reports → blocks → leases → epochs → four splits;
- derive logistic scoring and explain Platt calibration;
- hand-simulate the selected state machine including one-epoch lag;
- calculate friction, `ALLOW` exposure, not-`ISOLATE` exposure, churn, coverage, and delay;
- explain exact budget matching and every feasibility constraint;
- defend chronological whole-block splitting and paired block bootstrap;
- interpret all four formal gates and the separate friction upper bound;
- state the Slowloris, unseen-RNTI, HGB, timebase, and deployment limitations;
- explain why this is a network-level offline containment study rather than identity authentication or causal prevention; and
- point from every headline claim to its frozen artifact and code path.

If any one of those remains fuzzy, return to the corresponding section before presenting the paper.
