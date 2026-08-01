# Regulatory context

**Reviewed 2026-08-01.** Every date and document below was verified against a
primary or professional-analysis source on that date; where a figure could not be
verified it is marked as such rather than asserted.

This is not legal advice. It exists so that anyone integrating this library can
see which regulatory lines it is deliberately staying on the safe side of, and
re-check them against their own jurisdiction and intended use.

The operative intended-use statement lives in [`NOTICE`](NOTICE) and is not
duplicated here, so the two cannot drift apart.

---

## Why this file exists at all

DPB processes **streaming physiological signals**. That is the exact category
regulators have been tightening around, so "it's just a library" is not a
sufficient answer.

## United States — FDA

### Clinical Decision Support Software (final guidance, 2026-01-06)

FDA issued an updated final CDS guidance on **6 January 2026**, replacing the
version of 28 September 2022.

The change that matters here is the clarification of **Criterion 1**. A software
function is excluded from the device definition only if, among other things, it
does **not** acquire, process or analyse a *pattern or signal from a signal
acquisition system*. In the 2026 guidance FDA:

- describes signal acquisition systems as those measuring parameters from the
  body for a medical purpose through **continuous, near-continuous or streaming**
  measurement, and
- interprets **"pattern"** as multiple, sequential or repeated measurements —
  while discrete, point-in-time measurements (a routine vital sign at a visit)
  generally do not by themselves constitute a pattern.

**Read that against what DPB does.** Encoding a continuous EEG, ECG, EMG or PPG
stream is squarely "a pattern from a signal acquisition system". A build that
took DPB output and *interpreted clinical meaning from it* would therefore be in
device territory — the CDS exclusion would not rescue it.

DPB's position is drawn deliberately on the other side of that line: it is
measurement and research tooling. It transforms signals, and it does not produce
a diagnosis, a treatment recommendation, or a clinical directive. Outputs named
after clinical rating scales are model estimates, not clinical scores.

**If you build on DPB and your product interprets clinical meaning from streaming
physiologic data, the regulatory obligation is yours, and this library's intended
use does not transfer to you.**

### General Wellness (policy update, 2026-01-23)

FDA updated its General Wellness policy on the same run of CDRH releases in
January 2026, taking a more permissive view of non-invasive estimation of
physiologic parameters for general wellness purposes. General wellness claims
must not reference a specific disease or condition.

### Cuffless blood pressure (draft guidance, 2026)

FDA has a draft guidance covering clinical performance testing for cuffless
non-invasive blood pressure devices.

**DPB does not estimate blood pressure**, and this is an explicit scope boundary
rather than an omission — see [Out of scope](#out-of-scope-by-design) below.

### Contactless camera vitals

510(k) clearances now exist in this space (for example PanopticAI, pulse rate
K240890), which establishes a regulated pathway rather than an open field. DPB
ships no rPPG estimator today; adding one would place that code inside a cleared-
device category and would need its own regulatory analysis.

## European Union — AI Act

The **Digital Omnibus on AI** — the first substantive amendment to the AI Act
since its 2024 adoption — moved through 2026 as follows:

| Milestone | Date |
|---|---|
| Commission proposal published | 2025-11-19 |
| Provisional political agreement (Council + Parliament) | 2026-05-07 |
| Parliament adoption | 2026-06-16 |
| Council final approval | 2026-06-29 |
| Entry into force | July 2026 |

It **defers the high-risk compliance deadlines**:

| Category | Was | Now |
|---|---|---|
| Stand-alone Annex III systems | 2026-08-02 | **2027-12-02** |
| AI embedded in regulated products (Annex I) | — | **2028-08-02** |

The deferral was driven by implementation problems — national competent
authorities not yet designated, harmonised standards not final — not by a change
of policy direction. **Treat it as time, not as reprieve.**

### On the open-source exemption

Do not assume that publishing DPB under a permissive licence exempts a downstream
product from the AI Act. The open-source relief is narrow and is oriented to
general-purpose AI models; it does not carry over to a deployer placing a
high-risk system on the EU market. A downstream integrator's obligations are
theirs, and this repository's licence does not alter them.

## Out of scope by design

DPB does **not**, and is not intended to:

- estimate **blood pressure** (cuffless or otherwise)
- estimate **blood glucose**
- produce a **diagnosis**, a **treatment recommendation**, or any clinical
  directive
- generate a clinical rating-scale **score** of record — outputs bearing scale
  names are model estimates
- serve **normative reference values** fit for clinical interpretation; the
  values bundled with `dpb-norms` are illustrative placeholders

These are boundaries, not gaps in a roadmap. Several of them are the specific
categories that carry the heaviest regulatory load, and staying outside them is
what keeps the library's intended use coherent.

## If you are integrating DPB

1. Write your own intended-use statement. Ours does not transfer.
2. Decide whether your product interprets clinical meaning from streaming
   physiologic data. If it does, assume device territory and get advice.
3. Supply your own cited normative data if you compute percentiles or z-scores.
4. Re-check the dates above. This file was accurate on 2026-08-01 and regulatory
   timelines in this area have moved repeatedly.

## Sources

FDA CDS final guidance (2026-01-06) and General Wellness update (2026-01-23),
via FDA CDRH releases and contemporaneous professional analysis (Arnold & Porter;
FDA Law Blog; Latham & Watkins; Hardian Health). EU Digital Omnibus on AI
timeline and deferred deadlines via Council and Parliament records and
contemporaneous analysis (Gibson Dunn; Freshfields; DLA Piper; Sidley).
