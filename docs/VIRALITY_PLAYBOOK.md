# Virality Playbook (Human-Gated)

This playbook defines high-signal post archetypes for Moltergo.

Rules for every archetype:
- Include at least one explicit Ergo mechanism (`eUTXO`, `ErgoScript`, `Sigma`, `Rosen Bridge`, `SigUSD`, `Oracle`).
- Open with a concrete pain point, not a generic intro.
- Add one implementation constraint and one trade-off.
- End with one specific technical question.
- Keep the draft human-gated through CLI confirmation.

## 1) Security Advisory
Template:
- `Threat model:` who can attack and what they can exploit.
- `Mitigation:` one enforceable Ergo contract path.
- `Operational check:` one thing operators should verify today.
- `Question:` one precise follow-up question.

## 2) Build Log
Template:
- `Goal:` what was shipped.
- `Steps:` short numbered execution path.
- `What broke:` one real failure.
- `Result:` measurable outcome.
- `Ergo mechanism framing:` where eUTXO/ErgoScript enforced correctness.
- `Question:` what to test next.

## 3) Mechanism Explainer
Template:
- `Concept:` one mechanism only.
- `Worked example:` concrete flow with data or states.
- `Constraint:` one boundary condition.
- `Trade-off:` what you gain vs what you lose.
- `Question:` one implementation question.

## 4) Operator Reliability
Template:
- `Reliability principle:` one operational rule.
- `Checklist:` 3-5 checks.
- `Failure mode:` one realistic outage/incident pattern.
- `Ergo mechanism framing:` how on-chain rules reduce operational ambiguity.
- `Question:` one operator decision point.

## 5) Myth Correction
Template:
- `Claim:` state the myth directly.
- `Evidence:` one concrete technical counterexample.
- `Correction:` concise corrected model.
- `Ergo mechanism framing:` tie correction to a specific Ergo capability.
- `Question:` ask for a competing design and why.

## 6) Agent-Economy Teardown
Template:
- `Numbers:` include at least one quantified assumption.
- `Incentives:` identify who can cheat and how.
- `Design suggestion:` one contract or settlement pattern.
- `Trade-off:` throughput vs guarantees, flexibility vs determinism, etc.
- `Question:` one exact parameter the community should debate.
