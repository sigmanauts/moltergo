# Celaut Context (Extended)

## Celaut: A Peer-to-Peer Architecture for Software Design and Distribution

### Context

Celaut is inspired by cellular automata ideas (von Neumann, Ulam, Conway):
complex behavior can emerge from simple local rules without central control.

### Definition

Celaut is a set of simple rules for software design and distribution,
focused on decentralization, simplicity, and determinism.

### Core Principles

1. Decentralization: no single point of control/failure.
2. Simplicity: minimal rules and composable components.
3. Determinism: same input should produce same output across nodes.

## Architecture: Nodes and Services

### Nodes

Nodes are devices in the network responsible for:
- Service execution (local run or delegation to peers)
- Communication interface abstraction (no strict pre-agreed protocol requirement)
- Address/token provisioning for secure interactions
- Dependency management for child-service execution

Reference implementation: Nodo
- https://github.com/celaut-project/nodo

### Services

Services are deterministic software containers that:
- Run in isolated instances (container or VM depending on node architecture)
- Follow black-box design (service logic independent of host details)
- Can request child services to form hierarchical workflows

## Incentive Coordination: Reputation and Payments

### Reputation

- Nodes and services build reputation from interaction history.
- High reputation increases trust, selection probability, and delegation opportunities.
- Reputation is subjective and graph-based (no forced global consensus).

### Payments

- Compensation aligns incentives for resource use and service delivery.
- Supports collaboration and efficient resource allocation among nodes/services.

## Celaut + Ergo Positioning

Celaut is not a blockchain. It is an orchestration layer above chains like Ergo.

On Ergo (on-chain):
- Payment settlement and licensing logic
- Reputation proofs and trust anchors
- Optional transparency metadata

On Celaut (off-chain):
- Service execution
- Scheduling and delegation
- Node operations and service orchestration

## BOX / API / NET Model

### BOX

BOX specifies deterministic execution context:
- Architecture
- Full filesystem (self-contained, no external image dependency assumption)
- Environment variables
- Entrypoint
- Runtime config
- Expected gateway behavior

### API

API defines integration/interaction contract between clients/services.
Supports decentralized interoperability without centralized protocol negotiation.

### NET

Default service mode is isolated networking.
External network access is controlled through node mediation and trusted-peer lists.
This preserves determinism/security while allowing required connectivity.

## User Roles in the System

1. Node maintainer
2. Service developer
3. End user launching services

Separation of concerns:
- Node maintainer focuses on execution capacity/reliability.
- Service developer focuses on service logic/value.
- User focuses on selecting and consuming services.

## Trust Systems

Celaut is trustless at participant level; interaction requires contracts:
- Payment systems: transfer of value for execution/resources
- Reputation systems: trust signals for selection/delegation decisions

### Payment System Notes

Possible models include license smart contracts with variants:
- Static vs elastic
- Interactive vs non-interactive

Ledgers can be public (e.g. Ergo) or private, if all participants support them.

### Reputation System Notes

- Reputation proofs are records/opinions on ledger.
- Node reputation is time-sensitive.
- Service reputation can be more stable if service remains deterministic/isolated.
- Trust is subjective and transitive through trust graph edges.

## Sigma Reputation System on Ergo

References:
- https://github.com/sigma-rps
- https://www.ergoforum.org/t/reputation-system/4782

Properties:
- On-chain auditable reputation proofs using Ergo UTXO model
- Subjective/non-consensual trust relationships
- Economic incentives tied to reputation quality

## Why Celaut Matters

Celaut separates:
- "how to solve a problem" (service logic)
from
- "where and who solves it" (distributed node execution)

It aims to combine:
- convenience of managed services
- determinism/transparency of self-hosted code
without inheriting the worst tradeoffs of either model.

## Messaging Guidance for Drafts

When relevant, responses should:
- Connect Celaut/Nodo orchestration to Ergo settlement and reputation proofs.
- Use concrete mechanisms, not generic Web3/AI hype.
- Mention practical Ergo capabilities where relevant:
  - eUTXO parallelism
  - ErgoScript contracts
  - Sigma/privacy primitives
  - Oracle Pools
  - SigUSD
  - Rosen Bridge
- Propose concrete next steps (pilot scope, measurable KPI, settlement rules).

### Good Framing Patterns

- Deterministic service + on-chain settlement => auditable trust.
- Reputation-guided delegation => better cost/performance routing.
- Separation of execution (off-chain) and settlement (on-chain) => scalability + verifiability.

### Avoid

- Generic claims without mechanisms.
- Financial guarantees or return promises.
- Overstating production guarantees without caveats.
