# arc-agi-3-agent

A Claude-powered agent for the [ARC-AGI-3](https://three.arcprize.org/) interactive reasoning benchmark.

The focus of this project is **context management**: ARC-AGI-3 episodes consist of a stream of 64×64 grid frames, and a naive history blows the LLM token budget within a few steps. The agent is designed around frame compression, working memory, and selective replay so the LLM can reason across long episodes without falling off a cliff.

## Status

Pre-alpha — scaffolding. See `docs/` and the issue tracker for the current plan.

## Architecture (planned)

```
+-----------------------------+
|  ARC-AGI-3 API (HTTP)       |
+--------------+--------------+
               |
               v
+-----------------------------+        +-----------------+
|  Environment adapter        |<------>|  Frame encoder  |
|  (ARC-AGI-3-Agents fork)    |        |  (compression)  |
+--------------+--------------+        +-----------------+
               |                                ^
               v                                |
+-----------------------------+        +-----------------+
|  Agent loop                 |------->|  Working memory |
|  - perceive                 |<-------|  / state summary|
|  - decide (Claude)          |        +-----------------+
|  - act                      |
+--------------+--------------+
               |
               v
+-----------------------------+
|  MLflow run logger          |
+-----------------------------+
```

## Quick start

```bash
git clone --recurse-submodules git@github.com:nnaqsoft/arc-agi-3-agent.git
cd arc-agi-3-agent
cp .env.example .env   # then fill in ARC_API_KEY
# Setup steps will be filled in once the submodule is wired and deps are pinned.
```

## Repository layout

```
arc-agi-3-agent/
├── LICENSE                 # MIT
├── README.md
├── .env.example
├── .gitignore
├── docs/                   # design notes, frame format, context strategy
├── agents/                 # our custom agents (Claude-powered)
├── core/                   # context manager, frame encoder, MLflow logger
├── scripts/                # CLI entry points (run_random, run_claude, ...)
├── third_party/
│   └── ARC-AGI-3-Agents/   # git submodule — official framework
└── tests/
```

## References

- ARC Prize docs: https://docs.arcprize.org/
- ARC-AGI-3-Agents (official framework): https://github.com/arcprize/ARC-AGI-3-Agents
- ARC-AGI-3 benchmarking: https://github.com/arcprize/arc-agi-3-benchmarking
- ARC Toolkit overview: https://docs.arcprize.org/toolkit/overview

## License

[MIT](LICENSE) © 2026 Nnaqsoft
