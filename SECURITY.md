# Security and responsible disclosure

Please report vulnerabilities privately to repository maintainers through GitHub's private vulnerability reporting if enabled. Do not file public issues with real videos, identities, video-derived embeddings, access tokens, private camera information or sensitive operations.

## Threat model and non-goals

- This is **local research software**, not a hardened cloud service or production surveillance appliance.
- The Streamlit UI does not authenticate users; bind to loopback and do not expose it publicly.
- Treat uploaded/input videos and generated output as sensitive, even if track IDs are anonymous.
- Model `.pt` checkpoints can be unsafe when loaded from an untrusted source; acquire weights from trusted publishers and verify provenance/checksums. Never load arbitrary internet-uploaded weights.
- Python dependencies and model downloads may require network access; pin versions and scan dependencies before deployment.
- The code does not protect against forged timestamps, chain-of-custody attacks, malicious video inputs, or data leakage from host infrastructure.

For a secure organizational deployment, add authorization, audit logging, encrypted storage, retention/deletion policies, input size validation, malware checks, sandboxed inference, SIEM integration and environment-specific threat modeling.

