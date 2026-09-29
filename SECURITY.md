# Security

Please report security issues privately through GitHub's
[security advisories](https://github.com/SerendipityOneInc/instinct/security/advisories/new)
for this repository, not in public issues.

The servers in this repository (`instinct-serve` and the optional adapter in
`serving/vllm/`) are reference servers. They bind to `127.0.0.1` by default and
have no authentication, TLS or rate limiting. Put your own gateway in front of
them before exposing them to a network.
