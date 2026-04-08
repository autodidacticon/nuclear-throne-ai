#!/bin/bash
# init-firewall.sh — Official Anthropic Claude Code firewall script
# Source: https://github.com/anthropics/claude-code/blob/main/.devcontainer/init-firewall.sh
#
# To add project-specific domains, append them to ALLOWED_DOMAINS below.
# Each entry can be a hostname or IP. IPv6 is handled automatically.

set -e

# Domains always allowed for Claude Code operation
ALLOWED_DOMAINS=(
    # Anthropic API
    "api.anthropic.com"
    "statsig.anthropic.com"
    "sentry.io"

    # npm registry
    "registry.npmjs.org"
    "npm.pkg.github.com"

    # GitHub
    "github.com"
    "raw.githubusercontent.com"
    "objects.githubusercontent.com"
    "api.github.com"
    "uploads.github.com"
    "ghcr.io"
    "pkg-containers.githubusercontent.com"

    # apt / Debian
    "deb.debian.org"
    "security.debian.org"
    "ftp.debian.org"
    "archive.debian.org"

    # Ubuntu
    "archive.ubuntu.com"
    "security.ubuntu.com"
    "ports.ubuntu.com"

    # DNS
    "8.8.8.8"
    "8.8.4.4"
    "1.1.1.1"
    "1.0.0.1"

    # ===== PROJECT-SPECIFIC DOMAINS =====

    # PyPI — pip install gymnasium stable-baselines3 torch imitation wandb etc.
    "pypi.org"
    "files.pythonhosted.org"
    "pythonhosted.org"

    # PyTorch — wheel downloads (CPU builds used inside container)
    "download.pytorch.org"

# Weights & Biases — experiment tracking and training curve logging
    "api.wandb.ai"
    "wandb.ai"
    "files.wandb.ai"

    # GameMaker / YoYo Games — Igor CLI and runtime downloads
    "gms.yoyogames.com"
    "store.yoyogames.com"
    "cdn.yoyogames.com"

    # NodeSource — Node.js 20 apt repository (used during image build)
    "deb.nodesource.com"
)

# Resolve the vscode user's UID (firewall rules target this user only)
VSCODE_UID=$(id -u vscode 2>/dev/null || echo 1000)

# Flush existing rules
iptables -F OUTPUT 2>/dev/null || true
iptables -F INPUT 2>/dev/null || true
iptables -F FORWARD 2>/dev/null || true
ipset destroy allowed-domains 2>/dev/null || true

# Create ipset for allowed IPs
ipset create allowed-domains hash:ip

# Resolve and add each domain
for domain in "${ALLOWED_DOMAINS[@]}"; do
    # Skip comments and empty lines
    [[ "$domain" =~ ^#.*$ ]] && continue
    [[ -z "$domain" ]] && continue

    # If it looks like an IP, add directly
    if [[ "$domain" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        ipset add allowed-domains "$domain" 2>/dev/null || true
        continue
    fi

    # Resolve hostname
    IPs=$(dig +short "$domain" A 2>/dev/null | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' || true)
    for ip in $IPs; do
        ipset add allowed-domains "$ip" 2>/dev/null || true
    done
done

# Allow loopback
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A INPUT -i lo -j ACCEPT

# Allow established/related connections
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow DNS (UDP + TCP on port 53)
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

# Allow SSH outbound
iptables -A OUTPUT -p tcp --dport 22 -j ACCEPT

# Allow outbound to whitelisted IPs (HTTP + HTTPS)
iptables -A OUTPUT -m set --match-set allowed-domains dst -j ACCEPT

# Allow game socket bridge ports to the host (for RL agent ↔ game communication).
# host.docker.internal is the standard Docker/OrbStack alias for the host machine.
HOST_IP=$(getent hosts host.docker.internal 2>/dev/null | awk '{print $1}' || true)
GATEWAY_IP=$(ip route show default 2>/dev/null | awk '{print $3}' || true)
for ip in $HOST_IP $GATEWAY_IP; do
    if [ -n "$ip" ]; then
        iptables -A OUTPUT -p tcp -d "$ip" --dport 7777:7784 -j ACCEPT
        iptables -A OUTPUT -p udp -d "$ip" --dport 7777:7784 -j ACCEPT
        echo "  Game ports 7777-7784 (TCP+UDP) allowed to $ip"
    fi
done

# Only restrict the vscode user — root (dockerd, apt) is unrestricted
iptables -A OUTPUT -m owner --uid-owner "$VSCODE_UID" -j DROP

echo "Firewall initialized. Allowed domains: ${#ALLOWED_DOMAINS[@]} (restricted to UID $VSCODE_UID)"
