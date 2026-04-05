# ============================================================
# Nuclear Throne RL Agent — Dev Container
# Base: Python 3.12 (primary runtime) + Node.js 20 (asset
# extraction scripts + Claude Code)
# ============================================================
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

ARG TZ
ENV TZ="$TZ"

ARG CLAUDE_CODE_VERSION=latest

# ============================================================
# System packages
# ============================================================
RUN rm -f /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    # Claude Code baseline
    less git procps sudo fzf zsh man-db unzip gnupg2 gh \
    iptables ipset iproute2 dnsutils aggregate jq nano vim \
    # Docker CLI — for managing game simulation containers
    ca-certificates curl gnupg lsb-release \
    # Socket debugging — used by verify_build.sh and Agent 03 health checks
    netcat-openbsd \
    # X11 / display utilities — for inspecting Xvfb output during debugging
    x11-utils xvfb \
    # Audio dummy device — needed if running game locally (outside Docker)
    pulseaudio \
    # Build tools — required for some Python package native extensions
    build-essential libssl-dev libffi-dev \
    # HDF5 — for loading .hdf5 demonstration datasets (Agent 05)
    libhdf5-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# ============================================================
# Node.js 20 — for npm run regen (asset extraction) and Claude Code
# NOTE: The mcr.microsoft.com/devcontainers/python image does not
# bundle Node.js, so we install it here.
# ============================================================
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ============================================================
# User setup (mcr devcontainers use "vscode", not "node")
# ============================================================
ARG USERNAME=vscode

RUN mkdir -p /usr/local/share/npm-global && \
    chown -R ${USERNAME}:${USERNAME} /usr/local/share

# Shell history persistence
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory

ENV DEVCONTAINER=true

# Workspace and Claude config directories
RUN mkdir -p /workspace /home/${USERNAME}/.claude && \
    chown -R ${USERNAME}:${USERNAME} /workspace /home/${USERNAME}/.claude

WORKDIR /workspace

# ============================================================
# Python ML toolchain
# ============================================================
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        # RL core
        gymnasium \
        stable-baselines3 \
        imitation \
        # Deep learning
        # NOTE: torch is installed without CUDA/MPS extras here.
        # For MPS acceleration on Apple Silicon, run training
        # natively on the host (outside this container) using:
        #   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        # MPS is a macOS Metal feature and is unavailable in Linux containers.
        torch \
        numpy \
        # Data / serialisation
        h5py \
        # Experiment tracking
        wandb \
        # Testing
        pytest \
        pytest-timeout \
        # Utilities
        tqdm \
        scipy

# ============================================================
# Git delta (quality-of-life for git diffs)
# ============================================================
ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
    wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# ============================================================
# Switch to non-root user for remaining setup
# ============================================================
USER ${USERNAME}

ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin
ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano

# Zsh + Powerlevel10k
ARG ZSH_IN_DOCKER_VERSION=1.2.0
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
    -p git \
    -p fzf \
    -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
    -a "source /usr/share/doc/fzf/examples/completion.zsh" \
    -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    -x

# ============================================================
# Install Claude Code
# ============================================================
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# ============================================================
# Firewall setup
# ============================================================
COPY init-firewall.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/init-firewall.sh && \
    echo "${USERNAME} ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/${USERNAME}-firewall && \
    chmod 0440 /etc/sudoers.d/${USERNAME}-firewall

USER ${USERNAME}
