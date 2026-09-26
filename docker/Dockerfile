FROM python:3.12

# git: 从远端 clone QuantStudio 源码时使用（QS_SOURCE=git）
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# 容器内用户根目录下的目录约定：
#   ~/python_envs   Python 虚拟环境（QS）
#   ~/packages      第三方源码仓库（QuantStudio）
#   ~/workspace     QS 环境的配置目标目录（.mcp.json / CLAUDE.local.md / .claude/skills）
#   ~/data          挂载点：JYDBDoc 文档缓存、HDF5DB 因子库
ENV QS_ROOT=/root \
    QS_VENV=/root/python_envs/QS \
    QS_PACKAGES=/root/packages \
    QS_WORKSPACE=/root/workspace \
    QS_DATA=/root/data \
    QS_REPO=/root/packages/QuantStudio \
    QS_GIT_URL=https://github.com/Scorpi000/QuantStudio.git \
    QS_GIT_BRANCH=v2 \
    QS_SOURCE=git

ARG QS_GIT_URL
ARG QS_GIT_BRANCH
ARG QS_SOURCE

RUN mkdir -p "${QS_PACKAGES}" "${QS_WORKSPACE}" "${QS_DATA}" "${QS_ROOT}/python_envs"

# 1) 创建 QS 虚拟环境
RUN python -m venv "${QS_VENV}" \
    && "${QS_VENV}/bin/python" -m pip install --no-cache-dir --upgrade pip

# 2) 取得 QuantStudio 源码到 ~/packages
#    QS_SOURCE=git  → 从远端 clone（正式构建）
#    QS_SOURCE=copy → 使用构建上下文中的源码副本（本地测试，不依赖远端分支）
COPY . /tmp/qs-context

RUN if [ "${QS_SOURCE}" = "copy" ]; then \
        mkdir -p "${QS_REPO}" \
        && cp -a /tmp/qs-context/. "${QS_REPO}/" \
        && rm -rf "${QS_REPO}/.git"; \
    else \
        git clone --branch "${QS_GIT_BRANCH}" --depth 1 "${QS_GIT_URL}" "${QS_REPO}"; \
    fi \
    && rm -rf /tmp/qs-context

# 3) 以 ~/workspace 为目标目录，在 QS 虚拟环境下运行 setup_qs_env.py
#    JYDBDoc 缓存位于 ~/data/JYDBDoc（运行时由 volume 挂载）
RUN "${QS_VENV}/bin/python" "${QS_REPO}/scripts/setup_qs_env.py" \
        --target-dir "${QS_WORKSPACE}" \
        --cache-dir "${QS_DATA}/JYDBDoc"

# 默认进入 QS 环境
ENV PATH="${QS_VENV}/bin:${PATH}" \
    VIRTUAL_ENV="${QS_VENV}" \
    PYTHONPATH="${QS_REPO}"
WORKDIR "${QS_WORKSPACE}"

CMD ["bash"]
