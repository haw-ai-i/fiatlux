FROM docker.io/library/ros:kilted-ros-core AS build

# Install pixi
RUN apt update && apt install -y git && curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=0.67.2 bash
ENV PATH="/root/.pixi/bin:$PATH"

# Add base dependencies
COPY fiatlux_example_policies /ws_fiatlux/src/fiatlux/fiatlux_example_policies
COPY fiatlux_model /ws_fiatlux/src/fiatlux/fiatlux_model
COPY fiatlux_interfaces /ws_fiatlux/src/fiatlux/fiatlux_interfaces
COPY fiatlux_utils /ws_fiatlux/src/fiatlux/fiatlux_utils
COPY pixi.toml pixi.lock /ws_fiatlux/src/fiatlux/
COPY pixi_env_setup.sh /ws_fiatlux/src/fiatlux/

# Add other dependencies

SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/rattler/cache --mount=type=cache,target=/ws_fiatlux/src/fiatlux/.pixi/build \
    cd /ws_fiatlux/src/fiatlux && pixi install --locked

WORKDIR /ws_fiatlux/src/fiatlux
COPY --chmod=755 <<"EOF" /entrypoint.sh
#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if [[ -z "$FIATLUX_ROUTER_ADDR" ]]; then
  echo "FIATLUX_ROUTER_ADDR must be provided"
  exit 1
fi

should_enable_acl() {
  [[ "$FIATLUX_ENABLE_ACL" == "true" || "$FIATLUX_ENABLE_ACL" == "1" ]]
}

# prepare the credentials file
if should_enable_acl; then
  if [[ ! (-n "$FIATLUX_MODEL_PASSWD" ) ]]; then
    echo "FIATLUX_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$FIATLUX_MODEL_PASSWD" >> /credentials.txt
fi

# start the model
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$FIATLUX_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$FIATLUX_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"
exec pixi run --as-is ros2 run fiatlux_model fiatlux_model "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=fiatlux_example_policies.ros.CheatCode", "-p", "use_sim_time:=true"]
