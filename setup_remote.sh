#!/bin/bash

set -e
set -o pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 \"ssh_command\""
    exit 1
fi

ssh_command="$1"

# Extract relevant information from the SSH command
host=$(echo "$ssh_command" | awk '{print $2}' | cut -d@ -f2)
port=$(echo "$ssh_command" | awk '{print $4}')

# Update ~/.ssh/config
config_file="$HOME/.ssh/config"
temp_file=$(mktemp)

awk '
/^# START best-hn config$/,/^# END best-hn config$/ {next}
{print}
END {
    print "# START best-hn config"
    print "Host best-hn"
    print "    HostName '"$host"'"
    print "    Port '"$port"'"
    print "    User root"
    print "# END best-hn config"
}
' "$config_file" > "$temp_file"

mv "$temp_file" "$config_file"

# Copy remote_lowtrust keypair to the remote system and rename to id_rsa
scp -P "$port" ~/.ssh/remote_lowtrust "root@$host:~/.ssh/id_rsa"r
scp -P "$port" ~/.ssh/remote_lowtrust.pub "root@$host:~/.ssh/id_rsa.pub"

# Clone the repository and run install_deps.sh on the remote system
ssh -p "$port" "root@$host" << EOF
    # Set git username and email based on local configuration
    local_name="$(git config --get user.name)"
    local_email="$(git config --get user.email)"
    git config --global user.name "$local_name"
    git config --global user.email "$local_email"

    # Add GitHub's SSH key to known_hosts
    mkdir -p ~/.ssh
    ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

    mkdir -p /workspace
    cd /workspace
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:OpenPipe/best-hn.git
    cd best-hn/
    ./prepare_env.sh

EOF

# Copy the local .env file to the remote repository
scp .env "best-hn:/workspace/.env"

echo "Remote setup completed successfully!"