#!/bin/bash

set -e
set -o pipefail

# Default remote name
REMOTE_NAME="best-hn"

# Function to display usage
usage() {
    echo "Usage: $0 [--remote-name REMOTE_NAME] \"ssh_command\""
    exit 1
}

# Parse arguments
if [ "$#" -lt 1 ]; then
    usage
fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --remote-name|-n)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                REMOTE_NAME="$2"
                shift 2
            else
                echo "Error: --remote-name requires a non-empty value."
                usage
            fi
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            ssh_command="$1"
            shift
            ;;
    esac
done

if [ -z "$ssh_command" ]; then
    echo "Error: Missing ssh_command."
    usage
fi

# Extract relevant information from the SSH command
host=$(echo "$ssh_command" | awk '{print $2}' | cut -d@ -f2)
port=$(echo "$ssh_command" | awk '{print $4}')

# Update ~/.ssh/config
config_file="$HOME/.ssh/config"
temp_file=$(mktemp)

awk -v remote_name="$REMOTE_NAME" -v host="$host" -v port="$port" '
BEGIN { in_block = 0 }
/^# START/ && $0 ~ remote_name { in_block = 1; next }
/^# END/ && $0 ~ remote_name { in_block = 0; next }
!in_block { print }
END {
    print "# START " remote_name " config"
    print "Host " remote_name
    print "    HostName " host
    print "    Port " port
    print "    User root"
    print "# END " remote_name " config"
}
' "$config_file" > "$temp_file"

mv "$temp_file" "$config_file"
chmod 600 "$config_file"

# Copy remote_lowtrust keypair to the remote system and rename to id_rsa
scp -P "$port" ~/.ssh/remote_lowtrust "root@$host:~/.ssh/id_rsa"
scp -P "$port" ~/.ssh/remote_lowtrust.pub "root@$host:~/.ssh/id_rsa.pub"

# Copy the local .env file to the remote repository
scp .env "${REMOTE_NAME}:/workspace/.env"

local_name="$(git config --get user.name)"
local_email="$(git config --get user.email)"

# Clone the repository and run prepare_env.sh on the remote system
ssh -p "$port" "root@$host" << EOF
    set -e
    set -o pipefail

    # Set git username and email based on local configuration
    echo "Setting git user.name to $local_name and user.email to $local_email"
    git config --global user.name "$local_name"
    git config --global user.email "$local_email"

    # Add GitHub's SSH key to known_hosts
    mkdir -p ~/.ssh
    ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

    # Clone the repository
    mkdir -p /workspace
    cd /workspace
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:OpenPipe/best-hn.git
    cd best-hn/
    ./prepare_env.sh
EOF

echo "Remote setup completed successfully!"
