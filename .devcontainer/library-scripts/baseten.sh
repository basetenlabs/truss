#!/usr/bin/env bash
set -e

USERNAME=${1:-"automatic"}
if [ "${USERNAME}" = "root" ]; then
    user_rc_path="/root"
else
    user_rc_path="/home/${USERNAME}"
fi

echo "Installing git-lfs"
curl -L -o git-lfs.tar.gz https://github.com/git-lfs/git-lfs/releases/download/v2.9.0/git-lfs-linux-amd64-v2.9.0.tar.gz
tar -xvf git-lfs.tar.gz
mv git-lfs /usr/local/bin/git-lfs
rm git-lfs.tar.gz
chmod 0755 /usr/local/bin/git-lfs
