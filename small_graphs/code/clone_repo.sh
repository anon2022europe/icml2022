git config --global core.sshCommand 'ssh -i  /home/ubuntu/.ssh/github_deploy'
git clone git@github.com:anon-anon/peawgan.git
cd peawgan
git submodule init
git pull --recurse-submodules
