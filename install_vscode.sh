#!/bin/bash

echo ==INSTALLING CODE-SERVER==
yum install -y https://github.com/coder/code-server/releases/download/v4.11.0/code-server-4.11.0-amd64.rpm
/home/ec2-user/anaconda3/envs/JupyterSystemEnv/bin/pip install -U keytar jupyter-server-proxy

echo ==UPDATING JUPYTER SERVER CONFIG==
#########################################
### INTEGRATE CODE-SERVER WITH JUPYTER
#########################################
cat >>/home/ec2-user/.jupyter/jupyter_notebook_config.py <<EOC
c.ServerProxy.servers = {
  'vscode': {
      'launcher_entry': {
            'enabled': True,
            'title': 'VS Code',
      },
      'command': ['code-server', '--auth', 'none', '--disable-telemetry', '--bind-addr', '127.0.0.1:{port}'],
      'environment' : {'XDG_DATA_HOME' : '/home/ec2-user/SageMaker/vscode-config'},
      'absolute_url': False,
      'timeout': 30
  }
}
EOC

echo ==INSTALL SUCCESSFUL. RESTARTING JUPYTER==
# RESTART THE JUPYTER SERVER
systemctl restart jupyter-server