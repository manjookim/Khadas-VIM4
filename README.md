# Khadas new VIM4
Khadas new VIM4 NPU - AI 가속기 연구 

## 환경설정
- PC / Khadas new VIM4 따로 환경설정

### PC 환경설정
#### 1. Docker Install   
```
# Add Docker's official GPG key:
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
```
```
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl status docker
sudo systemctl start docker
sudo docker run hello-world
```
#### 2. Docker image pull 
```
docker pull numbqq/npu-vim4
```
#### 3. Get Convert Tool
```
wget https://dl.khadas.com/products/vim4/tools/npu-sdk/vim4_npu_sdk-ddk-3.4.7.7-250508.tgz
tar xvzf vim4_npu_sdk-ddk-3.4.7.7-250508.tgz
cd vim4_npu_sdk-ddk-3.4.7.7-250508
ls
#adla-toolkit-binary  adla-toolkit-binary-3.1.7.4  convert-in-docker.sh  Dockerfile  docs  README.md
```

### Khadas new VIM4 환경설정
#### 1. Git clone & ksnn install 
```
sudo apt update
source venv/bin/activate
git clone https://github.com/khadas/ksnn-vim4
pip3 install ksnn/ksnn_vim4-1.4.1-py3-none-any.whl
```



## Reference
https://www.khadas.com/vim4?page=6
https://docs.khadas.com/products/sbc/vim4/start
https://docs.ultralytics.com/ko/models/yolov8/
https://github.com/khadas/ksnn-vim4
