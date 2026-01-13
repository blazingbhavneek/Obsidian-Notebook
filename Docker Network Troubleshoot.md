
## Problem Overview

Docker containers running on Rocky Linux (or any Linux) cannot access the internet despite host having connectivity. Common symptoms:

- `ping`, `curl`, `apt update`, `dnf install` all timeout
- Host can access internet, containers cannot
- `--network host` doesn't help
- Proxy settings copied from host don't work

---

## Understanding Docker Network Layers

Docker has **4 distinct layers** that need proxy configuration:

1. **Docker Daemon** - for pulling/pushing images
2. **Docker Client** - for CLI commands
3. **Container Runtime** - environment variables in running containers
4. **Build Time** - during `docker build` operations

Missing configuration at ANY layer causes failures.

---

## Solution 1: Docker Daemon Proxy Configuration

### For systemd-based systems (Rocky Linux, Ubuntu, CentOS, RHEL)

**Step 1: Create systemd drop-in directory**

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
```

**Step 2: Create proxy configuration file**

```bash
sudo nano /etc/systemd/system/docker.service.d/http-proxy.conf
```

**Step 3: Add proxy settings**

```ini
[Service]
Environment="HTTP_PROXY=http://proxy.company.com:8080"
Environment="HTTPS_PROXY=http://proxy.company.com:8080"
Environment="NO_PROXY=localhost,127.0.0.1,172.17.0.0/16,*.local,.company.net"
```

**Important Notes:**

- Replace `proxy.company.com:8080` with your actual proxy
- `NO_PROXY` should include Docker network ranges (typically `172.17.0.0/16`)
- If proxy needs authentication: `http://username:password@proxy:8080`
- Special characters need double escaping: `http://user:p%23ss@proxy:8080` for `p#ss`

**Step 4: Reload and restart Docker**

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

**Step 5: Verify configuration**

```bash
sudo systemctl show --property=Environment docker
```

### Alternative: Using daemon.json (less common)

```bash
sudo nano /etc/docker/daemon.json
```

```json
{
  "proxies": {
    "http-proxy": "http://proxy.company.com:8080",
    "https-proxy": "http://proxy.company.com:8080",
    "no-proxy": "localhost,127.0.0.1,172.17.0.0/16"
  }
}
```

Then restart: `sudo systemctl restart docker`

---

## Solution 2: Docker Client Proxy Configuration

### Configure ~/.docker/config.json

**Create or edit the file:**

```bash
mkdir -p ~/.docker
nano ~/.docker/config.json
```

**Add proxy configuration:**

```json
{
  "proxies": {
    "default": {
      "httpProxy": "http://proxy.company.com:8080",
      "httpsProxy": "http://proxy.company.com:8080",
      "noProxy": "localhost,127.0.0.1,*.local,172.17.0.0/16"
    }
  }
}
```

**This automatically sets proxy for:**

- All containers started with `docker run`
- All builds with `docker build`
- Both `HTTP_PROXY` and `http_proxy` environment variables

---

## Solution 3: Container Runtime Proxy

### Method A: Using -e flags

```bash
docker run --rm \
  -e HTTP_PROXY=http://proxy.company.com:8080 \
  -e HTTPS_PROXY=http://proxy.company.com:8080 \
  -e NO_PROXY=localhost,127.0.0.1 \
  alpine sh -c 'apk add curl && curl https://google.com'
```

### Method B: Using Docker Compose

```yaml
version: '3.8'
services:
  myapp:
    image: ubuntu:22.04
    environment:
      - HTTP_PROXY=http://proxy.company.com:8080
      - HTTPS_PROXY=http://proxy.company.com:8080
      - NO_PROXY=localhost,127.0.0.1
```

### Method C: In Dockerfile

```dockerfile
FROM ubuntu:22.04
ENV HTTP_PROXY=http://proxy.company.com:8080
ENV HTTPS_PROXY=http://proxy.company.com:8080
ENV NO_PROXY=localhost,127.0.0.1
RUN apt-get update && apt-get install -y curl
```

---

## Solution 4: DNS Configuration

### Symptom: Can ping 8.8.8.8 but cannot resolve google.com

**Method A: Docker daemon DNS (permanent)**

```bash
sudo nano /etc/docker/daemon.json
```

```json
{
  "dns": ["8.8.8.8", "8.8.4.4"],
  "dns-search": ["company.com"]
}
```

Then restart: `sudo systemctl restart docker`

**Method B: Per-container DNS**

```bash
docker run --rm --dns 8.8.8.8 --dns 8.8.4.4 alpine nslookup google.com
```

**Method C: Docker Compose**

```yaml
version: '3.8'
services:
  myapp:
    image: ubuntu:22.04
    dns:
      - 8.8.8.8
      - 8.8.4.4
    dns_search:
      - company.com
```

**Check host's DNS to use correct servers:**

```bash
# Rocky Linux / CentOS / RHEL
cat /etc/resolv.conf
systemctl status systemd-resolved
systemd-resolve --status

# Ubuntu
resolvectl status
nmcli dev show | grep 'IP4.DNS'
```

---

## Solution 5: Firewall and IP Forwarding

### Rocky Linux / CentOS / RHEL (firewalld)

**Check firewall status:**

```bash
sudo firewall-cmd --state
```

**Add Docker to trusted zone:**

```bash
sudo firewall-cmd --permanent --zone=trusted --add-interface=docker0
sudo firewall-cmd --permanent --zone=public --add-masquerade
sudo firewall-cmd --reload
sudo systemctl restart docker
```

**Test if firewall is the issue:**

```bash
sudo systemctl stop firewalld
docker run --rm alpine ping -c 2 8.8.8.8
sudo systemctl start firewalld
```

### Check SELinux

```bash
getenforce
# If Enforcing, temporarily disable to test:
sudo setenforce 0
docker run --rm alpine ping -c 2 8.8.8.8
sudo setenforce 1
```

### Verify IP Forwarding

```bash
# Check current setting
sysctl net.ipv4.ip_forward
# Should output: net.ipv4.ip_forward = 1

# If it's 0, enable it:
echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Check iptables

```bash
# View current rules
sudo iptables -L -n -v
sudo iptables -t nat -L -n -v

# Check FORWARD chain policy (should be ACCEPT or have Docker rules)
sudo iptables -L FORWARD -n -v

# If FORWARD chain drops everything, allow Docker:
sudo iptables -I DOCKER-USER -i eth0 -o docker0 -j ACCEPT
sudo iptables -I DOCKER-USER -i docker0 -o eth0 -j ACCEPT
```

---

## Solution 6: Package Manager Proxy Configuration

Even with container environment variables set, package managers need explicit configuration.

### For Ubuntu/Debian (apt)

**Method A: In Dockerfile**

```dockerfile
FROM ubuntu:22.04

# Configure apt proxy
RUN echo 'Acquire::http::Proxy "http://proxy.company.com:8080";' > /etc/apt/apt.conf.d/proxy.conf
RUN echo 'Acquire::https::Proxy "http://proxy.company.com:8080";' >> /etc/apt/apt.conf.d/proxy.conf

RUN apt-get update && apt-get install -y curl
```

**Method B: Inline during docker run**

```bash
docker run --rm ubuntu bash -c \
  'echo "Acquire::http::Proxy \"http://proxy.company.com:8080\";" > /etc/apt/apt.conf.d/proxy.conf && \
   apt-get update && apt-get install -y curl'
```

**Method C: Using environment variable**

```bash
docker run --rm -e http_proxy=http://proxy:8080 ubuntu bash -c 'apt-get update'
```

### For CentOS/Rocky Linux/RHEL (yum/dnf)

**Method A: In Dockerfile**

```dockerfile
FROM rockylinux:9

# Configure dnf/yum proxy
RUN echo 'proxy=http://proxy.company.com:8080' >> /etc/dnf/dnf.conf
RUN echo 'proxy=http://proxy.company.com:8080' >> /etc/yum.conf

RUN dnf update -y && dnf install -y curl
```

**Method B: Inline**

```bash
docker run --rm rockylinux:9 bash -c \
  'echo "proxy=http://proxy:8080" >> /etc/dnf/dnf.conf && \
   dnf install -y curl'
```

**Method C: Using environment variable**

```bash
docker run --rm -e http_proxy=http://proxy:8080 rockylinux:9 dnf install -y curl
```

### For Alpine (apk)

**Alpine respects HTTP_PROXY environment variable:**

```bash
docker run --rm -e HTTP_PROXY=http://proxy:8080 alpine sh -c 'apk add curl'
```

**Or in Dockerfile:**

```dockerfile
FROM alpine:latest
ENV HTTP_PROXY=http://proxy.company.com:8080
RUN apk add --no-cache curl
```

---

## Solution 7: SOCKS5 Proxy Tunnel (Advanced)

When traditional proxy settings don't work, tunnel container traffic through a SOCKS5 proxy running on your PC.

### Setup A: Run SOCKS5 proxy on your PC

**Option 1: Using SSH**

```bash
# On your PC, create SOCKS5 proxy
ssh -D 0.0.0.0:1080 -N localhost
```

**Option 2: Using dante-server**

```bash
# On Ubuntu/Debian
sudo apt install dante-server

# Configure /etc/danted.conf
sudo nano /etc/danted.conf
```

### Setup B: Point containers to your SOCKS5 proxy

```bash
# Replace YOUR_PC_IP with your actual IP
docker run --rm \
  -e ALL_PROXY=socks5://YOUR_PC_IP:1080 \
  alpine sh -c 'apk add curl && curl https://google.com'
```

**Important:** Make sure firewall on your PC allows connections on port 1080 from the remote host.

### Using tun2socks in container

```bash
docker run --rm --cap-add=NET_ADMIN \
  -v /dev/net/tun:/dev/net/tun \
  alpine sh -c '
  apk add curl tun2socks ip
  ip tuntap add mode tun dev tun0
  ip addr add 198.18.0.1/15 dev tun0
  ip link set dev tun0 up
  ip route add default via 198.18.0.1 dev tun0
  tun2socks -device tun0 -proxy socks5://YOUR_PC_IP:1080 &
  sleep 2
  curl https://google.com
'
```

---

## Solution 8: Network Mode Options

### --network host

Uses host's network stack directly:

```bash
docker run --network host -e HTTP_PROXY=$HTTP_PROXY alpine sh -c 'apk add curl'
```

**Note:** Still needs proxy environment variables, just skips Docker networking layer.

### Custom bridge network

```bash
# Create custom bridge
docker network create --driver bridge my-bridge

# Run container with it
docker run --network my-bridge alpine ping 8.8.8.8
```

---

## Solution 9: Using Local Proxy Container

Run a proxy container on the host that handles authentication/certificates, then point other containers to it.

```bash
# Run Squid proxy container
docker run -d --name proxy --network host sameersbn/squid

# Use it in other containers
docker run --rm -e http_proxy=http://127.0.0.1:3128 alpine sh -c 'apk add curl'
```

---

## Diagnostic Commands

### Test connectivity layers

**1. Test raw IP connectivity:**

```bash
docker run --rm alpine ping -c 3 8.8.8.8
```

**2. Test DNS resolution:**

```bash
docker run --rm alpine nslookup google.com
docker run --rm alpine ping -c 3 google.com
```

**3. Test HTTP:**

```bash
docker run --rm alpine sh -c 'apk add curl && curl -I https://google.com'
```

**4. Check proxy from container:**

```bash
docker run --rm -e HTTP_PROXY=http://proxy:8080 alpine sh -c \
  'echo "Proxy: $HTTP_PROXY" && apk add curl && curl -v https://google.com'
```

### View Docker logs

```bash
# SystemD-based
sudo journalctl -u docker.service -f

# Traditional
sudo tail -f /var/log/docker.log
```

### Check Docker network

```bash
# List networks
docker network ls

# Inspect bridge network
docker network inspect bridge

# Check container's network
docker inspect <container_id> | grep -A 20 NetworkSettings
```

### Check iptables rules

```bash
# View all Docker chains
sudo iptables -t nat -L -n -v | grep DOCKER
sudo iptables -L DOCKER -n -v
sudo iptables -L DOCKER-USER -n -v
```

---

## Complete Working Example

### Scenario: Rocky Linux host behind corporate proxy

**1. Configure Docker daemon:**

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=http://proxy.company.com:8080"
Environment="HTTPS_PROXY=http://proxy.company.com:8080"
Environment="NO_PROXY=localhost,127.0.0.1,172.17.0.0/16"
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

**2. Configure Docker client:**

```bash
mkdir -p ~/.docker
tee ~/.docker/config.json <<EOF
{
  "proxies": {
    "default": {
      "httpProxy": "http://proxy.company.com:8080",
      "httpsProxy": "http://proxy.company.com:8080",
      "noProxy": "localhost,127.0.0.1,172.17.0.0/16"
    }
  }
}
EOF
```

**3. Configure DNS:**

```bash
sudo tee /etc/docker/daemon.json <<EOF
{
  "dns": ["8.8.8.8", "8.8.4.4"]
}
EOF
sudo systemctl restart docker
```

**4. Configure firewall:**

```bash
sudo firewall-cmd --permanent --zone=trusted --add-interface=docker0
sudo firewall-cmd --permanent --zone=public --add-masquerade
sudo firewall-cmd --reload
```

**5. Test:**

```bash
docker run --rm alpine sh -c 'apk add curl && curl https://google.com'
```

---

## Troubleshooting Checklist

- [ ] Host can access internet (`curl https://google.com`)
- [ ] Docker daemon has proxy configured (`systemctl show docker`)
- [ ] Docker client has proxy configured (`cat ~/.docker/config.json`)
- [ ] DNS is configured (`cat /etc/docker/daemon.json`)
- [ ] IP forwarding is enabled (`sysctl net.ipv4.ip_forward`)
- [ ] Firewall allows Docker traffic
- [ ] SELinux is not blocking (test with `setenforce 0`)
- [ ] Container has proxy environment variables (`docker run --rm alpine env | grep -i proxy`)
- [ ] Package manager proxy is configured (apt.conf.d, dnf.conf)
- [ ] Proxy server is reachable from container network
- [ ] NO_PROXY includes Docker network ranges

---

## Key Takeaways

1. **Docker has 4 layers** that need proxy configuration - missing any breaks connectivity
2. **~/.bashrc is NOT sourced** in Docker containers by default
3. **Package managers need separate configuration** from environment variables
4. **DNS issues are common** - configure in daemon.json
5. **Firewall/SELinux** on Rocky Linux often blocks by default
6. **IP forwarding must be enabled** for containers to reach internet
7. **NO_PROXY must include Docker networks** (172.17.0.0/16) to avoid routing loops

---

## Additional Resources

- Docker Official Proxy Docs: https://docs.docker.com/engine/daemon/proxy/
- Docker Network Docs: https://docs.docker.com/engine/network/
- tun2socks: https://github.com/xjasonlyu/tun2socks
- Corporate Proxy Guide: https://www.datacamp.com/tutorial/docker-proxy



# Quick Tests

```bash
docker run --rm ubuntu:24.04 bash -c 'apt update && apt install -y curl && curl -I https://google.com'
```

This tests:

1. DNS resolution (apt update needs to resolve archive.ubuntu.com)
2. Package manager connectivity (apt)
3. HTTPS connectivity (curl)

If this works, your network is configured correctly.

**Quick alternative tests:**

```bash
# Test just apt
docker run --rm ubuntu:24.04 apt update

# Test with Alpine (smaller, faster)
docker run --rm alpine sh -c 'apk update && apk add curl && curl -I https://google.com'
```


