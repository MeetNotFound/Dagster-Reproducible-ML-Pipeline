# PRACTICAL 6A — Docker Basic Commands

**Student:** Atharva Ahirrao | **GitHub:** https://github.com/Atharvaahirrao/pract5

---

## Aim
Understand Docker architecture and run basic commands to manage images and containers.

## Commands

### 1. Check Version
```cmd
docker --version
```

### 2. System Info
```cmd
docker info
```

### 3. Pull and Run Hello World
```cmd
docker pull hello-world
docker run hello-world
```

### 4. Run Apache HTTP Server
```cmd
docker run -d -p 8082:80 httpd
```
- Open browser → `http://localhost:8082` → **"It works!"**

### 5. Run Nginx
```cmd
docker run -d -p 8083:80 nginx
```
- Open browser → `http://localhost:8083` → **"Welcome to nginx!"**

### 6. Run Python Interactively
```cmd
docker run -it python
```
- Python shell opens → type `1+2` → output `3`
- Type `exit()` to come out

### 7. Manage Containers and Images
```cmd
docker ps           # Running containers
docker ps -a        # All containers
docker images       # All images
docker stop <id>    # Stop container
docker rm <id>      # Remove container
```

## Flags Reference
| Flag | Meaning |
|------|---------|
| `-d` | Detached — runs in background |
| `-p 8082:80` | Map host port 8082 to container port 80 |
| `-it` | Interactive terminal |
