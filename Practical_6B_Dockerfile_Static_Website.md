# PRACTICAL 6B — Dockerfile Static Website

**Student:** Atharva Ahirrao | **GitHub:** https://github.com/Atharvaahirrao/pract5

---

## Aim
Build a custom Docker image using Dockerfile to serve a static HTML page using Apache.

## Steps

### 1. Create Project Folder
```cmd
mkdir C:\Users\athar\docker-web1
cd C:\Users\athar\docker-web1
```

### 2. Create `index.html`
> Open Notepad → paste code → Save as `index.html` in docker-web1 folder

```html
<!DOCTYPE html>
<html>
<head>
    <title>Dockerfile website</title>
</head>
<body>
    <h1>Hello from Dockerfile!</h1>
    <p>This is hosted using Dockerfile</p>
</body>
</html>
```

### 3. Create `Dockerfile`
> Open Notepad → paste code → Save as `Dockerfile`
> ⚠️ NO extension! In Save dialog: File name = `Dockerfile`, Save as type = **All Files**

```dockerfile
FROM httpd:latest
COPY index.html /usr/local/apache2/htdocs/
EXPOSE 80
```

### 4. Build Image
```cmd
docker build -t myimage .
```

### 5. Run Container
```cmd
docker run -d -p 8085:80 myimage
```

### 6. Test
- Open browser → `http://localhost:8085`
- You see: **"Hello from Dockerfile!"**

## Dockerfile Explanation
| Line | Meaning |
|------|---------|
| `FROM httpd:latest` | Use official Apache image as base |
| `COPY index.html /usr/local/apache2/htdocs/` | Copy HTML into Apache web directory |
| `EXPOSE 80` | Document that app uses port 80 |
