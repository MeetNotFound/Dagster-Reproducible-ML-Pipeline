# PRACTICAL 6C — Dockerfile PHP Dynamic App

**Student:** Atharva Ahirrao | **GitHub:** https://github.com/Atharvaahirrao/pract5

---

## Aim
Build a Docker image with PHP + Apache to handle a form and generate dynamic response.

## Steps

### 1. Create Project Folder
```cmd
mkdir C:\Users\athar\docker-web2
cd C:\Users\athar\docker-web2
```

### 2. Create `index.html`
> Open Notepad → paste code → Save as `index.html` in docker-web2

```html
<!DOCTYPE html>
<html>
<head>
    <title>Dockerfile form</title>
</head>
<body>
    <h2>Simple form (Docker)</h2>
    <form action="submit.php" method="post">
        Name: <input type="text" name="username" required>
        <br><br>
        <input type="submit" value="Submit">
    </form>
</body>
</html>
```

### 3. Create `submit.php`
> Open Notepad → paste code → Save as `submit.php` in docker-web2

```php
<?php
$name = $_POST['username'];
echo "<h1>Hello, $name!</h1>";
echo "<p>This response is generated dynamically using PHP inside Docker.</p>";
?>
```

### 4. Create `Dockerfile`
> Open Notepad → paste code → Save as `Dockerfile` (All Files, no extension)

```dockerfile
FROM php:8.2-apache
COPY . /var/www/html/
EXPOSE 80
```

### 5. Build and Run
```cmd
docker build -t docker-web2 .
docker run -d -p 8088:80 docker-web2
```

### 6. Test
- Open browser → `http://localhost:8088`
- You see the form → Enter your name → Click Submit
- PHP responds: **"Hello, Atharva!"**

## Dockerfile Explanation
| Line | Meaning |
|------|---------|
| `FROM php:8.2-apache` | PHP 8.2 with Apache built-in |
| `COPY . /var/www/html/` | Copy ALL files to web root |
| `EXPOSE 80` | Port 80 for HTTP |
