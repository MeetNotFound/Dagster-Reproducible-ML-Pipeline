# PRACTICAL 5A — Maven Project Build via Jenkins

**Student:** Atharva Ahirrao | **GitHub:** https://github.com/Atharvaahirrao/pract5

---

## Aim
Configure Jenkins to automatically build a Java Maven project — download dependencies, compile, test, generate JAR.

---

## PART 1 — Fix pom.xml in NetBeans

### 1. Open pom.xml
- Open NetBeans → Expand `mavenproject11`
- Double click `pom.xml`

### 2. Change Java Version
Find:
```xml
<maven.compiler.release>23</maven.compiler.release>
```
Change to:
```xml
<maven.compiler.release>11</maven.compiler.release>
```
- Press `Ctrl + S` to save

---

## PART 2 — Push to GitHub

### 3. Git Bash Commands

```bash
cd "C:\Users\athar\OneDrive\Documents\NetBeansProjects\mavenproject11"
git add .
git commit -m "fix java version to 11"
git push origin master
```

---

## PART 3 — Configure Jenkins

### 4. Install Maven Plugin
- Jenkins → **Manage Jenkins** → **Plugins**
- Click **Available plugins**
- Search: `Maven Integration`
- Check it → Click **Install**

### 5. Configure Maven Tool
- **Manage Jenkins** → **Tools**
- Scroll to **Maven installations** → Click **Add Maven**
- Name: `maven`
- Check **Install automatically**
- Version: `3.9.11`
- Click **Save**

### 6. Create Maven Job
- Click **New Item**
- Name: `prac5`
- Select **Maven project** → Click **OK**

### 7. Configure SCM
- Source Code Management → Select **Git**
- Repository URL: `https://github.com/Atharvaahirrao/pract5`
- Credentials: `--none--`
- Branch Specifier: `*/master`

### 8. Configure Build
- Scroll to **Build** section
- Root POM: `pom.xml`
- Goals and options: `clean test package`

### 9. Post Build Actions
- Click **Add post-build action**
- Select **Archive the artifacts**
- Files to archive: `target/*.jar`

### 10. Save and Build
- Click **Save** → Click **Build Now**
- Expected Console Output:

```
[INFO] Building mavenproject11 1.0-SNAPSHOT
[INFO] BUILD SUCCESS
Archiving artifacts
Finished: SUCCESS
```
