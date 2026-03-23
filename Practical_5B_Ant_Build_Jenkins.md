# PRACTICAL 5B — Ant Project Build via Jenkins

**Student:** Atharva Ahirrao | **GitHub:** https://github.com/Atharvaahirrao/pract5

---

## Aim
Configure Jenkins to automatically build a Java Ant project — compile, test, generate JAR artifact.

---

## PART 1 — Create Ant Project in NetBeans

### 1. Create New Project
- File → New Project
- Categories: **Java with Ant**
- Projects: **Java Application**
- Name: `JavaApplication1` → Click **Finish**

> NetBeans auto-generates `build.xml` — DO NOT delete it

---

## PART 2 — Push to GitHub

### 2. Create New GitHub Repo
- Go to https://github.com → Click **+** → **New repository**
- Name: `ant-project`
- Keep Public — do NOT add README
- Click **Create repository**

### 3. Git Bash Commands

```bash
cd "C:\Users\athar\OneDrive\Documents\NetBeansProjects\JavaApplication1"
git init
git add .
git commit -m "java ant commit"
git remote add origin https://github.com/Atharvaahirrao/ant-project
git push origin -u master
```

---

## PART 3 — Configure Jenkins

### 4. Configure Ant Tool
- **Manage Jenkins** → **Tools**
- Scroll to **Ant installations** → Click **Add Ant**
- Name: `ant`
- Check **Install automatically**
- Version: `1.10.15`
- Click **Save**

### 5. Create Freestyle Job
- New Item → Name: `prac5b`
- Select **Freestyle project** → Click **OK**

### 6. Configure SCM
- Source Code Management → **Git**
- URL: `https://github.com/Atharvaahirrao/ant-project`
- Branch: `*/master`

### 7. Build Step
- Build Steps → **Add build step** → **Invoke Ant**
- Ant Version: `ant`
- Targets: `clean jar`

### 8. Post Build
- Add post-build action → **Archive the artifacts**
- Files: `dist/*.jar`
- Click **Save** → Click **Build Now**

> Expected: `Finished: SUCCESS` with JAR archived
