# PRACTICAL 4A — Pipeline with File Archiving + Scheduled Trigger

**Student:** Atharva Ahirrao | **GitHub:** https://github.com/Atharvaahirrao/pract5

---

## Aim
Configure Jenkins Pipeline to run automatically every minute and archive a file.

## Steps

### 1. Create New Item
- Click **New Item**
- Name: `prac4a`
- Select **Pipeline** → Click **OK**

### 2. Set Build Trigger
- Scroll to **General**
- Check **Build periodically**
- Schedule:

```
* * * * *
```

### 3. Pipeline Script

```groovy
pipeline {
    agent any
    stages {
        stage('Create File') {
            steps {
                bat 'echo Hello Jenkins Pipeline > demo.txt'
            }
        }
        stage('Archive File') {
            steps {
                archiveArtifacts 'demo.txt'
            }
        }
    }
}
```

### 4. Save and Verify
- Click **Save**
- Wait 1 minute → Build triggers automatically
- Console Output first line: `Started by timer`
- Project page shows **Last Successful Artifacts** → `demo.txt`

## Cron Reference
| Schedule | Meaning |
|----------|---------|
| `* * * * *` | Every minute |
| `H/5 * * * *` | Every 5 minutes |
| `H * * * *` | Every hour |
| `0 10 * * *` | Daily at 10 AM |
