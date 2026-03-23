# PRACTICAL 4B — Freestyle Job with Auto Scheduling

**Student:** Atharva Ahirrao | **GitHub:** https://github.com/Atharvaahirrao/pract5

---

## Aim
Configure a Freestyle Jenkins job to run automatically every minute.

## Steps

### 1. Create New Item
- Click **New Item**
- Name: `prac4b`
- Select **Freestyle project** → Click **OK**

### 2. Set Trigger
- Scroll to **Build Triggers**
- Check **Build periodically**
- Schedule: `* * * * *`

### 3. Build Step
- Scroll to **Build Steps**
- Click **Add build step**
- Select **Execute Windows batch command**
- Enter:

```batch
echo This job is running automatically
date /T
time /T
```

### 4. Save and Verify
- Click **Save** → Wait 1 minute
- Console Output shows:

```
Started by timer
This job is running automatically
Sun 03/22/2026
09:18 PM
Finished: SUCCESS
```
