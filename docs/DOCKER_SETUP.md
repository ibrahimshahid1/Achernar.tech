# Docker Setup Guide for Achernar

## Step 1: Install Docker Desktop (Windows)

1. **Download Docker Desktop**
   - Go to: https://www.docker.com/products/docker-desktop/
   - Click "Download for Windows"
   - Run the installer: `Docker Desktop Installer.exe`

2. **During Installation**
   - Enable WSL 2 features (recommended)
   - Allow it to create shortcuts
   - Restart when prompted

3. **First Launch**
   - Open Docker Desktop
   - Accept the service agreement
   - Skip tutorial (optional)
   - Wait for "Docker Desktop is running" status

4. **Verify Installation**
   ```powershell
   docker --version
   docker-compose --version
   ```

## Step 2: Start Achernar Backend Services

```powershell
# Navigate to project root
cd c:\Users\Ibrah\Documents\GitHub\archenar.tech

# Start all services
docker-compose up -d
```

**What this starts:**
- PostgreSQL (database) on port 5432
- Redis (cache/broker) on port 6379
- Django + Daphne (API) on port 8000
- Celery (task worker)

## Step 3: Run Database Migrations

```powershell
# Wait ~10 seconds for PostgreSQL to initialize, then:
docker-compose exec backend python manage.py migrate
```

**Expected output:**
```
Running migrations:
  Applying contenttypes.0001_initial... OK
  Applying auth.0001_initial... OK
  ...
```

## Step 4: Create Django Superuser (Optional)

```powershell
docker-compose exec backend python manage.py createsuperuser
```

Follow prompts to create admin account.

## Step 5: Verify Services

```powershell
# Check all containers are running
docker-compose ps
```

**Expected: All services show "Up"**

```powershell
# Check Django is responding
curl http://localhost:8000/api/v1/
```

## Common Issues & Solutions

### Issue: "Docker daemon is not running"
**Solution:** Open Docker Desktop app and wait for it to start

### Issue: Port conflicts (address already in use)
**Solution:** 
```powershell
# Find process on port 5432 (PostgreSQL)
netstat -ano | findstr :5432
taskkill /PID <PID> /F

# Or change ports in docker-compose.yml
```

### Issue: WSL 2 installation is incomplete
**Solution:**
1. Open PowerShell as Administrator
2. Run: `wsl --install`
3. Restart computer

## Stopping Services

```powershell
# Stop all services
docker-compose down

# Stop and remove volumes (nuclear option)
docker-compose down -v
```

## Viewing Logs

```powershell
# All services
docker-compose logs

# Just Django
docker-compose logs backend

# Follow logs in real-time
docker-compose logs -f backend
```

## Next Steps

Once Docker is running:
1. Backend will be at: http://localhost:8000
2. Refresh web app at: http://localhost:5173
3. App should connect successfully!
