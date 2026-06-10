from locust import HttpUser, TaskSet, task
import jwt
import random
from collections import deque

# Option 1: Instance attribute with existence check
class UserBehavior1(TaskSet):
    def on_start(self):
        token = jwt.encode({"sub": "test_user"}, "test-secret-123", algorithm="HS256")
        self.client.headers.update({"Authorization": f"Bearer {token}"})
        self.job_id = None  # Initialize as None
        return super().on_start()
    
    @task(1)
    def job(self):
        response = self.client.post('/api/scraping/start',
                         json={"site": "tokopedia", "query": "Thinkpad X1 Second", "max_pages": 10})
        self.job_id = response.json().get("job_id")  # Fixed syntax
   
    @task(2)
    def get_status(self):
        if self.job_id:  # Check if job_id exists
            self.client.get(f'/api/scraping/status/{self.job_id}')
        # else: skip this task execution

# Option 2: List/queue approach - stores multiple job IDs
class UserBehavior2(TaskSet):
    def on_start(self):
        token = jwt.encode({"sub": "test_user"}, "test-secret-123", algorithm="HS256")
        self.client.headers.update({"Authorization": f"Bearer {token}"})
        self.job_ids = deque(maxlen=10)  # Keep last 10 job IDs
        return super().on_start()
    
    @task(1)
    def job(self):
        response = self.client.post('/api/scraping/start',
                         json={"site": "tokopedia", "query": "Thinkpad X1 Second", "max_pages": 10})
        job_id = response.json().get("job_id")
        if job_id:
            self.job_ids.append(job_id)
   
    @task(2)
    def get_status(self):
        if self.job_ids:  # Check if we have any job IDs
            job_id = random.choice(self.job_ids)  # Pick random job ID
            self.client.get(f'/api/scraping/status/{job_id}')

# Option 3: Class-level shared storage
class UserBehavior3(TaskSet):
    # Class-level storage shared across all instances
    shared_job_ids: deque[str] = deque(maxlen=50) # Shared among all users
    
    def on_start(self):
        token = jwt.encode({"sub": "test_user"}, "test-secret-123", algorithm="HS256")
        self.client.headers.update({"Authorization": f"Bearer {token}"})
        return super().on_start()
    
    @task(1)
    def job(self):
        response = self.client.post('/api/scraping/start',
                         json={"site": "tokopedia", "query": "Thinkpad X1 Second", "max_pages": 10})
        job_id = response.json().get("job_id")
        if job_id:
            UserBehavior3.shared_job_ids.append(job_id)
   
    @task(2)
    def get_status(self):
        if UserBehavior3.shared_job_ids:
            job_id = random.choice(UserBehavior3.shared_job_ids)
            self.client.get(f'/api/scraping/status/{job_id}')

# Option 4: Combination - Recent job priority with fallback
class UserBehavior4(TaskSet):
    shared_job_ids: deque[str] = deque(maxlen=100)
    
    def on_start(self):
        token = jwt.encode({"sub": "test_user"}, "test-secret-123", algorithm="HS256")
        self.client.headers.update({"Authorization": f"Bearer {token}"})
        self.recent_job_id = None
        return super().on_start()
    
    @task(1)
    def job(self):
        response = self.client.post('/api/scraping/start',
                         json={"site": "tokopedia", "query": "Thinkpad X1 Second", "max_pages": 10})
        job_id = response.json().get("job_id")
        if job_id:
            self.recent_job_id = job_id  # Store for this user
            UserBehavior4.shared_job_ids.append(job_id)  # Store globally
   
    @task(2)
    def get_status(self):
        # Priority: own recent job, then any shared job
        job_id = self.recent_job_id or (
            random.choice(UserBehavior4.shared_job_ids) 
            if UserBehavior4.shared_job_ids else None
        )
        if job_id:
            self.client.get(f'/api/scraping/status/{job_id}')

# Choose one of these classes for your WebsiteUser
class WebsiteUser(HttpUser):
    tasks = [UserBehavior1]  # Change to UserBehavior2, 3, or 4 as needed