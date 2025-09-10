from locust import HttpUser, TaskSet, task
import jwt

class UserBehavior(TaskSet):

    def on_start(self):
        token = jwt.encode({"sub": "test_user"}, "test-secret-123", algorithm="HS256")
        
        self.client.headers.update({"Authorization": f"Bearer {token}"})
        return super().on_start()

    @task(1)
    def job(self):
        response =  self.client.post('/api/scraping/start',
                         json={"site": "tokopedia", "query": "Thinkpad X1 Second", "max_pages": 10})
        self.job_id: super = response.json().get("job_id")
    
    @task(2)
    def get_status(self):
        self.client.get(f'/api/scraping/status/{self.job_id}')

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    