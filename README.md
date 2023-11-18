# SmartLift

## Running the project
### 1) Clone the repository
'''git clone https://github.com/vandit98/SmartLift.git'''


### 2. Install the dependencies

```
pip install -r'path/requirement.txt'
```
### 3) run the frontend
```
cd frontend/ 
npm start   
```

 the above code will start the react js \
 
 or __Frontend Deployed here__- https://vandit98.github.io/
 
###  4) running the fastapi instance

 ```
uvicorn    model:app --reload
```
  Wait for 5-10 sec let the tensorflow model to load.

Now head to the link in the terminal

 ```
http://127.0.0.1:8000/docs
```
 now you are good to go and easily click on either knee bend or arm and run the excercise. 


. 
