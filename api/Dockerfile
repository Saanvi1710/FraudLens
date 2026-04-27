# Step 1: Use an official, lightweight Python runtime as a parent image
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file into the container
COPY requirements.txt .

# Step 4: Install the required libraries
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the API code and the machine learning model into the container
COPY main.py .
COPY xgb_paysim_model.joblib .

# Step 6: Expose the port FastAPI runs on
EXPOSE 8000

# Step 7: Define the command to run the web server when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]