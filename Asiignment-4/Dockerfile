FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy required files into the container
COPY app.py score.py test.py requirements.txt Support_Vector_Machine_prob_final.joblib ./
COPY templates/ templates/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port (5000)
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
