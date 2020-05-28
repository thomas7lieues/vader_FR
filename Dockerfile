FROM python:3.7-alpine

# docker run -it -v $(pwd):/app container
# docker run -it --entrypoint /bin/sh -v $(pwd):/app 9dfc0662e2c4
# docker exec -it gnagnagna /bin/sh
# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
# Run the application:
# COPY chatting.py .

CMD ["python3", "chatting.py"]
