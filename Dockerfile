FROM python:3.11
WORKDIR /code
COPY . /code
RUN pip install -r requirements.txt
RUN pip install -r requirements1.txt
RUN pip install -r requirements3.txt
CMD ["uvicorn","main:app","--reload","--host","0.0.0.0","--port","8000"]


# Install necessary dependencies for cv2 module
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev
