# Dockerfile for Streamlit Payment Screening App
# Use official Python image

# Use Python 3.11 (matches requirements.txt)
FROM python:3.11-slim

# Set work directory
WORKDIR /app


# Install system dependencies and libpostal build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    autoconf \
    automake \
    libtool \
    pkg-config \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install libpostal from source
RUN git clone https://github.com/openvenues/libpostal /opt/libpostal \
    && cd /opt/libpostal \
    && ./bootstrap.sh \
    && ./configure --datadir=/opt/libpostal_data \
    && make -j$(nproc) \
    && make install \
    && ldconfig

# Set libpostal data directory environment variable
ENV LIBPOSTAL_DATA_DIR=/opt/libpostal_data

# Ensure /usr/local/lib is in the library path
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Streamlit entrypoint
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
