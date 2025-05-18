# 1. Base Python image
FROM python:3.12-slim

# 2. Create a non-root user
RUN useradd --create-home --shell /bin/bash vcip

# 3. Set working directory
WORKDIR /home/vcip/app

# 4. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your code & scripts
COPY . .

# 6. Make entrypoint executable & fix ownership
RUN chmod +x entrypoint.sh \
    && chown -R vcip:vcip /home/vcip/app

# 7. Switch to non-root user
USER vcip

# 8. Document Jupyter port
EXPOSE 8888

# 9. Entrypoint for setup logic
ENTRYPOINT ["./entrypoint.sh"]

# 10. Default to launching Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
