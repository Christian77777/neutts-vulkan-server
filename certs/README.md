# SSL / TLS Certificates Directory

Place your SSL/TLS certificates and private keys here to enable HTTPS on the NeuTTS server.

## Quick Start with Self-Signed Certificate (for local development or LAN testing)

Run the following OpenSSL command to generate a self-signed certificate:

```bash
openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -sha256 -days 365 -nodes \
    -subj "/C=US/ST=State/L=City/O=Local/CN=localhost"
```

## Usage with Docker Compose

1. Put your certificate file (`cert.pem`) and private key file (`key.pem`) into this directory.
2. In `docker-compose.yml` (or `docker-compose.nvidia.yml`), uncomment the volume mount:
   ```yaml
   volumes:
     - ./certs:/app/certs:ro
   ```
3. Uncomment the environment variables:
   ```yaml
   environment:
     - SSL_KEYFILE=/app/certs/key.pem
     - SSL_CERTFILE=/app/certs/cert.pem
     # Optional if your private key is encrypted:
     # - SSL_KEYFILE_PASSWORD=your-key-password
     # Optional CA certificate chain:
     # - SSL_CA_CERTS=/app/certs/ca.pem
   ```
4. Restart the container:
   ```bash
   docker compose up -d
   ```

## Git Handling

Certificate and key files (`*.pem`, `*.key`, `*.crt`) in this folder are ignored by `.gitignore`. Never commit private keys to version control.
