# tc-skills-extraction-service
FastAPI Python service for extracting skills from text

## 🐳 Local Docker build & run

To build and run the Skills Extraction Service locally:

### Build the image
```bash
docker build -t tc-skills:local .
```

### Run the container (connecting to local Talent Catalog API)
```bash
docker run --rm -p 8000:8000 \
  -e SKILLS_BASE_URL=http://host.docker.internal:8080/api/public/skill/names \
  tc-skills:local
```
