# AIStudent

An AI chatbot that has ability to show case what Terrence has learn about AI and OOP from his university, further question will be answered using web search document.

# Instruction for Deployed use

Postgres database free tier expires in a month. Once expired, recreated a new instances, then change the internal database URL in .env and render.com environment.

Render server take time to wake for the first use of each 10 minutes which might caused bad server request error.
Langsmith only work for main branch

# Instruction for local run

Start frontend:
cd .\frontend
npm run dev

Start backend:
cd .\backend
uvicorn api.main:app
