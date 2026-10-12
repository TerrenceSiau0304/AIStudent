# AIStudent

AIStudent is an AI-powered chatbot designed to showcase Terrence's understanding of artificial intelligence and object-oriented programming through interactive conversations. The system is built to answer questions related to AI concepts, OOP principles, and university learning topics, while using web-based document search to support more complete answers when needed.

## Overview

This project combines a modern frontend interface with a backend service that processes user questions, retrieves relevant information, and responds in a clear and educational way. It is intended to act as both a learning assistant and a demonstration project for AI-driven conversational experiences.

## Features

- AI chatbot interface for Q&A conversations
- Educational focus on AI and OOP concepts
- Web-search-backed answer support for additional context
- Separate frontend and backend services for modular development
- Deployment-ready configuration for hosted environments

## Project Structure

- `frontend/` - React frontend for the chat experience
- `backend/` - FastAPI backend for API logic and AI processing
- `data/` - Local persistence and application data

## Why This Project Exists

AIStudent is more than a simple chatbot. It reflects a university learning journey in AI and software engineering, turning those concepts into a practical application that users can interact with in real time. The project demonstrates how AI systems can be designed to explain fundamentals, answer follow-up questions, and support learning through accessible conversations.

## Getting Started

For local development, use the commands in the local run section below. For deployment, follow the production notes in the deployed section below.

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
