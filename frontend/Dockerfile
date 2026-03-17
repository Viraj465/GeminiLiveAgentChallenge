# Stage 1: Build React app
FROM node:20 AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .

# --- DECLARE YOUR SPECIFIC VITE VARIABLES HERE ---
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

ARG VITE_WS_BASE_URL
ENV VITE_WS_BASE_URL=$VITE_WS_BASE_URL
# -------------------------------------------------

RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html

# Cloud Run serves container traffic on port 8080.
RUN printf "server {\n\
    listen 8080;\n\
    server_name _;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location / {\n\
        try_files \$uri \$uri/ /index.html;\n\
    }\n\
}\n" > /etc/nginx/conf.d/default.conf

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]